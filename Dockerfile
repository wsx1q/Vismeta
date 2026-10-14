# syntax=docker/dockerfile:1
# 两个目标：
#   cpu：网页服务 + SAM（CPU，ONNX），不带 torch，约 200 MB
#   gpu：再加 CUDA 版 torch + SAM3，以及自己编译的 llama-server（复用 torch 带的 CUDA 库，
#        大模型容器也用这个镜像，不用再拉 llama.cpp 官方镜像）
#        SAM3 和 llama.cpp 的源码从本地 vendor/ 复制，先运行 scripts/fetch-vendor.sh（deploy.sh 会自动运行）
# 镜像里只有代码，模型权重和模型路径都在运行时挂载、配置。

ARG PY_IMAGE=python:3.10-slim-bookworm

# ---------- 公共底座 ----------
FROM ${PY_IMAGE} AS base
ARG PIP_INDEX=https://pypi.org/simple
ARG APT_MIRROR=
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    HOME=/tmp \
    LABEL_DATA=/data \
    LABEL_PORT=8088
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s#deb.debian.org#${APT_MIRROR}#g" /etc/apt/sources.list.d/debian.sources; fi
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
WORKDIR /app
EXPOSE 8088
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('LABEL_PORT','8088'), timeout=4)"
CMD ["python", "server.py"]

# ---------- CPU 版 ----------
FROM base AS cpu
COPY *.py /app/
COPY static /app/static

# ---------- 编译 llama-server（只在 GPU 版用到，产物是一个文件）----------
# 和运行镜像同一个 Debian 底座，只从 NVIDIA 的 apt 源装编译要用的几个 CUDA 12.4 包，不用几 GB 的 CUDA 开发镜像。
FROM ${PY_IMAGE} AS llama-build
ARG APT_MIRROR=
ARG CUDA_APT_REPO=https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64
# 显卡架构：75 图灵(20系/T4) 80 A100 86 安培(30系/A10) 89 Ada(40系/L4) 90 H100；只填自己的卡编译更快
ARG CUDA_ARCHS=75;80;86;89;90
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s#deb.debian.org#${APT_MIRROR}#g" /etc/apt/sources.list.d/debian.sources; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg cmake make g++ \
 && curl -fsSL ${CUDA_APT_REPO}/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cuda.gpg] ${CUDA_APT_REPO}/ /" > /etc/apt/sources.list.d/cuda.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvcc-12-4 cuda-cudart-dev-12-4 cuda-cccl-12-4 libcublas-dev-12-4 \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda-12.4/bin:${PATH}
# llama.cpp 源码来自本地 vendor/（scripts/fetch-vendor.sh 准备），构建时不访问 GitHub；不编译网页界面，只用它的 API。
COPY vendor/llama.cpp /src
RUN cd /src \
 && cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --target llama-server -j"$(nproc)" \
 && mkdir -p /out && cp build/bin/llama-server /out/ && strip /out/llama-server

# ---------- GPU 版 ----------
FROM base AS gpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
# gcc / libc6-dev：triton 运行时编译 SAM3 的后处理内核要用
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
# 安装和瘦身必须在同一层：删掉 SAM3 推理用不到的 CUDA 库和头文件（约 750 MB）。
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url ${TORCH_INDEX} --extra-index-url ${PIP_INDEX} \
 && pip install -r /tmp/requirements-gpu.txt \
 && cd /usr/local/lib/python3.10/site-packages \
 && rm -rf nvidia/cusolver torch/lib/libtorch_cuda_linalg.so \
      nvidia/cudnn/lib/libcudnn_adv.so.9 nvidia/cudnn/lib/libcudnn_ops.so.9 nvidia/cudnn/lib/libcudnn_cnn.so.9 \
      nvidia/cuda_cupti/lib/libcheckpoint.so nvidia/cuda_cupti/lib/libnvperf_host.so \
      nvidia/cuda_cupti/lib/libnvperf_target.so nvidia/cuda_cupti/lib/libpcsamplingutil.so \
      torch/include triton/backends/amd nvidia/*/include \
 && find . -name "*.a" -delete
# SAM3 只装代码，源码来自本地 vendor/sam3（已打好 sam3.patch：修正点选后 bfloat16 / float 混用报错，
# 并把文字分割结果及时移到 CPU 释放显存）。权重在运行时挂载。
COPY vendor/sam3 /opt/sam3
RUN pip install --no-deps -e /opt/sam3
# llama-server 用 torch 自带的 cudart / cublas
COPY --from=llama-build /out/llama-server /opt/llama/llama-server
ENV LLAMA_LD_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib
COPY *.py /app/
COPY static /app/static
