# Docker 部署详细说明（Ubuntu）

快速上手见 [README](../README.md#docker-一键部署)，这里是完整说明。

| | CPU 版 | GPU 版 |
|---|---|---|
| 适合 | 没有 N 卡，或只做手画框、审核文字 | 有 NVIDIA 显卡 |
| 分割模型 | SAM（CPU，ONNX），点一下约 4 秒 | SAM3（GPU，点选 / 框选 / 文字，约 0.5 秒）+ SAM（CPU） |
| 多模态大模型 | 不带；可在网页「大模型设置」或 `VLM_URL` 接外部 OpenAI 兼容服务 | 可选附带 llama.cpp 本地大模型（Qwen3-VL-8B Q4 一张图约 0.5–1 秒），也可接外部服务 |
| 镜像 | `vismeta:cpu`，不含 torch | `vismeta:gpu` 一个镜像，标注服务和大模型容器共用 |

镜像按需精简：CPU 版不装 torch；GPU 版删掉了 SAM3 推理用不到的 CUDA 库（cuSOLVER、cuDNN 的 adv/ops/cnn 子库、头文件等），llama-server 从源码编译后放进同一个镜像，直接用 torch 自带的 CUDA 运行库，不再单独下载 llama.cpp 官方镜像。

GPU 版要用的 SAM3 和 llama.cpp 源码（固定版本）放在本地 `vendor/` 目录，构建时直接复制进镜像，不再每次访问 GitHub。`deploy.sh` 只在 `vendor/` 缺失或版本号变化时调用 `scripts/fetch-vendor.sh` 下载一次；无法联网的机器可以把别处准备好的 `vendor/` 整个拷过来。

镜像里只有代码。模型权重、权重文件名、大模型文件、LoRA、接口地址都在 `.env` 里配置，运行时挂载进容器，改完重新执行 `./deploy.sh` 即可，不需要重新构建镜像。

## 快速开始

```bash
git clone https://github.com/wsx1q/Vismeta.git && cd Vismeta
./deploy.sh          # 自动判断：GPU 容器环境可用就用 GPU 版，否则 CPU 版
./deploy.sh cpu      # 强制 CPU 版
./deploy.sh gpu      # 强制 GPU 版
```

第一次运行会生成 `.env`，按需修改后再运行一次，然后打开 `http://服务器IP:8088`。

```bash
./deploy.sh logs      # 看日志
./deploy.sh ps        # 看状态
./deploy.sh restart   # 重启
./deploy.sh down      # 停止并删除容器（数据和模型目录不受影响）
```

当前用户不在 docker 组时，deploy.sh 会自动用 sudo。

## 目录和模型

```
LABEL_DATA_DIR（默认 ./data）       项目、图片、标注、标签、vlm.json（大模型设置）
LABEL_MODELS_DIR（默认 数据目录/models）
├── sam3.pt                       SAM3 权重，文件名可用 SAM3_WEIGHTS 改
├── sam1/                         SAM（CPU）ONNX，目录名可用 SAM1_WEIGHTS_DIR 改
│   ├── encoder.onnx
│   └── decoder.onnx
└── packs/                        自定义模型清单（可选）
```

- **SAM3**：在 Hugging Face [facebook/sam3](https://huggingface.co/facebook/sam3) 申请访问后下载 `sam3.pt`。SAM3 代码在准备 `vendor/` 时会打上 `sam3.patch`（修正点选后 bfloat16 / float 混用的报错，并及时释放文字分割的显存）。
- **SAM（CPU）**：X-AnyLabeling 发布的 SAM ViT-B 量化 ONNX，下载后改名放进 `sam1/`：
  ```bash
  curl -L -o data/models/sam1/encoder.onnx https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/sam_vit_b_01ec64.encoder.quant.onnx
  curl -L -o data/models/sam1/decoder.onnx https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.2.0/sam_vit_b_01ec64.decoder.quant.onnx
  ```
- 没有权重也能启动，对应的模型显示「缺权重」，手画矩形 / 多边形和大模型功能不受影响。

## 多模态大模型

**接外部服务（CPU、GPU 版都可以）**：在网页「大模型设置」里添加接口，或者在 `.env` 里填写首次启动时的默认接口：

```ini
VLM_URL=http://10.0.0.5:8000/v1
VLM_URL_MODEL=Qwen2.5-VL-7B-Instruct
VLM_API_KEY=
```

**GPU 版附带本地大模型**：填好 GGUF 文件，deploy.sh 会多启动一个 llama.cpp 容器，默认接口自动指向它（`http://vlm:8080/v1`）：

```ini
VLM_MODEL_DIR=/data/models/Qwen3-VL-8B           # 宿主机目录
VLM_MODEL=Qwen3VL-8B-Instruct-Q4_K_M.gguf
VLM_MMPROJ=mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf  # 视觉投影，多模态模型必填
VLM_LORA_DIR=/data/models/lora                    # 可选
VLM_LORA=full_lora.gguf
VLM_ALIAS=Qwen3-VL-8B-LoRA
VLM_GPU=1                                         # 和 SAM3（WEB_GPU）分开放更稳
VLM_EXTRA_ARGS=--flash-attn on
```

默认接口只在第一次启动（数据目录里还没有 `vlm.json`）时生成，之后以网页「大模型设置」为准。

显存参考：SAM3 常驻约 3.5 GB，文字分割时峰值约 5 GB；Qwen3-VL-8B Q4 加 mmproj 约 6.5 GB。10 GB 的卡上两者最好分开放。

## GPU 环境准备

需要 NVIDIA 驱动和 NVIDIA Container Toolkit。国内可以用中科大镜像：

```bash
nvidia-smi    # 能看到显卡说明驱动正常

curl -fsSL -o /tmp/nv.gpgkey https://mirrors.ustc.edu.cn/libnvidia-container/gpgkey
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg /tmp/nv.gpgkey
curl -fsSL https://mirrors.ustc.edu.cn/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://nvidia.github.io#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://mirrors.ustc.edu.cn#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl reload docker      # reload 即可，不会重启已有容器

docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ubuntu nvidia-smi -L   # 验证
```

compose 里用的是 `runtime: nvidia`，所以只需要上面的 reload；不需要 `systemctl restart docker`，机器上已有的容器不会受影响。

## 国内网络

`.env` 末尾的「构建镜像」一节可以按需打开。各镜像站速度差别很大，建议先在服务器上测一下。实测可用的组合：

- PyPI：中科大 `PIP_INDEX=https://mirrors.ustc.edu.cn/pypi/simple`
- pytorch.org：直连
- GitHub（只在第一次准备 `vendor/` 时用到）：走代理 `BUILD_PROXY`，或设 `GITHUB_PREFIX=https://ghfast.top/`。代理只在你电脑上时，可以先 `ssh -R 7897:127.0.0.1:7897 用户@服务器` 转发过去，再设 `BUILD_NETWORK=host`
- 编译 llama-server 要从 NVIDIA 的 apt 源装 CUDA 编译器（只在构建阶段，不进最终镜像），国内用 `CUDA_APT_REPO=https://developer.download.nvidia.cn/compute/cuda/repos/debian12/x86_64`
- `CUDA_ARCHS` 只填自己显卡的架构（如 30 系填 `86`），编译快很多
- Docker Hub：在 `/etc/docker/daemon.json` 的 `registry-mirrors` 里配置

## 从本机直接运行迁移过来

原来的数据目录可以直接挂进容器（`LABEL_DATA_DIR=/home/ubuntu/label-data`），项目路径会按新位置自动识别。迁移前先停掉原来的进程，两边不要同时写同一个数据目录。

## 不用 Docker 直接运行

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # CPU 用 .../whl/cpu
pip install -r requirements.txt -r requirements-gpu.txt    # CPU 只装 requirements.txt，不需要 torch
bash scripts/fetch-vendor.sh && pip install --no-deps -e vendor/sam3
LABEL_DATA=/path/to/data python server.py
```

环境变量：`LABEL_DATA` 数据目录、`LABEL_MODELS` 权重目录、`SAM3_CKPT`、`SAM1_DIR`、`LABEL_PORT`、`LABEL_DEVICE`（auto / cpu / cuda）、`SAM3`（auto / on / off）、`VLM_URL` / `VLM_MODEL_NAME` / `VLM_API_KEY` 默认大模型接口。
