#!/usr/bin/env bash
# 一键部署：./deploy.sh [auto|cpu|gpu]   默认 auto，有可用的 GPU 容器环境就用 GPU。
# 其他命令：./deploy.sh down | logs | ps | restart
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m错误：\033[0m%s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "没有安装 Docker。Ubuntu 上可以执行：curl -fsSL https://get.docker.com | sh"
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null || [ -t 0 ]; then
    DOCKER=(sudo docker)
    "${DOCKER[@]}" info >/dev/null || die "Docker 没有运行"
  else
    die "当前用户不能访问 Docker：sudo usermod -aG docker \$USER 后重新登录"
  fi
fi
"${DOCKER[@]}" compose version >/dev/null 2>&1 || die "缺少 docker compose 插件：sudo apt install docker-compose-plugin"

if [ ! -f .env ]; then
  cp .env.example .env
  say "已生成 .env，可按需修改后重新运行"
fi
set_env() {  # set_env KEY VALUE：只在 .env 里这一项为空时写入
  if grep -qE "^$1=$" .env; then sed -i "s#^$1=\$#$1=$2#" .env; fi
}
set_env LABEL_UID "${SUDO_UID:-$(id -u)}"
set_env LABEL_GID "${SUDO_GID:-$(id -g)}"
envget() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }
for key in VLM_MODEL VLM_MMPROJ VLM_LORA VLM_MODEL_DIR VLM_LORA_DIR VLM_URL LABEL_DATA_DIR LABEL_MODELS_DIR LABEL_PORT SAM3_WEIGHTS SAM1_WEIGHTS_DIR BUILD_PROXY GITHUB_PREFIX; do
  printf -v "$key" '%s' "$(envget "$key")"
done
DATA_DIR="${LABEL_DATA_DIR:-./data}"
# 权重目录没单独指定时放在数据目录下；导出给 docker compose 用（命令行环境变量优先于 .env）。
export LABEL_MODELS_DIR="${LABEL_MODELS_DIR:-$DATA_DIR/models}"

gpu_ready() {
  command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 || return 1
  "${DOCKER[@]}" info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}

ACTION="${1:-auto}"
MODE="$ACTION"
case "$ACTION" in
  down|logs|ps|restart)
    MODE="$(cat .deploy-mode 2>/dev/null || echo cpu)" ;;
  auto)
    if gpu_ready; then MODE=gpu; else MODE=cpu; fi ;;
  gpu)
    gpu_ready || die "GPU 容器环境不可用。需要 NVIDIA 驱动 + NVIDIA Container Toolkit（并执行 nvidia-ctk runtime configure），见 DEPLOY.md「GPU 环境准备」" ;;
  cpu) ;;
  *) die "用法：./deploy.sh [auto|cpu|gpu|down|logs|ps|restart]" ;;
esac

FILES=(-f docker-compose.yml)
PROFILES=()
WITH_VLM=""
if [ "$MODE" = gpu ]; then
  FILES+=(-f docker-compose.gpu.yml)
  if [ -n "${VLM_MODEL:-}" ]; then
    WITH_VLM=1
    PROFILES=(--profile vlm)
    # 没有另外指定外部接口时，默认接口指向本地 vlm 容器。
    export VLM_DEFAULT_URL="http://vlm:8080/v1"
  fi
fi
COMPOSE=("${DOCKER[@]}" compose "${FILES[@]}" "${PROFILES[@]}")

case "$ACTION" in
  down) exec "${COMPOSE[@]}" down --remove-orphans ;;
  logs) exec "${COMPOSE[@]}" logs -f --tail 100 ;;
  ps) exec "${COMPOSE[@]}" ps ;;
  restart) exec "${COMPOSE[@]}" restart ;;
esac

mkdir -p "$DATA_DIR/projects" "$LABEL_MODELS_DIR"
# 用 sudo 运行时目录会是 root 的，容器里的普通用户写不进去，交还给 LABEL_UID。
if [ "$(id -u)" = 0 ]; then
  chown "$(envget LABEL_UID):$(envget LABEL_GID)" "$DATA_DIR" "$DATA_DIR/projects" "$LABEL_MODELS_DIR"
fi
if [ -n "$WITH_VLM" ]; then
  [ -n "${VLM_MODEL_DIR:-}" ] || die "填了 VLM_MODEL，还要填模型所在目录 VLM_MODEL_DIR"
  [ -f "$VLM_MODEL_DIR/$VLM_MODEL" ] || die "找不到大模型文件 $VLM_MODEL_DIR/$VLM_MODEL"
  [ -z "${VLM_MMPROJ:-}" ] || [ -f "$VLM_MODEL_DIR/$VLM_MMPROJ" ] || die "找不到 mmproj 文件 $VLM_MODEL_DIR/$VLM_MMPROJ"
  if [ -n "${VLM_LORA:-}" ]; then
    [ -n "${VLM_LORA_DIR:-}" ] || die "填了 VLM_LORA，还要填 LoRA 所在目录 VLM_LORA_DIR"
    [ -f "$VLM_LORA_DIR/$VLM_LORA" ] || die "找不到 LoRA 文件 $VLM_LORA_DIR/$VLM_LORA"
  fi
  export VLM_LORA_DIR="${VLM_LORA_DIR:-$VLM_MODEL_DIR}"
elif [ "$MODE" = cpu ] && [ -n "${VLM_MODEL:-}" ]; then
  say "CPU 版不启动本地大模型（VLM_MODEL 只在 GPU 版生效）；可以用 VLM_URL 或网页「大模型设置」接外部服务"
fi

if [ "$MODE" = gpu ]; then
  # SAM3、llama.cpp 源码放在 vendor/，只在缺失或版本变化时下载一次。
  https_proxy="${BUILD_PROXY:-${https_proxy:-}}" GITHUB_PREFIX="${GITHUB_PREFIX:-}" bash scripts/fetch-vendor.sh     || die "下载 vendor 源码失败；可以设 BUILD_PROXY 或 GITHUB_PREFIX 后重试，或手动放到 vendor/"
fi

say "部署模式：$MODE$([ -n "$WITH_VLM" ] && echo "，含本地大模型 $VLM_MODEL")"
"${COMPOSE[@]}" up -d --build --remove-orphans
echo "$MODE" > .deploy-mode

say "等待服务启动…"
PORT="${LABEL_PORT:-8088}"
for _ in $(seq 1 90); do
  if curl -fs "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    say "启动完成：http://${IP:-127.0.0.1}:$PORT"
    SAM3_FILE="$LABEL_MODELS_DIR/${SAM3_WEIGHTS:-sam3.pt}"
    SAM1_PATH="$LABEL_MODELS_DIR/${SAM1_WEIGHTS_DIR:-sam1}"
    if [ "$MODE" = gpu ] && [ ! -f "$SAM3_FILE" ]; then echo "   提示：没有 SAM3 权重（$SAM3_FILE）"; fi
    [ -f "$SAM1_PATH/encoder.onnx" ] || echo "   提示：没有 SAM（CPU）权重（$SAM1_PATH/encoder.onnx、decoder.onnx）"
    [ -n "$WITH_VLM" ] && echo "   大模型容器加载模型需要一点时间：./deploy.sh ps 看到 vlm 为 healthy 即可用"
    echo "   查看日志：./deploy.sh logs    停止：./deploy.sh down"
    exit 0
  fi
  sleep 2
done
die "服务没有在 3 分钟内就绪，看看日志：./deploy.sh logs"
