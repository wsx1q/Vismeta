"""运行配置，全部来自环境变量，方便本机直接跑和 Docker 部署共用一套代码。"""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# 数据目录：项目、图片、标注、标签、大模型配置都在这里。
DATA_ROOT = Path(os.environ.get("LABEL_DATA") or APP_DIR / "data")
MODELS_DIR = Path(os.environ.get("LABEL_MODELS") or DATA_ROOT / "models")
SAM3_CKPT = Path(os.environ.get("SAM3_CKPT") or MODELS_DIR / "sam3.pt")
SAM1_DIR = Path(os.environ.get("SAM1_DIR") or MODELS_DIR / "sam1")
PACKS_DIR = MODELS_DIR / "packs"

HOST = os.environ.get("LABEL_HOST", "0.0.0.0")
PORT = int(os.environ.get("LABEL_PORT", "8088"))

# auto：有 CUDA 就用 GPU；cpu / cuda 强制指定。
DEVICE = os.environ.get("LABEL_DEVICE", "auto").strip().lower()
# SAM3：auto 只在 GPU 上启用（CPU 上太慢）；on / off 强制。
SAM3 = os.environ.get("SAM3", "auto").strip().lower()
# Release image features after inactivity; keep model weights resident. Zero disables it.
SAM_CACHE_IDLE_SECONDS = max(0, float(os.environ.get("SAM_CACHE_IDLE_SECONDS", "120")))

# 第一次启动、还没有 vlm.json 时，用这里的地址生成默认大模型接口。
VLM_URL = os.environ.get("VLM_URL", "").strip()
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "").strip()
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
VLM_IMAGE_SIDE = int(os.environ.get("VLM_IMAGE_SIDE", "512"))


def torch_device():
    if DEVICE == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if DEVICE.startswith("cuda") or torch.cuda.is_available():
        return DEVICE if DEVICE.startswith("cuda:") else "cuda:0"
    return "cpu"


def sam3_enabled(device):
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    if SAM3 in {"on", "1", "true", "yes"}:
        return True
    if SAM3 in {"off", "0", "false", "no"}:
        return False
    return device.startswith("cuda")
