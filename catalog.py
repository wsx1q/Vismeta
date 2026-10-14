import json
from pathlib import Path

import config

PACKS_DIR = config.PACKS_DIR
SAM3_CKPT = config.SAM3_CKPT
SAM1_DIR = config.SAM1_DIR
SAM3_ON = config.sam3_enabled(config.torch_device())
RUNTIMES = {"onnx-sam1", "pytorch-sam3", "tensorrt-sam"}


def _confidence(spec):
    raw = spec or {}
    try:
        default = float(raw.get("default", 0.4))
    except (TypeError, ValueError):
        default = 0.4
    return {
        "min": 0,
        "max": 1,
        "step": 0.05,
        "default": min(1.0, max(0.0, default)),
    }


def missing_message(spec):
    if spec.get("runtime") == "pytorch-sam3" and not SAM3_ON:
        return "SAM3 需要 GPU（当前是 CPU 模式）"
    missing = []
    for key, path in (spec.get("files") or {}).items():
        if not Path(path).is_file():
            missing.append(key + " 放到 " + path)
    if not missing:
        return "模型文件还没放好"
    return "缺少权重：" + "；".join(missing)


def _files_ready(files):
    if not files:
        return False
    return all(Path(path).is_file() for path in files.values())


def builtin_models():
    sam1_files = {
        "encoder": str(SAM1_DIR / "encoder.onnx"),
        "decoder": str(SAM1_DIR / "decoder.onnx"),
    }
    sam3_files = {"checkpoint": str(SAM3_CKPT)}
    return [
        {
            "id": "sam1-onnx",
            "name": "SAM（CPU）",
            "runtime": "onnx-sam1",
            "device": "cpu",
            "prompts": ["point", "box"],
            "outputs": ["mask", "box"],
            "confidence": _confidence({"default": 0.5}),
            "files": sam1_files,
            "ready": _files_ready(sam1_files),
            "hint": "左键正点，右键负点，或拖一个框。适合只有 CPU 的机器。",
            "builtin": True,
        },
        {
            "id": "sam3",
            "name": "SAM3",
            "runtime": "pytorch-sam3",
            "device": "cuda",
            "prompts": ["point", "box", "text"],
            "outputs": ["mask", "box"],
            "confidence": _confidence({"default": 0.4}),
            "files": sam3_files,
            "ready": SAM3_ON and SAM3_CKPT.is_file(),
            "hint": "点和框之外，还可以用文字找出这类物体。" if SAM3_ON else "SAM3 需要 GPU，用 GPU 版部署后可用。",
            "builtin": True,
        },
    ]


def validate_pack(data):
    if not isinstance(data, dict):
        raise ValueError("模型清单必须是 JSON 对象")
    mid = str(data.get("id") or "").strip()
    name = str(data.get("name") or "").strip()[:40]
    runtime = str(data.get("runtime") or "").strip()
    if not mid or not mid.replace("-", "").replace("_", "").isalnum():
        raise ValueError("id 只能用字母、数字、下划线和短横线")
    if mid in {"sam1-onnx", "sam3", "sam3-trt"}:
        raise ValueError("这个 id 是内置模型")
    if not name:
        raise ValueError("请填写显示名称")
    if runtime not in RUNTIMES:
        raise ValueError("runtime 只能是 " + "、".join(sorted(RUNTIMES)))
    prompts = [item for item in (data.get("prompts") or []) if item in {"point", "box", "text"}]
    outputs = [item for item in (data.get("outputs") or []) if item in {"mask", "box"}]
    if not prompts or not outputs:
        raise ValueError("prompts 需要 point/box/text，outputs 需要 mask/box")
    files = data.get("files") or {}
    if not isinstance(files, dict) or not files:
        raise ValueError("files 里要写权重的绝对路径")
    clean_files = {}
    for key, value in files.items():
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise ValueError(key + " 必须是绝对路径")
        clean_files[key] = str(path)
    device = "cpu" if data.get("device") == "cpu" else "cuda"
    return {
        "id": mid,
        "name": name,
        "runtime": runtime,
        "device": device,
        "prompts": prompts,
        "outputs": outputs,
        "confidence": _confidence(data.get("confidence")),
        "files": clean_files,
        "ready": _files_ready(clean_files),
        "hint": str(data.get("hint") or "")[:120],
        "builtin": False,
    }


def custom_models():
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(PACKS_DIR.glob("*.json")):
        try:
            rows.append(validate_pack(json.loads(path.read_text(encoding="utf-8"))))
        except Exception as exc:
            rows.append({
                "id": path.stem,
                "name": path.stem,
                "runtime": "",
                "device": "",
                "prompts": [],
                "outputs": [],
                "confidence": _confidence({}),
                "files": {},
                "ready": False,
                "hint": "清单无效：" + str(exc),
                "builtin": False,
            })
    return rows


def list_models():
    return builtin_models() + custom_models()


def get_model(mid):
    for item in list_models():
        if item["id"] == mid:
            return item
    return None


def save_pack(data):
    pack = validate_pack(data)
    PACKS_DIR.mkdir(parents=True, exist_ok=True)
    target = PACKS_DIR / (pack["id"] + ".json")
    target.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return pack
