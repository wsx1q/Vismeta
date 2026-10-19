import gc
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
try:
    import torch
except ImportError:  # CPU 版镜像不带 torch，只能用 ONNX 模型和手画
    torch = None
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image

import config
from memory_utils import trim_cpu_memory

DATA_ROOT = config.DATA_ROOT
PROJECTS_FILE = DATA_ROOT / "projects.json"
PROJECTS_DIR = DATA_ROOT / "projects"
SAM3_CKPT = config.SAM3_CKPT
SAM_DEVICE = config.torch_device()
SAM3_ON = config.sam3_enabled(SAM_DEVICE)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder="static")
# 本地导入一次可能带很多文件；放宽 multipart 限制，避免 413。
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024
app.config["MAX_FORM_PARTS"] = 20000
app.config["MAX_FORM_MEMORY_SIZE"] = 64 * 1024 * 1024
LOCK = threading.Lock()

STATE = {
    "model": None,
    "sam3_model": None,
    "sam3_processor": None,
    "image_key": None,
    "sam3_error": None,
}


@app.after_request
def reclaim_inference_memory(response):
    # View frames and their large image arrays have been released at this point.
    if request.endpoint == "segment":
        trim_cpu_memory()
    return response


def load_projects():
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROJECTS_FILE.is_file():
        save_projects([])
        return []
    data = json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
    rows = [item for item in (data.get("projects") or []) if item.get("id") != "default"]
    if rows != (data.get("projects") or []):
        save_projects(rows)
    for item in rows:
        # 项目目录跟着数据目录走：数据目录换了挂载位置（比如进了容器）也能找到。
        root = PROJECTS_DIR / item["id"]
        if root.is_dir():
            item["images"] = str(root / "images")
            item["annotations"] = str(root / "annotations")
            item["labels"] = str(root / "labels.json")
    return rows


def save_projects(rows):
    PROJECTS_FILE.write_text(json.dumps({"projects": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def find_project(pid):
    for item in load_projects():
        if item.get("id") == pid:
            return item
    return None


def current_project():
    pid = (request.headers.get("X-Project") or request.args.get("project") or "default").strip()
    item = find_project(pid)
    if not item:
        abort(404, "项目不存在")
    return item


def read_doc(path):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and ("shapes" in doc or "imageData" in doc):
            # Images are separate files. Never send, re-save, or export embedded pixels.
            doc["imageData"] = None
        return doc
    except Exception:
        return None


def scan_annotations(ann_root):
    """返回 {stem: {"labeled": bool, "text": none|pending|approved|rejected}}"""
    import texts
    found = {}
    if not ann_root.is_dir():
        return found
    for path in ann_root.glob("*.json"):
        data = read_doc(path)
        if data is None:
            continue
        found[path.stem] = {
            "labeled": bool(data.get("shapes")),
            "text": texts.text_status(data),
        }
    return found


def labeled_stems(ann_root):
    return {stem for stem, info in scan_annotations(ann_root).items() if info["labeled"]}


def project_stats(item):
    images = Path(item["images"])
    anns = Path(item["annotations"])
    files = []
    if images.is_dir():
        files = [p for p in images.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    annotated = 0
    shapes = 0
    for path in files:
        target = anns / (path.stem + ".json")
        if not target.is_file():
            continue
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            continue
        count = len(data.get("shapes") or [])
        if count:
            annotated += 1
            shapes += count
    return {"images": len(files), "annotated": annotated, "shapes": shapes}


def unique_dest(root, name):
    dest = root / name
    if not dest.exists():
        return dest
    stem, ext = Path(name).stem, Path(name).suffix
    n = 2
    while True:
        cand = root / f"{stem}_{n}{ext}"
        if not cand.exists():
            return cand
        n += 1


def copy_image(src, root):
    dest = unique_dest(root, src.name)
    if src.resolve() == dest.resolve():
        return None
    shutil.copy2(src, dest)
    return dest.name


def safe_image(name):
    filename = Path(str(name)).name
    if not filename or Path(filename).suffix.lower() not in EXTS:
        abort(400)
    root = Path(current_project()["images"]).resolve()
    path = (root / filename).resolve()
    if path.parent != root or not path.is_file():
        abort(404)
    return path


def ann_path(image_path):
    root = Path(current_project()["annotations"])
    root.mkdir(parents=True, exist_ok=True)
    return root / (image_path.stem + ".json")


def polygons_from_mask(mask):
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[0]
    binary = (values > 0).astype(np.uint8)
    if binary.max() == 0:
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue
        epsilon = max(1.0, 0.0015 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [[int(x), int(y)] for x, y in approx.reshape(-1, 2)]
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def box_from_mask(mask):
    binary = np.asarray(mask)
    if binary.ndim == 3:
        binary = binary[0]
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def release_cuda():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def clone_tensors(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tensors(item) for item in value)
    return value


def free_models():
    STATE["sam3_model"] = None
    STATE["sam3_processor"] = None
    STATE["sam3_state"] = None
    STATE["sam3_text_state"] = None
    STATE["sam3_text_key"] = None
    STATE["sam3_epoch"] = 0
    STATE["image_key"] = None
    STATE["model"] = None
    release_cuda()


def load_sam3():
    if STATE["model"] == "sam3" and STATE["sam3_processor"] is not None:
        return
    if not SAM3_ON:
        STATE["sam3_error"] = "SAM3 需要 GPU（当前是 CPU 模式）"
        raise RuntimeError(STATE["sam3_error"])
    free_models()
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        STATE["sam3_error"] = "SAM3 代码没装好：" + str(exc)
        raise RuntimeError(STATE["sam3_error"]) from exc
    if not SAM3_CKPT.is_file():
        message = "把 SAM3 权重放到 " + str(SAM3_CKPT)
        STATE["sam3_error"] = message
        free_models()
        raise RuntimeError(message)
    try:
        if SAM_DEVICE.startswith("cuda"):
            torch.cuda.set_device(torch.device(SAM_DEVICE))
        model = build_sam3_image_model(
            device=SAM_DEVICE,
            enable_inst_interactivity=True,
            checkpoint_path=str(SAM3_CKPT),
            load_from_HF=False,
        )
        model.to(SAM_DEVICE)
        model.eval()
    except Exception as exc:
        message = "SAM3 权重读取失败：" + str(exc)
        STATE["sam3_error"] = message
        free_models()
        raise RuntimeError(message) from exc
    STATE["sam3_model"] = model
    STATE["sam3_processor"] = Sam3Processor(model, device=SAM_DEVICE, confidence_threshold=0.4)
    STATE["model"] = "sam3"
    STATE["sam3_error"] = None


def ensure(model_name):
    if model_name in {"sam3", "pytorch-sam3"}:
        load_sam3()
    else:
        raise RuntimeError("这个运行时不用 SAM3 权重")


def read_rgb(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        array = np.array(rgb)
    return array, width, height


def cache_image(kind, path, array):
    return


def sam3_state(path, text=False):
    key = (str(path), path.stat().st_mtime_ns, STATE.get("sam3_epoch", 0) if text else None)
    state_key, image_key = ("sam3_text_state", "sam3_text_key") if text else ("sam3_state", "image_key")
    STATE["image_last_used"] = time.monotonic()
    if STATE.get(image_key) == key and STATE.get(state_key) is not None:
        return STATE[state_key]
    # Drop the previous feature tensors before allocating their replacement.
    STATE[state_key] = None
    STATE[image_key] = None
    with Image.open(path) as image, image.convert("RGB") as rgb, torch.inference_mode():
        state = STATE["sam3_processor"].set_image(rgb)
    backbone = state["backbone_out"]
    if text:
        backbone.pop("sam2_backbone_out", None)
    else:
        state["backbone_out"] = {"sam2_backbone_out": backbone["sam2_backbone_out"]}
    STATE[state_key] = state
    STATE[image_key] = key
    return state


def expire_image_cache(now=None):
    timeout = config.SAM_CACHE_IDLE_SECONDS
    if not timeout or not LOCK.acquire(blocking=False):
        return False
    try:
        last = STATE.get("image_last_used")
        if last is None or (time.monotonic() if now is None else now) - last < timeout:
            return False
        for key in ("sam3_state", "sam3_text_state", "image_key", "sam3_text_key", "image_last_used"):
            STATE[key] = None
        release_cuda()
        trim_cpu_memory(force=True)
        return True
    finally:
        LOCK.release()


def sam3_interactive(path, points, box, output, confidence):
    ensure("sam3")
    if STATE["sam3_model"].inst_interactive_predictor is None:
        raise RuntimeError("这份 SAM3 没有点选头")
    state = sam3_state(path)
    width, height = state["original_width"], state["original_height"]
    point_coords = None
    point_labels = None
    if points:
        point_coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        point_labels = np.array([1 if p.get("positive", True) else 0 for p in points], dtype=np.int32)
    box_arr = np.array(box, dtype=np.float32) if box else None
    multi = bool(points) and len(points) == 1 and box_arr is None
    isolated = {
        "original_height": state["original_height"],
        "original_width": state["original_width"],
        "backbone_out": {
            "sam2_backbone_out": state["backbone_out"]["sam2_backbone_out"],
        },
    }
    with torch.inference_mode():
        masks, scores, _ = STATE["sam3_model"].predict_inst(
            isolated,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
            multimask_output=multi,
        )
    best = int(np.argmax(scores))
    score = float(scores[best])
    kept = [masks[best]] if score >= confidence else []
    if kept:
        STATE["sam3_epoch"] = STATE.get("sam3_epoch", 0) + 1
    return pack_output(output, width, height, kept, None)


def sam3_text(path, text, box, output, confidence):
    ensure("sam3")
    processor = STATE["sam3_processor"]
    processor.confidence_threshold = confidence
    # 点选会把部分权重转成 bfloat16，不能复用点选之前缓存的图像特征。
    cached = sam3_state(path, text=True)
    width, height = cached["original_width"], cached["original_height"]
    state = {
        "original_height": cached["original_height"],
        "original_width": cached["original_width"],
        "backbone_out": clone_tensors(cached["backbone_out"]),
        "boxes_only": output == "box",
    }
    with torch.inference_mode():
        if text:
            state = processor.set_text_prompt(text, state)
        if box:
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            bw = abs(x2 - x1) / width
            bh = abs(y2 - y1) / height
            state = processor.add_geometric_prompt([cx, cy, bw, bh], True, state)
    masks = state.get("masks")
    boxes = state.get("boxes")
    mask_list = []
    box_list = []
    if masks is not None and output != "box":
        data = masks.detach().cpu().numpy()
        for mask in data:
            mask_list.append(mask)
    if boxes is not None:
        box_list = [[float(v) for v in row] for row in boxes.detach().float().cpu().numpy()]
    return pack_output(output, width, height, mask_list, box_list)


def pack_output(output, width, height, masks, boxes):
    polygons = []
    result_boxes = []
    if output == "box" and boxes:
        for box in boxes:
            x1, y1, x2, y2 = box
            result_boxes.append([int(x1), int(y1), int(x2), int(y2)])
    else:
        for mask in masks:
            if output == "box":
                box = box_from_mask(mask)
                if box:
                    result_boxes.append(box)
            else:
                polygons.extend(polygons_from_mask(mask))
    return {"width": width, "height": height, "output": output, "polygons": polygons, "boxes": result_boxes}


@app.get("/")
def index():
    response = app.send_static_file("index.html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health():
    sam3_code = False
    try:
        import sam3  # noqa: F401
        sam3_code = True
    except Exception as exc:
        STATE["sam3_error"] = str(exc)
    vram = None
    reserved = None
    if SAM_DEVICE.startswith("cuda"):
        vram = round(torch.cuda.memory_allocated(torch.device(SAM_DEVICE)) / (1024 * 1024))
        reserved = round(torch.cuda.memory_reserved(torch.device(SAM_DEVICE)) / (1024 * 1024))
    return jsonify({
        "ok": True,
        "device": SAM_DEVICE,
        "ready_model": STATE["model"],
        "sam3": SAM3_CKPT.is_file(),
        "sam3_enabled": SAM3_ON,
        "sam3_resident": STATE.get("sam3_processor") is not None,
        "sam3_device": SAM_DEVICE,
        "sam3_vram_mb": vram,
        "sam3_reserved_mb": reserved,
        "sam3_error": STATE["sam3_error"],
    })


@app.post("/api/load")
def load_model():
    import catalog
    name = (request.get_json(force=True) or {}).get("model") or ""
    spec = catalog.get_model(name)
    if not spec:
        return jsonify({"ok": False, "error": "没有这个模型"}), 404
    if not spec.get("ready"):
        return jsonify({"ok": False, "error": catalog.missing_message(spec)}), 400
    if spec["runtime"] != "pytorch-sam3":
        return jsonify({"ok": True, "model": spec["id"]})
    try:
        with LOCK:
            ensure("sam3")
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "model": STATE["model"]})


@app.get("/api/models")
def models():
    import catalog
    return jsonify({"models": catalog.list_models(), "runtimes": sorted(catalog.RUNTIMES)})


@app.post("/api/models")
def import_model():
    import catalog
    try:
        pack = catalog.save_pack(request.get_json(force=True) or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(pack)


@app.get("/api/images")
def images():
    proj = current_project()
    image_root = Path(proj["images"])
    ann_root = Path(proj["annotations"])
    query = request.args.get("q", "").strip().lower()
    offset = max(0, int(request.args.get("offset", 0)))
    limit = min(50000, max(1, int(request.args.get("limit", 80))))
    text_filter = request.args.get("t", "").strip()
    shape_filter = request.args.get("s", "").strip()
    info = scan_annotations(ann_root)
    names = []
    if not image_root.is_dir():
        return jsonify({"total": 0, "done": 0, "items": [], "focus": -1, "texts": {}})
    for path in sorted(image_root.iterdir()):
        if path.suffix.lower() not in EXTS:
            continue
        if query and query not in path.name.lower():
            continue
        state = info.get(path.stem) or {"labeled": False, "text": "none"}
        if text_filter and state["text"] != text_filter:
            continue
        if shape_filter and state["labeled"] != (shape_filter == "yes"):
            continue
        names.append((path, state))
    focus = request.args.get("focus", "").strip()
    if focus:
        pos = next((i for i, (path, _) in enumerate(names) if path.name == focus), -1)
        if pos >= 0:
            offset = (pos // limit) * limit
    page = names[offset:offset + limit]
    focus_index = next((i for i, (path, _) in enumerate(page) if path.name == focus), -1)
    counts = {"none": 0, "pending": 0, "approved": 0, "rejected": 0}
    for _, state in names:
        counts[state["text"]] = counts.get(state["text"], 0) + 1
    return jsonify({
        "total": len(names),
        "done": sum(1 for _, state in names if state["labeled"]),
        "focus": focus_index,
        "texts": counts,
        "items": [{"name": p.name, "saved": state["labeled"], "text": state["text"]} for p, state in page],
    })


@app.get("/api/image/<path:name>")
def image_file(name):
    path = safe_image(name)
    response = send_file(path)
    response.headers["Cache-Control"] = "no-store"
    return response


def image_mutation_project():
    project = current_project()
    with IMPORT_JOBS_LOCK:
        if any(j['project_id'] == project['id'] and j['status'] == 'running' for j in IMPORT_JOBS.values()):
            abort(409, '正在导入，请完成后再删除或恢复图片')
    import project_backups
    if project_backups.status(DATA_ROOT, project['id'])['status'] in {'queued', 'running'}:
        abort(409, '正在备份，请完成后再删除或恢复图片')
    import texts
    job = texts.job_state()
    if job.get('running') and job.get('project') == project['id']:
        abort(409, '正在批量生成，请完成后再删除或恢复图片')
    return project


@app.post('/api/images/delete')
def delete_images():
    import image_trash
    project = image_mutation_project()
    try:
        return jsonify(image_trash.remove(project, (request.get_json(force=True) or {}).get('names')))
    except (ValueError, OSError) as exc:
        return jsonify({'error': str(exc)}), 400


@app.get('/api/images/trash')
def image_trash_list():
    import image_trash
    return jsonify({'batches': image_trash.list_batches(current_project())})


@app.post('/api/images/restore')
def restore_images():
    import image_trash
    project = image_mutation_project()
    try:
        return jsonify(image_trash.restore(project, (request.get_json(force=True) or {}).get('batch')))
    except (ValueError, OSError) as exc:
        return jsonify({'error': str(exc)}), 400


def labels_path():
    return Path(current_project()["labels"])


def _read_labels():
    path = labels_path()
    if not path.is_file():
        return [], True
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = []
    seen = set()
    for item in data.get("labels") or []:
        name = str(item.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        labels.append({"name": name, "locked": bool(item.get("locked"))})
    return labels, False


@app.get("/api/labels")
def get_labels():
    labels, fresh = _read_labels()
    return jsonify({"labels": labels, "fresh": fresh})


@app.post("/api/labels")
def save_labels():
    import json
    body = request.get_json(force=True)
    labels = []
    seen = set()
    for item in body.get("labels") or []:
        name = str(item.get("name") or "").strip()[:64]
        if not name or name in seen:
            continue
        seen.add(name)
        labels.append({"name": name, "locked": bool(item.get("locked"))})
    path = labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels}, ensure_ascii=False), encoding="utf-8")
    return jsonify({"labels": labels})


@app.get("/api/annotation/<path:name>")
def get_annotation(name):
    path = safe_image(name)
    target = ann_path(path)
    if not target.is_file():
        return jsonify({"shapes": []})
    return jsonify(read_doc(target) or {"shapes": []})


@app.post("/api/segment")
def segment():
    import catalog
    import runtimes
    body = request.get_json(force=True)
    path = safe_image(body.get("name", ""))
    spec = catalog.get_model(body.get("model") or "")
    if not spec or not spec.get("ready"):
        return jsonify({"error": catalog.missing_message(spec) if spec else "没有这个模型"}), 400
    output = "box" if body.get("output") == "box" else "mask"
    if output not in spec["outputs"]:
        return jsonify({"error": "这个模型没有这种结果"}), 400
    points = body.get("points") or []
    box = body.get("box")
    text = (body.get("text") or "").strip()
    if "text" not in spec["prompts"]:
        text = ""
    try:
        confidence = min(1.0, max(0.0, float(body.get("confidence", spec["confidence"]["default"]))))
    except (TypeError, ValueError):
        confidence = spec["confidence"]["default"]
    if not points and not box and not text:
        abort(400)
    try:
        with LOCK:
            if spec["runtime"] == "pytorch-sam3" and text and not points:
                payload = sam3_text(path, text, box, output, confidence)
            elif spec["runtime"] == "pytorch-sam3":
                payload = sam3_interactive(path, points, box, output, confidence)
            elif spec["runtime"] == "onnx-sam1":
                array, width, height = read_rgb(path)
                masks, scores = runtimes.ONNX.masks(array, points, box, spec["files"])
                keep = []
                if len(scores):
                    best = int(np.argmax(scores))
                    if float(scores[best]) >= confidence:
                        keep = [masks[best]]
                payload = pack_output(output, width, height, keep, None)
            elif spec["runtime"] == "tensorrt-sam":
                array, width, height = read_rgb(path)
                masks, scores = runtimes.TRT.masks(array, points, box, spec["files"])
                keep = [masks[i] for i, score in enumerate(scores) if float(score) >= confidence][:1]
                payload = pack_output(output, width, height, keep, None)
            else:
                return jsonify({"error": "不认识的运行时 " + spec["runtime"]}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    return jsonify(payload)


@app.post("/api/save")
def save():
    import json
    body = request.get_json(force=True)
    path = safe_image(body.get("name", ""))
    with Image.open(path) as image:
        width, height = image.size
    shapes = []
    for shape in body.get("shapes") or []:
        kind = shape.get("shape_type") or "polygon"
        points = shape.get("points") or []
        raw_flags = shape.get("flags") or {}
        flags = {}
        if shape.get("locked") or raw_flags.get("locked"):
            flags["locked"] = True
        if kind == "rectangle" and len(points) >= 2:
            shapes.append({
                "label": shape.get("label") or "目标",
                "points": [[float(points[0][0]), float(points[0][1])], [float(points[1][0]), float(points[1][1])]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": flags,
            })
        elif len(points) >= 3:
            shapes.append({
                "label": shape.get("label") or "目标",
                "points": [[float(x), float(y)] for x, y in points],
                "group_id": None,
                "shape_type": "polygon",
                "flags": flags,
            })
    target = ann_path(path)
    old = read_doc(target) if target.is_file() else None
    document = {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    if old and old.get("texts"):
        document["texts"] = old["texts"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return jsonify({"saved": str(target), "count": len(shapes)})


@app.post("/api/texts/save")
def save_texts():
    import texts
    body = request.get_json(force=True) or {}
    path = safe_image(body.get("name", ""))
    target = ann_path(path)
    doc = read_doc(target) if target.is_file() else None
    if doc is None:
        doc = blank_doc(path)
    items = texts.clean_items(body.get("items"))
    old = doc.get("texts") or {}
    doc["texts"] = {
        "generator": old.get("generator"),
        "generated_at": old.get("generated_at"),
        "reviewed_at": int(time.time()),
        "items": items,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return jsonify({"texts": doc["texts"], "status": texts.text_status(doc)})


def blank_doc(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def project_label_names(proj):
    path = Path(proj["labels"])
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(item.get("name")) for item in data.get("labels") or [] if item.get("name")]


def prompt_context(proj, image_path, doc):
    return {
        "labels": project_label_names(proj),
        "shapes": doc.get("shapes") or [],
        "filename": image_path.name,
        "width": doc.get("imageWidth"),
        "height": doc.get("imageHeight"),
    }


def pick_prompts(proj, ids):
    import texts
    prompts = texts.project_settings(proj)["prompts"]
    if ids:
        wanted = set(ids)
        return [p for p in prompts if p["id"] in wanted and p["format"] != "boxes"]
    return [p for p in prompts if p["batch"]]


@app.post("/api/texts/generate")
def generate_texts():
    """后台给项目里的图片生成文字。scope: missing（没有文字或被打回的）| all | one。"""
    import texts
    body = request.get_json(force=True) or {}
    proj = current_project()
    provider = texts.resolve_provider(proj, body.get("provider"))
    if not provider:
        return jsonify({"error": "还没有配置大模型接口，先到「大模型设置」里加一个"}), 400
    prompts = pick_prompts(proj, body.get("prompts"))
    if not prompts:
        return jsonify({"error": "没有可用的提示词：到「大模型设置」里勾选「参与批量生成」"}), 400
    image_root = Path(proj["images"])
    ann_root = Path(proj["annotations"])
    scope = body.get("scope") or "missing"
    info = scan_annotations(ann_root)
    tasks = []
    if scope == "one":
        path = safe_image(body.get("name", ""))
        tasks.append((path, ann_path(path)))
    else:
        for path in sorted(image_root.iterdir()) if image_root.is_dir() else []:
            if path.suffix.lower() not in EXTS:
                continue
            state = info.get(path.stem) or {"labeled": False, "text": "none"}
            if scope == "missing" and state["text"] in {"pending", "approved"}:
                continue
            tasks.append((path, ann_root / (path.stem + ".json")))
    if not tasks:
        return jsonify({"ok": True, "queued": 0, "job": texts.job_state()})

    def worker(image_path, target):
        doc = read_doc(target) if target.is_file() else None
        if doc is None:
            doc = blank_doc(image_path)
        context = prompt_context(proj, image_path, doc)
        items = []
        for prompt in prompts:
            items.extend(texts.run_prompt(provider, prompt, image_path, context, track=True)["items"])
        # 生成期间用户可能保存过，写盘前重新读一次，只换文字部分。
        latest = read_doc(target) if target.is_file() else None
        if latest is not None:
            doc = latest
        texts.merge_generated(doc, items, provider["name"], [p["id"] for p in prompts])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    if not texts.start_job(proj["id"], tasks, worker):
        return jsonify({"error": "后台还在生成，等它跑完再点"}), 409
    return jsonify({"ok": True, "queued": len(tasks), "job": texts.job_state()})


@app.get("/api/texts/status")
def texts_status():
    import texts
    state = texts.job_state()
    provider = None
    try:
        provider = texts.resolve_provider(current_project())
    except Exception:
        pass
    state["generator"] = provider["name"] if provider else ""
    return jsonify(state)


@app.get("/api/vlm/config")
def vlm_config():
    import texts
    proj = current_project()
    return jsonify({
        **texts.public_config(texts.load_config()),
        "project": texts.project_settings(proj),
    })


@app.post("/api/vlm/config")
def save_vlm_config():
    """一次保存全局接口和当前项目的提示词。"""
    import texts
    body = request.get_json(force=True) or {}
    proj = current_project()
    try:
        settings = texts.clean_project_settings(body.get("project") or {})
        config = texts.save_config(body)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if settings["provider"] and not any(p["id"] == settings["provider"] for p in config["providers"]):
        settings["provider"] = ""
    rows = load_projects()
    for row in rows:
        if row.get("id") == proj["id"]:
            row["vlm"] = settings
    save_projects(rows)
    return jsonify({**texts.public_config(config), "project": settings})


def provider_from_body(body, proj):
    """试运行时可以直接用编辑框里还没保存的接口。"""
    import texts
    raw = body.get("provider")
    if isinstance(raw, dict):
        old = next((p for p in texts.load_config()["providers"] if p["id"] == raw.get("id")), None)
        return texts.clean_provider(raw, old)
    return texts.resolve_provider(proj, raw)


@app.post("/api/vlm/test")
def test_vlm():
    import texts
    body = request.get_json(force=True) or {}
    try:
        provider = provider_from_body(body, current_project())
        if not provider:
            return jsonify({"error": "没有接口"}), 400
        started = time.time()
        models = texts.test_provider(provider)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "models": models, "ms": int((time.time() - started) * 1000)})


@app.post("/api/vlm/run")
def run_vlm():
    """对一张图跑一条提示词，不写盘。prompt 可以是 id，也可以是编辑中的提示词对象。"""
    import texts
    body = request.get_json(force=True) or {}
    proj = current_project()
    path = safe_image(body.get("name", ""))
    try:
        provider = provider_from_body(body, proj)
        if not provider:
            return jsonify({"error": "还没有配置大模型接口"}), 400
        raw_prompt = body.get("prompt")
        if isinstance(raw_prompt, dict):
            prompt = texts.clean_prompt(raw_prompt)
        else:
            prompt = next((p for p in texts.project_settings(proj)["prompts"] if p["id"] == raw_prompt), None)
            if not prompt:
                return jsonify({"error": "没有这条提示词"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    target = ann_path(path)
    doc = read_doc(target) if target.is_file() else None
    doc = doc or blank_doc(path)
    if isinstance(body.get("shapes"), list):
        doc["shapes"] = body["shapes"]
    started = time.time()
    try:
        result = texts.run_prompt(provider, prompt, path, prompt_context(proj, path, doc))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
    result["ms"] = int((time.time() - started) * 1000)
    result["generator"] = provider["name"]
    result["prompt"] = prompt
    return jsonify(result)


def _iter_docs(proj):
    image_root = Path(proj["images"])
    ann_root = Path(proj["annotations"])
    if not image_root.is_dir():
        return
    for path in sorted(image_root.iterdir()):
        if path.suffix.lower() not in EXTS:
            continue
        target = ann_root / (path.stem + ".json")
        doc = read_doc(target) if target.is_file() else None
        if doc:
            yield path, doc


@app.get("/api/export/<kind>")
def export(kind):
    """vlm.jsonl：已通过的问答（LLaMA-Factory sharegpt 格式）；yolo.zip / labelme.zip：框和多边形。"""
    import io
    import zipfile
    proj = current_project()
    with IMPORT_JOBS_LOCK:
        if any(job['project_id'] == proj['id'] and job['status'] == 'running'
               for job in IMPORT_JOBS.values()):
            return jsonify({'error': '项目仍在导入，请完成后再导出和备份'}), 409
    if kind not in {'vlm.jsonl', 'yolo.zip', 'labelme.zip'}:
        abort(404)
    import project_backups
    project_backups.schedule(DATA_ROOT, proj)
    with_images = request.args.get("images") == "1"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if kind == "vlm.jsonl":
        statuses = {"approved"} if request.args.get("all") != "1" else {"approved", "pending"}
        lines = []
        for path, doc in _iter_docs(proj):
            for item in (doc.get("texts") or {}).get("items") or []:
                if item.get("status") not in statuses:
                    continue
                lines.append(json.dumps({
                    "messages": [
                        {"role": "user", "content": "<image>" + item.get("question", "")},
                        {"role": "assistant", "content": item.get("answer", "")},
                    ],
                    "images": [str(path)],
                    "kind": item.get("kind") or "qa",
                }, ensure_ascii=False))
        data = io.BytesIO(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
        return send_file(data, mimetype="application/jsonl", as_attachment=True,
                         download_name="%s_vlm_%s.jsonl" % (proj["name"], stamp))
    if kind not in {"yolo.zip", "labelme.zip"}:
        abort(404)
    import tempfile
    export_root = DATA_ROOT / '.exports'
    export_root.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=export_root)
    tmp.close()
    names = project_label_names(proj)
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, doc in _iter_docs(proj):
            shapes = doc.get("shapes") or []
            if not shapes:
                continue
            if kind == "labelme.zip":
                archive.writestr("annotations/" + path.stem + ".json", json.dumps(doc, ensure_ascii=False))
            else:
                w = float(doc.get("imageWidth") or 1)
                h = float(doc.get("imageHeight") or 1)
                rows = []
                for shape in shapes:
                    label = shape.get("label") or "目标"
                    if label not in names:
                        names.append(label)
                    pts = shape.get("points") or []
                    if not pts:
                        continue
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x1, x2 = max(0, min(xs)), min(w, max(xs))
                    y1, y2 = max(0, min(ys)), min(h, max(ys))
                    rows.append("%d %.6f %.6f %.6f %.6f" % (
                        names.index(label), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
                archive.writestr("labels/" + path.stem + ".txt", "\n".join(rows) + "\n")
            if with_images:
                archive.write(path, "images/" + path.name, compress_type=zipfile.ZIP_STORED)
        if kind == "yolo.zip":
            archive.writestr("classes.txt", "\n".join(names) + "\n")
            archive.writestr("data.yaml", "path: .\ntrain: images\nval: images\nnames:\n" + "".join(
                "  %d: %s\n" % (n, json.dumps(name, ensure_ascii=False)) for n, name in enumerate(names)))
    response = send_file(tmp.name, mimetype="application/zip", as_attachment=True,
                         download_name="%s_%s_%s" % (proj["name"], kind.replace(".zip", ""), stamp) + ".zip")
    response.call_on_close(lambda: Path(tmp.name).unlink(missing_ok=True))
    return response


@app.get('/api/backup/status')
def backup_status():
    import project_backups
    return jsonify(project_backups.status(DATA_ROOT, current_project()['id']))


@app.post("/api/texts/stop")
def texts_stop():
    import texts
    texts.stop_job()
    return jsonify(texts.job_state())


@app.get("/api/projects")
def list_projects():
    rows = []
    for item in load_projects():
        stats = project_stats(item)
        rows.append({
            "id": item["id"],
            "name": item["name"],
            "builtin": bool(item.get("builtin")),
            **stats,
        })
    return jsonify({"projects": rows})


@app.post("/api/projects")
def create_project():
    body = request.get_json(force=True) or {}
    name = str(body.get("name") or "").strip()[:40]
    if not name:
        return jsonify({"error": "请填写项目名称"}), 400
    rows = load_projects()
    if any(item.get("name") == name for item in rows):
        return jsonify({"error": "已经有同名项目"}), 400
    pid = "p" + str(int(time.time() * 1000))
    root = PROJECTS_DIR / pid
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    rows.append({
        "id": pid,
        "name": name,
        "images": str(root / "images"),
        "annotations": str(root / "annotations"),
        "labels": str(root / "labels.json"),
        "builtin": False,
    })
    save_projects(rows)
    return jsonify({"id": pid, "name": name})


@app.patch("/api/projects/<pid>")
def rename_project(pid):
    body = request.get_json(force=True) or {}
    name = str(body.get("name") or "").strip()
    if not name or len(name) > 40:
        return jsonify({"error": "项目名称须为 1～40 个字符"}), 400
    rows = load_projects()
    item = next((row for row in rows if row.get("id") == pid), None)
    if item is None:
        return jsonify({"error": "项目不存在"}), 404
    if any(row.get("id") != pid and row.get("name") == name for row in rows):
        return jsonify({"error": "已经有同名项目"}), 400
    item["name"] = name
    save_projects(rows)
    return jsonify({"id": pid, "name": name})


@app.delete("/api/projects/<pid>")
def delete_project(pid):
    rows = load_projects()
    item = next((row for row in rows if row.get("id") == pid), None)
    if not item:
        return jsonify({"error": "项目不存在"}), 404
    if item.get("builtin") or pid == "default":
        return jsonify({"error": "默认项目里是原来的图片，不能删除"}), 400
    root = (PROJECTS_DIR / pid).resolve()
    if root.parent == PROJECTS_DIR.resolve() and root.is_dir():
        shutil.rmtree(root)
    save_projects([row for row in rows if row.get("id") != pid])
    return jsonify({"ok": True})


ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}
IMPORT_CAP = 20000
IMPORT_JOBS = {}
IMPORT_JOBS_LOCK = threading.Lock()


def import_job_snapshot(job):
    return {k: v for k, v in job.items() if k != 'project_id'}


@app.get('/api/import/active')
def active_import_jobs():
    projects = {item['id']: item['name'] for item in load_projects()}
    with IMPORT_JOBS_LOCK:
        return jsonify({'jobs': [{**import_job_snapshot(job), 'project_id': job['project_id'],
                                  'project_name': projects[job['project_id']]}
                                 for job in IMPORT_JOBS.values()
                                 if job['status'] == 'running' and job['project_id'] in projects]})


@app.get("/api/import/jobs/<job_id>")
def import_job_status(job_id):
    project = current_project()
    with IMPORT_JOBS_LOCK:
        job = IMPORT_JOBS.get(job_id)
        if not job or job['project_id'] != project['id']:
            return jsonify({'error': '导入任务不存在'}), 404
        return jsonify(import_job_snapshot(job))


def start_path_import(src, project, mode='copy'):
    from dataset_import import import_tree
    with IMPORT_JOBS_LOCK:
        for key in list(IMPORT_JOBS):
            job = IMPORT_JOBS[key]
            if job['status'] != 'running' and time.time() - job['updated'] > 3600:
                del IMPORT_JOBS[key]
        for job in IMPORT_JOBS.values():
            if job['project_id'] == project['id'] and job['status'] == 'running':
                return jsonify(import_job_snapshot(job)), 202
        if len(IMPORT_JOBS) >= 100:
            finished = [key for key, job in IMPORT_JOBS.items() if job['status'] != 'running']
            if not finished:
                return jsonify({'error': '导入任务过多，请稍后重试'}), 429
            del IMPORT_JOBS[finished[0]]
        job_id = uuid.uuid4().hex
        job = {'id': job_id, 'project_id': project['id'], 'mode': mode, 'source': str(src), 'status': 'running', 'phase': 'scanning',
               'processed': 0, 'total': 0, 'count': 0, 'skipped': 0, 'shapes': 0, 'updated': time.time()}
        IMPORT_JOBS[job_id] = job
        snapshot = import_job_snapshot(job)

    def update(**values):
        with IMPORT_JOBS_LOCK:
            job.update(values, updated=time.time())

    def worker():
        try:
            options = {'transfer': 'move'} if mode == 'move' else {}
            result = import_tree(src, project, IMPORT_CAP, progress=update, **options)
            update(status='complete', phase='complete', result=result)
        except Exception as exc:
            update(status='failed', phase='failed', error=str(exc))
    threading.Thread(target=worker, daemon=True, name='dataset-import-' + job_id[:8]).start()
    return jsonify(snapshot), 202


@app.post("/api/import/files")
def import_files():
    """Import images with Labelme / YOLO sidecars, preserving uploaded paths."""
    import tempfile
    from dataset_import import import_tree, extract_archive
    staging = DATA_ROOT / ".imports"
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upload-", dir=staging) as tmp:
        root = Path(tmp) / "dataset"
        root.mkdir()
        try:
            for index, handle in enumerate(request.files.getlist("files")):
                raw = (handle.filename or "").replace("\\", "/")
                dest = (root / raw).resolve()
                if not raw or ":" in raw or root.resolve() not in dest.parents:
                    return jsonify({"error": "上传文件路径无效"}), 400
                if Path(raw).suffix.lower() in ARCHIVE_EXTS:
                    archive = Path(tmp) / ("archive_" + str(index))
                    handle.save(archive)
                    extract_archive(archive, root / ("archive_" + str(index)))
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    handle.save(dest)
            return jsonify(import_tree(root, current_project(), IMPORT_CAP))
        except (ValueError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400


@app.post("/api/import/path")
def import_server_path():
    from dataset_import import import_tree
    body = request.get_json(force=True) or {}
    raw = str(body.get("path") or "").strip()
    src = Path(raw).expanduser()
    if not raw or not src.is_absolute():
        return jsonify({"error": "请填服务器绝对路径，例如 /data/dataset"}), 400
    if not src.exists():
        return jsonify({"error": "路径不存在"}), 400
    mode = body.get('mode') or os.environ.get('IMPORT_MODE', 'copy')
    if mode not in {'copy', 'move'}:
        return jsonify({'error': '未知导入方式'}), 400
    if mode == 'move':
        allowed = os.environ.get('MOVE_IMPORT_ROOT', '')
        if not allowed:
            return jsonify({'error': '当前部署未启用移动导入'}), 400
        allowed = Path(allowed).resolve()
        # Old /imports bookmarks still refer to the same dataset directory.
        if Path('/imports') in src.parents:
            src = allowed / src.relative_to('/imports')
        src = src.resolve()
        if src != allowed and allowed not in src.parents:
            return jsonify({'error': '移动导入仅限配置的数据集目录'}), 400
    if body.get("async"):
        return start_path_import(src, current_project(), mode)
    try:
        options = {'transfer': 'move'} if mode == 'move' else {}
        return jsonify(import_tree(src, current_project(), IMPORT_CAP, **options))
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get('/api/import/config')
def import_config():
    return jsonify({'default_mode': os.environ.get('IMPORT_MODE', 'copy'),
                    'move_enabled': bool(os.environ.get('MOVE_IMPORT_ROOT'))})


def warmup_sam3():
    try:
        with LOCK:
            load_sam3()
        print("SAM3 ready on", SAM_DEVICE, flush=True)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    print("data:", DATA_ROOT, "device:", SAM_DEVICE, "sam3:", "on" if SAM3_ON else "off", flush=True)
    if SAM3_ON:
        threading.Thread(target=warmup_sam3, daemon=True, name="sam3-warmup").start()
    if config.SAM_CACHE_IDLE_SECONDS:
        def maintain_memory():
            while True:
                time.sleep(min(30, config.SAM_CACHE_IDLE_SECONDS))
                expire_image_cache()
        threading.Thread(target=maintain_memory, daemon=True, name="image-cache-cleanup").start()
    app.run(host=config.HOST, port=config.PORT, threaded=True)
