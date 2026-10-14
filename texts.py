"""多模态大模型标注：接入配置、提示词、调用和结果解析。

接入（providers）是全局的，存在 数据目录/vlm.json；
提示词（prompts）跟着项目走，存在 projects.json 里每个项目的 "vlm" 字段。
所有接口都按 OpenAI 兼容的 /chat/completions 调用。
"""
import base64
import copy
import gc
import http.client
import io
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import config

VLM_CONFIG = config.DATA_ROOT / "vlm.json"
STATUSES = {"pending", "approved", "rejected"}
FORMATS = {"text", "json", "boxes"}
COORDS = {"qwen1000", "norm", "pixel"}
KEY_MASK = "******"

DEFAULT_PROMPTS = [
    {
        "id": "category",
        "name": "隐患类别",
        "kind": "category",
        "system": "",
        "question": "请判断图中隐患的类别。",
        "format": "text",
        "coords": "qwen1000",
        "batch": True,
        "max_tokens": 0,
    },
]

JOB = {
    "running": False,
    "project": None,
    "total": 0,
    "done": 0,
    "failed": 0,
    "current": "",
    "error": None,
    "started": 0,
    "finished": 0,
    "stop": False,
    "canceled": False,
    "conn": None,
}
JOB_LOCK = threading.Lock()


class Canceled(Exception):
    """User asked to stop the current generation."""


# ---------- 配置 ----------

def _num(value, default, low, high, cast=float):
    try:
        value = cast(value)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, value))


def _slug(text, fallback):
    text = re.sub(r"[^0-9A-Za-z_-]+", "", str(text or ""))[:40]
    return text or fallback


def clean_provider(row, old=None):
    row = row or {}
    url = str(row.get("url") or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("接口地址要以 http:// 或 https:// 开头")
    key = str(row.get("key") or "")
    if key == KEY_MASK:
        key = (old or {}).get("key") or ""
    name = str(row.get("name") or "").strip()[:40] or str(row.get("model") or "") or "接口"
    return {
        "id": _slug(row.get("id"), "p%d" % int(time.time() * 1000)),
        "name": name,
        "url": url,
        "model": str(row.get("model") or "").strip()[:120] or "default",
        "key": key,
        "temperature": _num(row.get("temperature"), 0.2, 0.0, 2.0),
        "max_tokens": _num(row.get("max_tokens"), 512, 16, 32768, int),
        "timeout": _num(row.get("timeout"), 180, 5, 3600, int),
        "image_side": _num(row.get("image_side"), 1024, 128, 4096, int),
    }


def clean_prompt(row, n=0):
    row = row or {}
    question = str(row.get("question") or "").strip()[:4000]
    if not question:
        raise ValueError("提示词「%s」没有填写问题" % (row.get("name") or n + 1))
    fmt = row.get("format") if row.get("format") in FORMATS else "text"
    return {
        "id": _slug(row.get("id"), "q%d_%d" % (int(time.time() * 1000), n)),
        "name": str(row.get("name") or "").strip()[:40] or question[:16],
        "kind": str(row.get("kind") or ("boxes" if fmt == "boxes" else "qa")).strip()[:32] or "qa",
        "system": str(row.get("system") or "").strip()[:4000],
        "question": question,
        "format": fmt,
        "coords": row.get("coords") if row.get("coords") in COORDS else "qwen1000",
        "batch": bool(row.get("batch")) and fmt != "boxes",
        "max_tokens": _num(row.get("max_tokens"), 0, 0, 32768, int),
    }


def load_config():
    """{"providers": [...], "default_provider": id}；兼容老的单接口格式。"""
    data = {}
    if VLM_CONFIG.is_file():
        try:
            data = json.loads(VLM_CONFIG.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
    if not data and config.VLM_URL:
        # 第一次启动：用环境变量里的地址生成默认接口（Docker 部署时指向 vlm 容器）。
        data = {"url": config.VLM_URL, "model": config.VLM_MODEL_NAME or "default", "key": config.VLM_API_KEY,
                "max_tokens": 512, "timeout": 600, "image_side": config.VLM_IMAGE_SIDE,
                "name": config.VLM_MODEL_NAME or "本机大模型"}
    if data.get("url") and not data.get("providers"):
        legacy = {
            "id": "default",
            "name": data.get("name") or data.get("model") or "默认接口",
            "url": data["url"],
            "model": data.get("model") or "default",
            "key": data.get("key") or "",
            "temperature": data.get("temperature", 0.2),
            "max_tokens": data.get("max_tokens", 256),
            "timeout": data.get("timeout", 300),
            "image_side": data.get("image_side", 512),
        }
        data = {"providers": [legacy], "default_provider": "default"}
    providers = []
    for row in data.get("providers") or []:
        try:
            providers.append(clean_provider(row, row))
        except ValueError:
            continue
    default = data.get("default_provider")
    if not any(p["id"] == default for p in providers):
        default = providers[0]["id"] if providers else None
    return {"providers": providers, "default_provider": default}


def save_config(body):
    old = {p["id"]: p for p in load_config()["providers"]}
    providers = []
    seen = set()
    for row in body.get("providers") or []:
        item = clean_provider(row, old.get(str(row.get("id") or "")))
        if item["id"] in seen:
            item["id"] = item["id"] + "_" + str(len(seen))
        seen.add(item["id"])
        providers.append(item)
    default = body.get("default_provider")
    if default not in seen:
        default = providers[0]["id"] if providers else None
    data = {"providers": providers, "default_provider": default}
    VLM_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    VLM_CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def public_config(config):
    rows = []
    for p in config["providers"]:
        row = dict(p)
        row["key"] = KEY_MASK if p.get("key") else ""
        rows.append(row)
    return {"providers": rows, "default_provider": config["default_provider"]}


def project_settings(project):
    vlm = project.get("vlm") or {}
    prompts = vlm.get("prompts")
    if not isinstance(prompts, list):
        prompts = copy.deepcopy(DEFAULT_PROMPTS)
    clean = []
    for n, row in enumerate(prompts):
        try:
            clean.append(clean_prompt(row, n))
        except ValueError:
            continue
    return {"provider": vlm.get("provider") or "", "prompts": clean}


def clean_project_settings(body):
    prompts = []
    seen = set()
    for n, row in enumerate(body.get("prompts") or []):
        item = clean_prompt(row, n)
        if item["id"] in seen:
            item["id"] = item["id"] + "_" + str(n)
        seen.add(item["id"])
        prompts.append(item)
    return {"provider": str(body.get("provider") or ""), "prompts": prompts}


def resolve_provider(project, override=None):
    config = load_config()
    wanted = override or project_settings(project)["provider"] or config["default_provider"]
    for p in config["providers"]:
        if p["id"] == wanted:
            return p
    return config["providers"][0] if config["providers"] else None


# ---------- 状态 ----------

def text_status(doc):
    """一张图的文字总状态：none / pending / approved / rejected。"""
    items = (doc.get("texts") or {}).get("items") or []
    if not items:
        return "none"
    states = {item.get("status") for item in items}
    if "pending" in states:
        return "pending"
    if "rejected" in states:
        return "rejected"
    return "approved"


# ---------- 调用 ----------

def _encode_image(image_path, side):
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((side, side), Image.Resampling.LANCZOS)
        size = rgb.size
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), size


def _shape_rows(shapes):
    rows = []
    for shape in shapes or []:
        pts = shape.get("points") or []
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        rows.append({
            "label": shape.get("label") or "目标",
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
        })
    return rows


def render_prompt(text, context):
    """替换 {labels} {shapes} {filename} {width} {height}；其余花括号原样保留。"""
    shapes = _shape_rows(context.get("shapes"))
    values = {
        "{labels}": "、".join(context.get("labels") or []) or "（无）",
        "{shapes}": json.dumps(shapes, ensure_ascii=False) if shapes else "（无）",
        "{filename}": context.get("filename") or "",
        "{width}": str(context.get("width") or ""),
        "{height}": str(context.get("height") or ""),
    }
    for key, value in values.items():
        text = text.replace(key, value)
    return text


def _connection(url, timeout):
    parsed = urlparse(url)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return cls(parsed.hostname, port, timeout=timeout), parsed


def _headers(provider):
    headers = {"Content-Type": "application/json"}
    if provider.get("key"):
        headers["Authorization"] = "Bearer " + provider["key"]
    return headers


def test_provider(provider):
    """GET {url}/models，返回接口上的模型名。"""
    conn, parsed = _connection(provider["url"] + "/models", min(15, provider.get("timeout") or 15))
    try:
        conn.request("GET", parsed.path, headers=_headers(provider))
        resp = conn.getresponse()
        raw = resp.read()
    finally:
        conn.close()
    if resp.status >= 400:
        raise RuntimeError("接口返回 %s：%s" % (resp.status, raw[:200].decode("utf-8", "replace")))
    data = json.loads(raw.decode("utf-8") or "{}")
    return [str(row.get("id")) for row in data.get("data") or [] if row.get("id")]


def chat(provider, prompt, image_path, context, track=False):
    """调用一次模型，返回 (原始文字, 发送给模型的图片尺寸)。"""
    if track and JOB.get("stop"):
        raise Canceled()
    b64, sent = _encode_image(image_path, provider.get("image_side") or 1024)
    messages = []
    system = render_prompt(prompt.get("system") or "", context)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": render_prompt(prompt["question"], context)},
        ],
    })
    del b64
    body = json.dumps({
        "model": provider.get("model") or "default",
        "temperature": float(provider.get("temperature", 0.2)),
        "max_tokens": int(prompt.get("max_tokens") or provider.get("max_tokens") or 512),
        "messages": messages,
    }).encode("utf-8")
    conn, parsed = _connection(provider["url"] + "/chat/completions", float(provider.get("timeout") or 180))
    if track:
        JOB["conn"] = conn
    try:
        if track and JOB.get("stop"):
            raise Canceled()
        conn.request("POST", parsed.path, body=body, headers=_headers(provider))
        del body
        resp = conn.getresponse()
        raw = resp.read()
        if track and (JOB.get("stop") or resp.status == 499):
            raise Canceled()
        if resp.status >= 400:
            raise RuntimeError("模型接口 %s：%s" % (resp.status, raw[:300].decode("utf-8", "replace")))
        data = json.loads(raw.decode("utf-8"))
    except Canceled:
        raise
    except Exception:
        if track and JOB.get("stop"):
            raise Canceled()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if track and JOB.get("conn") is conn:
            JOB["conn"] = None
        gc.collect()
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content), sent


def _strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def _json_array(text):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        rows = json.loads(text[start:end + 1])
    except Exception:
        return None
    return rows if isinstance(rows, list) else None


def _to_pixels(box, coords, sent, width, height):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if coords == "norm" or max(x1, y1, x2, y2) <= 1.0:
        sx, sy = width, height
    elif coords == "pixel":
        sx, sy = width / max(sent[0], 1), height / max(sent[1], 1)
    else:
        sx, sy = width / 1000.0, height / 1000.0
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    clamp = lambda v, hi: int(round(min(max(v, 0), hi)))
    return [clamp(x1, width), clamp(y1, height), clamp(x2, width), clamp(y2, height)]


def parse(prompt, raw, context, sent):
    """按提示词的输出格式解析。返回 {"items": [...], "boxes": [...]}。"""
    text = _strip_think(raw)
    question = render_prompt(prompt["question"], context)
    if prompt["format"] == "boxes":
        boxes = []
        for row in _json_array(text) or []:
            if not isinstance(row, dict):
                continue
            box = row.get("bbox_2d") or row.get("bbox") or row.get("box")
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            try:
                pixels = _to_pixels(box, prompt["coords"], sent, context["width"], context["height"])
            except (TypeError, ValueError):
                continue
            if pixels[2] - pixels[0] < 2 or pixels[3] - pixels[1] < 2:
                continue
            label = str(row.get("label") or row.get("name") or row.get("category") or "目标").strip()[:64]
            boxes.append({"label": label or "目标", "box": pixels})
        return {"items": [], "boxes": boxes}
    if prompt["format"] == "json":
        items = []
        for row in _json_array(text) or []:
            if not isinstance(row, dict):
                continue
            q = str(row.get("question") or "").strip()
            a = str(row.get("answer") or "").strip()
            if q and a:
                items.append({"kind": str(row.get("kind") or prompt["kind"]), "question": q, "answer": a})
        if items:
            return {"items": items, "boxes": []}
    if not text:
        raise RuntimeError("模型返回为空")
    return {"items": [{"kind": prompt["kind"], "question": question, "answer": text}], "boxes": []}


def run_prompt(provider, prompt, image_path, context, track=False):
    raw, sent = chat(provider, prompt, image_path, context, track=track)
    result = parse(prompt, raw, context, sent)
    for item in result["items"]:
        item["prompt"] = prompt["id"]
    result["raw"] = raw
    return result


def merge_generated(doc, items, generator, prompt_ids):
    """并入新生成的文字：同一提示词下的待审/打回会被替换，已通过和人工改过的保留。"""
    texts = doc.get("texts") or {}
    old = texts.get("items") or []
    prompt_ids = set(prompt_ids)

    def keep(item):
        if item.get("status") == "approved" or item.get("source") == "human":
            return True
        return item.get("prompt") not in prompt_ids and item.get("prompt") is not None

    now = int(time.time())
    fresh = []
    for n, item in enumerate(items):
        fresh.append({
            "id": "t%d_%d" % (now, n),
            "kind": item.get("kind") or "qa",
            "prompt": item.get("prompt"),
            "question": item["question"],
            "answer": item["answer"],
            "source": "model",
            "status": "pending",
            "draft": {"question": item["question"], "answer": item["answer"]},
        })
    doc["texts"] = {
        "generator": generator,
        "generated_at": now,
        "reviewed_at": texts.get("reviewed_at"),
        "items": [item for item in old if keep(item)] + fresh,
    }
    return doc


def clean_items(rows):
    """人工提交的文字条目，做字段清洗。"""
    out = []
    seen = set()
    for n, row in enumerate(rows or []):
        question = str(row.get("question") or "").strip()[:4000]
        answer = str(row.get("answer") or "").strip()[:8000]
        if not question and not answer:
            continue
        rid = str(row.get("id") or "h%d_%d" % (int(time.time()), n))
        if rid in seen:
            rid = rid + "_" + str(n)
        seen.add(rid)
        status = row.get("status") if row.get("status") in STATUSES else "pending"
        draft = row.get("draft") if isinstance(row.get("draft"), dict) else None
        source = "human" if row.get("source") == "human" else "model"
        if draft and (draft.get("question") != question or draft.get("answer") != answer):
            source = "human"
        item = {
            "id": rid,
            "kind": str(row.get("kind") or "qa")[:32],
            "question": question,
            "answer": answer,
            "source": source,
            "status": status,
        }
        if row.get("prompt"):
            item["prompt"] = str(row["prompt"])[:64]
        if draft:
            item["draft"] = {
                "question": str(draft.get("question") or ""),
                "answer": str(draft.get("answer") or ""),
            }
        out.append(item)
    return out


# ---------- 后台任务 ----------

def stop_job():
    """停掉队列，并断开当前请求，不用等模型算完。"""
    JOB["stop"] = True
    JOB["canceled"] = True
    conn = JOB.get("conn")
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def start_job(project_id, tasks, worker):
    """tasks: [(image_path, ann_path)]；worker(image_path, ann_path) 生成并写盘。"""
    with JOB_LOCK:
        if JOB["running"]:
            return False
        JOB.update({
            "running": True,
            "project": project_id,
            "total": len(tasks),
            "done": 0,
            "failed": 0,
            "current": "",
            "error": None,
            "started": time.time(),
            "finished": 0,
            "stop": False,
            "canceled": False,
            "conn": None,
        })

    def run():
        try:
            for image_path, ann_path in tasks:
                if JOB["stop"]:
                    JOB["canceled"] = True
                    break
                JOB["current"] = Path(image_path).name
                try:
                    worker(image_path, ann_path)
                    JOB["done"] += 1
                except Canceled:
                    JOB["canceled"] = True
                    break
                except Exception as exc:  # noqa: BLE001
                    if JOB.get("stop"):
                        JOB["canceled"] = True
                        break
                    JOB["failed"] += 1
                    JOB["error"] = Path(image_path).name + "：" + str(exc)
                    JOB["done"] += 1
                finally:
                    gc.collect()
        finally:
            JOB["running"] = False
            JOB["current"] = ""
            JOB["finished"] = time.time()
            JOB["conn"] = None
            gc.collect()

    threading.Thread(target=run, daemon=True).start()
    return True


def job_state():
    total = JOB["total"] or 0
    done = JOB["done"]
    started = JOB["started"] or 0
    ended = JOB["finished"] or (time.time() if JOB["running"] or started else 0)
    elapsed = int(max(0, ended - started)) if started else 0
    return {
        "running": JOB["running"],
        "project": JOB["project"],
        "total": total,
        "done": done,
        "failed": JOB["failed"],
        "current": JOB["current"],
        "error": JOB["error"],
        "canceled": bool(JOB.get("canceled")),
        "stopping": bool(JOB.get("stop") and JOB["running"]),
        "percent": int(done * 100 / total) if total else 0,
        "elapsed": elapsed,
    }
