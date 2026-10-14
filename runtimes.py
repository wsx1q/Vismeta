import numpy as np
import cv2
from pathlib import Path

TARGET_SIZE = 1024
INPUT_SIZE = (684, 1024)


def as_rgb(array):
    image = np.asarray(array)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def preprocess_shape(oldh, oldw, long_side):
    scale = long_side * 1.0 / max(oldh, oldw)
    return int(oldh * scale + 0.5), int(oldw * scale + 0.5)


def apply_coords(coords, original_size, target_length):
    old_h, old_w = original_size
    new_h, new_w = preprocess_shape(old_h, old_w, target_length)
    coords = np.array(coords, dtype=np.float32, copy=True)
    coords[..., 0] *= new_w / old_w
    coords[..., 1] *= new_h / old_h
    return coords


class OnnxSam1:
    def __init__(self):
        self.key = None
        self.session_key = None
        self.encoder = None
        self.decoder = None
        self.embedding = None
        self.transform = None

    def _sessions(self, files):
        key = (files["encoder"], files["decoder"])
        if self.session_key == key and self.encoder is not None:
            return
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(files["encoder"], options, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(files["decoder"], options, providers=["CPUExecutionProvider"])
        self.session_key = key
        self.key = None

    def _encode(self, array):
        image = as_rgb(array)
        height, width = image.shape[:2]
        key = (self.session_key, id(array), image.shape)
        if self.key == key and self.embedding is not None:
            return height, width
        scale = min(INPUT_SIZE[1] / width, INPUT_SIZE[0] / height)
        transform = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
        warped = cv2.warpAffine(image, transform[:2], (INPUT_SIZE[1], INPUT_SIZE[0]), flags=cv2.INTER_LINEAR)
        name = self.encoder.get_inputs()[0].name
        self.embedding = self.encoder.run(None, {name: warped.astype(np.float32)})[0]
        self.transform = transform
        self.key = key
        return height, width

    def masks(self, array, points, box, files):
        self._sessions(files)
        height, width = self._encode(array)
        coords = []
        labels = []
        if box:
            x1, y1, x2, y2 = box
            coords.extend([[float(x1), float(y1)], [float(x2), float(y2)]])
            labels.extend([2, 3])
        for point in points or []:
            coords.append([float(point["x"]), float(point["y"])])
            labels.append(1 if point.get("positive", True) else 0)
        if not coords:
            coords = [[0.0, 0.0]]
            labels = [-1]
        coords.append([0.0, 0.0])
        labels.append(-1)
        onnx_coord = apply_coords(np.array(coords, np.float32)[None], INPUT_SIZE, TARGET_SIZE)
        ones = np.ones((1, onnx_coord.shape[1], 1), np.float32)
        onnx_coord = np.matmul(np.concatenate([onnx_coord, ones], axis=2), self.transform.T)[:, :, :2]
        outputs = self.decoder.run(None, {
            "image_embeddings": self.embedding,
            "point_coords": onnx_coord.astype(np.float32),
            "point_labels": np.array([labels], np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array(INPUT_SIZE, np.float32),
        })
        raw = np.asarray(outputs[0])
        scores = np.asarray(outputs[1]).reshape(-1) if len(outputs) > 1 else np.ones((raw.shape[1],), np.float32)
        if raw.ndim == 3:
            raw = raw[None]
        inv = np.linalg.inv(self.transform)
        masks = [
            cv2.warpAffine(mask.astype(np.float32), inv[:2], (width, height), flags=cv2.INTER_LINEAR)
            for mask in raw[0]
        ]
        return np.stack(masks, axis=0), scores


class TensorRtSam:
    def __init__(self):
        self.key = None
        self.engines = None
        self.embedding = None

    def _load(self, files):
        key = (files["encoder"], files["decoder"])
        if self.key == key and self.engines is not None:
            return
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        loaded = []
        for path in (files["encoder"], files["decoder"]):
            blob = Path(path).read_bytes()
            loaded.append(runtime.deserialize_cuda_engine(blob))
        self.engines = loaded
        self.key = key
        self.embedding = None

    def masks(self, array, points, box, files):
        self._load(files)
        raise RuntimeError("TensorRT 引擎已读到，但这套 SAM3 还没有官方的 encoder/decoder 引擎可跑。请把导出的 engine 按清单导入后再启用。")


ONNX = OnnxSam1()
TRT = TensorRtSam()
