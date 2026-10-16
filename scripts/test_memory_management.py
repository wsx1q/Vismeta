import ast
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import server


@unittest.skipIf(server.torch is None, 'PyTorch required for SAM3 cache tests')
class ImageCacheTest(unittest.TestCase):
    def test_cache_hit_does_not_decode_and_each_mode_keeps_only_its_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jpg'
            Image.new('RGB', (200, 100)).save(path)
            def encode(image):
                return {'original_width': 200, 'original_height': 100,
                        'backbone_out': {'sam2_backbone_out': object(), 'vision_features': object()}}
            processor = Mock(); processor.set_image.side_effect = encode
            state = {'sam3_processor': processor}
            with patch.object(server, 'STATE', state):
                first = server.sam3_state(path)
                with patch.object(server.Image, 'open', side_effect=AssertionError('redecoded cache hit')):
                    self.assertIs(first, server.sam3_state(path))
                self.assertEqual(set(first['backbone_out']), {'sam2_backbone_out'})
                text = server.sam3_state(path, text=True)
                self.assertEqual(set(text['backbone_out']), {'vision_features'})
                state['sam3_epoch'] = 1
                self.assertIsNot(text, server.sam3_state(path, text=True))
                self.assertEqual(processor.set_image.call_count, 3)

    def test_idle_cleanup_retains_model_and_respects_active_inference_lock(self):
        model = object()
        state = {'sam3_model': model, 'sam3_processor': object(), 'sam3_state': object(),
                 'sam3_text_state': object(), 'image_last_used': 100}
        with patch.object(server, 'STATE', state), patch.object(server.config, 'SAM_CACHE_IDLE_SECONDS', 120), \
                patch.object(server, 'release_cuda') as release, patch.object(server, 'trim_cpu_memory'):
            self.assertFalse(server.expire_image_cache(now=219))
            with server.LOCK:
                self.assertFalse(server.expire_image_cache(now=221))
            self.assertTrue(server.expire_image_cache(now=221))
            self.assertIsNone(state['sam3_state'])
            self.assertIsNone(state['sam3_text_state'])
            self.assertIs(state['sam3_model'], model)
            release.assert_called_once()


@unittest.skipIf(server.torch is None or importlib.util.find_spec('sam3') is None, 'SAM3 GPU image required')
class CompactMasksTest(unittest.TestCase):
    def setUp(self):
        import torch
        from torch.nn.functional import interpolate
        self.torch, self.interpolate = torch, interpolate
        root = Path(importlib.util.find_spec('sam3').origin).parent
        module = ast.parse((root / 'model' / 'sam3_image_processor.py').read_text())
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'Sam3Processor')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_grounding')
        method.decorator_list = []
        def boxes(x):
            c, size = x[..., :2], x[..., 2:]
            return torch.cat([c-size/2, c+size/2], dim=-1)
        self.scope = {'torch': torch, 'Dict': dict, 'interpolate': interpolate,
                      'box_ops': types.SimpleNamespace(box_cxcywh_to_xyxy=boxes)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<processor>', 'exec'), self.scope)
        self.logits = torch.tensor([[[[-2., 2.], [2., -2.]], [[1., -1.], [-1., 1.]]]])
        outputs = {'pred_boxes': torch.tensor([[[.5, .5, .4, .2], [.4, .4, .2, .2]]]),
                   'pred_logits': torch.ones(1, 2, 1)*10, 'presence_logit_dec': torch.ones(1)*10,
                   'pred_masks': self.logits}
        self.processor = types.SimpleNamespace(model=types.SimpleNamespace(forward_grounding=lambda **kw: outputs),
                                               find_stage=None, confidence_threshold=.4, device='cpu')

    def run_grounding(self, box_only=False):
        return self.scope['_forward_grounding'](self.processor, {'backbone_out': {}, 'geometric_prompt': None,
                  'original_width': 20, 'original_height': 10, 'boxes_only': box_only})

    def test_box_only_does_not_interpolate_masks(self):
        self.scope['interpolate'] = Mock(side_effect=AssertionError('expanded masks for a box request'))
        result = self.run_grounding(True)
        self.assertIsNone(result['masks'])
        self.assertEqual(tuple(result['boxes'].shape), (2, 4))

    def test_mask_pixels_equal_batched_reference_without_retained_float_masks(self):
        result = self.run_grounding()
        expected = self.interpolate(self.logits[0].unsqueeze(1), (10, 20), mode='bilinear', align_corners=False).sigmoid() > .5
        self.assertTrue(self.torch.equal(result['masks'], expected))
        self.assertIsNone(result['masks_logits'])
        self.assertEqual(result['masks'].dtype, self.torch.bool)

    def test_empty_detections_keep_valid_empty_mask_shape(self):
        self.processor.confidence_threshold = 1
        self.assertEqual(tuple(self.run_grounding()['masks'].shape), (0, 1, 10, 20))
