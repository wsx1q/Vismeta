"""Run with python -m unittest discover -s scripts -p test_dataset_import.py."""
import io
import json
import tempfile
import unittest
import zipfile
import shutil
import threading
import time
from unittest.mock import patch
from pathlib import Path

from PIL import Image
from dataset_import import import_tree, extract_archive


class DatasetImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / 'source'
        self.src.mkdir()
        self.project = {key: str(self.root / 'project' / value) for key, value in
                        [('images', 'images'), ('annotations', 'annotations'), ('labels', 'labels.json')]}

    def image(self, name):
        p = self.src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (200, 100)).save(p)
        return p

    def text(self, name, value):
        p = self.src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value, encoding='utf-8')

    def docs(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in Path(self.project['annotations']).glob('*.json')]

    def test_detection_and_segmentation(self):
        self.image('images/train/a.jpg')
        self.image('images/val/b.jpg')
        self.text('labels/train/a.txt', '0 0.5 0.5 0.4 0.2\n')
        self.text('labels/val/b.txt', '1 0 0 1 0 1 1\n')
        self.text('data.yaml', 'names: [car, person]\n')
        result = import_tree(self.src, self.project)
        self.assertEqual((result['count'], result['annotated'], result['shapes']), (2, 2, 2))
        docs = {d['imagePath']: d for d in self.docs()}
        self.assertEqual(docs['a.jpg']['shapes'][0]['points'], [[60, 40], [140, 60]])
        self.assertEqual(docs['b.jpg']['shapes'][0]['points'], [[0, 0], [200, 0], [200, 100]])
        self.assertEqual(docs['b.jpg']['shapes'][0]['label'], 'person')

    def test_missing_class_table_and_invalid_coords_do_not_import_unlabeled(self):
        self.image('a.jpg'); self.text('a.txt', '0 0.5 0.5 0.2 0.2')
        with self.assertRaisesRegex(ValueError, '导入前检查未通过'):
            import_tree(self.src, self.project)
        self.text('classes.txt', 'car')
        for value in ('nan', '1.2'):
            self.text('a.txt', f'0 {value} 0.5 0.2 0.2')
            self.assertEqual(import_tree(self.src, self.project)['count'], 0)

    def test_empty_and_plain_images(self):
        self.image('a.jpg'); self.text('a.txt', '')
        self.image('b.jpg')
        result = import_tree(self.src, self.project)
        self.assertEqual((result['count'], result['empty_annotations']), (2, 1))

    def test_labelme_collision_and_original_unchanged(self):
        original = {'shapes': [{'label': 'car', 'shape_type': 'rectangle', 'points': [[1, 2], [30, 40]]}], 'imageData': 'embedded'}
        for split in ('train', 'val'):
            self.image(f'{split}/same.jpg')
            self.text(f'{split}/same.json', json.dumps(original))
        result = import_tree(self.src, self.project)
        self.assertEqual((result['count'], result['shapes']), (2, 2))
        self.assertEqual({d['imagePath'] for d in self.docs()}, {'same.jpg', 'same_2.jpg'})
        self.assertTrue(all(d['imageData'] is None for d in self.docs()))
        self.assertEqual(json.loads((self.src / 'train/same.json').read_text()), original)

    def test_zip_preserves_layout_rejects_traversal(self):
        image = io.BytesIO(); Image.new('RGB', (200, 100)).save(image, format='JPEG')
        archive = self.root / 'data.zip'
        with zipfile.ZipFile(archive, 'w') as arc:
            arc.writestr('dataset/images/train/a.jpg', image.getvalue())
            arc.writestr('dataset/labels/train/a.txt', '0 .5 .5 .2 .2')
            arc.writestr('dataset/data.yaml', 'names: {0: car}')
            arc.writestr('../escape.txt', 'bad')
        extract_archive(archive, self.src)
        self.assertFalse((self.root / 'escape.txt').exists())
        self.assertEqual(import_tree(self.src, self.project)['annotated'], 1)

    def test_parallel_copy_is_bounded_and_same_stems_remain_unique(self):
        for i in range(24):
            self.image(f'{i}/same.jpg')
            self.text(f'{i}/same.json', json.dumps({'shapes': [{'label': str(i), 'shape_type': 'rectangle', 'points': [[1, 1], [10, 9]]}]}))
        original = shutil.copy2
        active = peak = 0
        lock = threading.Lock()
        def copy(src, dst):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            try:
                time.sleep(.01)
                return original(src, dst)
            finally:
                with lock: active -= 1
        reports = []
        with patch.dict('os.environ', {'IMPORT_WORKERS': '4'}), patch('dataset_import.shutil.copy2', side_effect=copy):
            result = import_tree(self.src, self.project, progress=lambda **kw: reports.append(kw))
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 4)
        self.assertEqual((result['count'], result['shapes'], result['skipped']), (24, 24, 0))
        docs = self.docs()
        self.assertEqual(len({d['imagePath'] for d in docs}), 24)
        self.assertEqual({d['shapes'][0]['label'] for d in docs}, {str(i) for i in range(24)})
        self.assertEqual([r['processed'] for r in reports if r['phase'] == 'importing' and r['processed'] > 0], list(range(1, 25)))

    def test_copy_failure_removes_partial_destination_and_keeps_other_images(self):
        self.image('bad.jpg'); self.image('good.jpg')
        original = shutil.copy2
        def copy(src, dst):
            if src.name == 'bad.jpg':
                dst.write_bytes(b'partial')
                raise OSError('simulated disk failure')
            return original(src, dst)
        with patch('dataset_import.shutil.copy2', side_effect=copy):
            result = import_tree(self.src, self.project)
        self.assertEqual((result['count'], result['skipped']), (1, 1))
        self.assertFalse((Path(self.project['images'])/'bad.jpg').exists())
        self.assertTrue((Path(self.project['images'])/'good.jpg').exists())

    def test_preflight_missing_classes_moves_nothing_and_stops_at_first_file(self):
        for i in range(4):
            self.image(f'images/{i}.jpg')
            self.text(f'labels/{i}.txt', '0 .5 .5 .2 .2')
        reports = []
        with patch('dataset_import.shutil.copy2') as copier, patch.object(Path, 'replace') as mover:
            with self.assertRaisesRegex(ValueError, '缺少 YOLO 类别表'):
                import_tree(self.src, self.project, transfer='move', progress=lambda **kw: reports.append(kw))
        copier.assert_not_called(); mover.assert_not_called()
        self.assertEqual(len(list((self.src/'images').glob('*.jpg'))), 4)
        self.assertEqual(list(Path(self.project['images']).iterdir()), [])
        self.assertFalse((Path(self.project['images']).parent/'move-import.jsonl').exists())
        self.assertEqual(reports[-1]['phase'], 'validating')
        self.assertFalse(any(r['phase'] == 'importing' for r in reports))

    def test_preflight_bad_class_in_later_file_blocks_the_entire_move(self):
        self.image('images/a.jpg'); self.image('images/z.jpg')
        self.text('labels/a.txt', '0 .5 .5 .2 .2')
        self.text('labels/z.txt', '2 .5 .5 .2 .2')
        self.text('classes.txt', 'car')
        with self.assertRaisesRegex(ValueError, '类别编号 2'):
            import_tree(self.src, self.project, transfer='move')
        self.assertTrue((self.src/'images/a.jpg').exists())
        self.assertEqual(list(Path(self.project['images']).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
