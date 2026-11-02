"""Ensure legacy embedded images never leak back into API responses or exports."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import server


class AnnotationStorageTest(unittest.TestCase):
    def test_labelme_import_normalizes_four_corners_without_changing_source(self):
        from dataset_import import import_tree
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            Image.new('RGB', (200, 100)).save(source / 'test.jpg')
            original = json.dumps({'shapes': [{'label': 'cone', 'shape_type': 'rectangle',
                                              'points': [[10, 20], [30, 20], [30, 70], [10, 70]]}]})
            (source / 'test.json').write_text(original)
            project = {'images': str(root / 'project/images'), 'annotations': str(root / 'project/annotations'),
                       'labels': str(root / 'project/labels.json')}
            result = import_tree(source, project)
            self.assertEqual(result['count'], 1)
            imported = json.loads((Path(project['annotations']) / 'test.json').read_text())
            self.assertEqual(imported['shapes'][0]['points'], [[10, 20], [30, 70]])
            self.assertEqual((source / 'test.json').read_text(), original)

    def test_four_corner_rectangle_is_read_saved_and_exported_as_opposite_corners(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.multiple(server, DATA_ROOT=root, PROJECTS_DIR=root / 'projects', PROJECTS_FILE=root / 'projects.json'):
                client = server.app.test_client()
                pid = client.post('/api/projects', json={'name': 'four corners'}).get_json()['id']
                headers = {'X-Project': pid}
                project = server.find_project(pid)
                Image.new('RGB', (200, 100)).save(Path(project['images']) / 'test.jpg')
                shape = {'label': 'cone', 'shape_type': 'rectangle',
                         'points': [[10, 20], [30, 20], [30, 70], [10, 70]]}
                ann = Path(project['annotations']) / 'test.json'
                original = json.dumps({'shapes': [shape], 'imageWidth': 200, 'imageHeight': 100})
                ann.write_text(original)
                expected = [[10, 20], [30, 70]]
                self.assertEqual(client.get('/api/annotation/test.jpg', headers=headers).get_json()['shapes'][0]['points'], expected)
                self.assertEqual(ann.read_text(), original, 'reading must preserve the source file')
                response = client.get('/api/export/labelme.zip', headers=headers)
                with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                    self.assertEqual(json.loads(archive.read('annotations/test.json'))['shapes'][0]['points'], expected)
                response.close()
                response = client.post('/api/save', headers=headers, json={'name': 'test.jpg', 'shapes': [shape]})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(json.loads(ann.read_text())['shapes'][0]['points'], expected)

    def test_legacy_image_data_is_removed_on_read_export_and_text_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.multiple(server, DATA_ROOT=root, PROJECTS_DIR=root / 'projects', PROJECTS_FILE=root / 'projects.json'):
                client = server.app.test_client()
                project_id = client.post('/api/projects', json={'name': 'storage check'}).get_json()['id']
                headers = {'X-Project': project_id}
                project = server.find_project(project_id)
                Image.new('RGB', (200, 100)).save(Path(project['images']) / 'test.jpg')
                shape = {'label': 'car', 'shape_type': 'rectangle', 'points': [[10, 10], [20, 20]]}
                ann = Path(project['annotations']) / 'test.json'
                ann.write_text(json.dumps({'shapes': [shape], 'imageData': 'LEGACY_BASE64_PIXELS',
                                           'imagePath': 'test.jpg', 'imageWidth': 200, 'imageHeight': 100}))
                response = client.get('/api/annotation/test.jpg', headers=headers)
                self.assertIsNone(response.get_json()['imageData'])
                response = client.get('/api/export/labelme.zip', headers=headers)
                with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                    doc = json.loads(archive.read('annotations/test.json'))
                response.close()
                self.assertIsNone(doc['imageData'])
                self.assertEqual(doc['shapes'], [shape])
                response = client.post('/api/texts/save', headers=headers, json={'name': 'test.jpg', 'items': []})
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(json.loads(ann.read_text())['imageData'])


if __name__ == '__main__':
    unittest.main()
