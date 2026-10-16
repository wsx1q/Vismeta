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
