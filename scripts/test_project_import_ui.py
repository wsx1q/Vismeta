import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import server


class ProjectImportTest(unittest.TestCase):
    def test_project_entries_do_not_wait_for_annotation_statistics(self):
        with patch.object(server, 'project_stats', side_effect=AssertionError('must not scan annotations')):
            response = self.client.get('/api/projects?summary=0')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['projects'], [
            {'id': self.pid, 'name': 'original', 'builtin': False}])
        with patch.object(server, 'project_stats', return_value={'images': 2, 'annotated': 1, 'shapes': 3}):
            self.assertEqual(self.client.get('/api/projects').get_json()['projects'][0]['images'], 2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = patch.multiple(server, DATA_ROOT=self.root, PROJECTS_DIR=self.root / 'projects',
                                       PROJECTS_FILE=self.root / 'projects.json')
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.client = server.app.test_client()
        self.pid = self.client.post('/api/projects', json={'name': 'original'}).get_json()['id']

    def test_rename_retains_paths_metadata_and_files(self):
        rows = server.load_projects()
        rows[0]['vlm'] = {'provider': 'configured-model'}
        server.save_projects(rows)
        project = server.find_project(self.pid)
        image = Path(project['images']) / 'test.jpg'
        Image.new('RGB', (200, 100)).save(image)
        response = self.client.patch('/api/projects/' + self.pid, json={'name': '  安全帽与反光衣  '})
        self.assertEqual(response.status_code, 200)
        renamed = server.find_project(self.pid)
        self.assertEqual(renamed['name'], '安全帽与反光衣')
        for key in ('id', 'images', 'annotations', 'labels', 'vlm'):
            self.assertEqual(renamed[key], project[key])
        self.assertTrue(image.exists())

    def test_rename_rejects_duplicate_blank_long_or_missing_project(self):
        with patch.object(server.time, 'time', return_value=1000):
            self.client.post('/api/projects', json={'name': 'existing'})
        for name in ('existing', '   ', 'a'*41):
            self.assertEqual(self.client.patch('/api/projects/' + self.pid, json={'name': name}).status_code, 400)
        self.assertEqual(self.client.patch('/api/projects/missing', json={'name': 'valid'}).status_code, 404)
        self.assertEqual(server.find_project(self.pid)['name'], 'original')

    def test_nested_local_yolo_upload_and_manual_classes_use_data_disk_staging(self):
        image = io.BytesIO()
        Image.new('RGB', (200, 100)).save(image, format='JPEG')
        image.seek(0)
        response = self.client.post('/api/import/files', headers={'X-Project': self.pid}, data={
            'files': [(image, 'my_dataset/images/train/a.jpg'),
                      (io.BytesIO(b'0 .5 .5 .4 .2'), 'my_dataset/labels/train/a.txt'),
                      (io.BytesIO(b'helmet\nvest\n'), 'classes.txt')],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.get_json()['count'], response.get_json()['shapes']), (1, 1))
        project = server.find_project(self.pid)
        doc = json.loads((Path(project['annotations']) / 'a.json').read_text())
        self.assertEqual(doc['shapes'][0]['label'], 'helmet')
        self.assertEqual(doc['shapes'][0]['points'], [[60, 40], [140, 60]])
        self.assertIsNone(doc['imageData'])
        self.assertTrue((self.root / '.imports').is_dir())
        self.assertEqual(list((self.root / '.imports').iterdir()), [])

    def test_yolo_zip_with_class_table_upload(self):
        image = io.BytesIO()
        Image.new('RGB', (200, 100)).save(image, format='JPEG')
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, 'w') as archive:
            archive.writestr('dataset/images/a.jpg', image.getvalue())
            archive.writestr('dataset/labels/a.txt', '1 .5 .5 .2 .2')
            archive.writestr('dataset/data.yaml', 'names: [helmet, vest]\n')
        payload.seek(0)
        response = self.client.post('/api/import/files', headers={'X-Project': self.pid},
                                    data={'files': [(payload, 'dataset.zip')]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['annotated'], 1)
        project = server.find_project(self.pid)
        doc = json.loads((Path(project['annotations']) / 'a.json').read_text())
        self.assertEqual(doc['shapes'][0]['label'], 'vest')
