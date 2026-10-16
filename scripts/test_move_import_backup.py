import errno
import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import server
import project_backups
from dataset_import import import_tree

class MoveBackupTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir()
        self.project={'id':'move-test','name':'Move test','images':str(self.root/'projects/move-test/images'),
                      'annotations':str(self.root/'projects/move-test/annotations'),'labels':str(self.root/'projects/move-test/labels.json')}
        self.image=self.source/'one.jpg';Image.new('RGB',(20,10)).save(self.image)
        (self.source/'one.txt').write_text('0 .5 .5 .2 .2');(self.source/'classes.txt').write_text('helmet\n')
        self.client=server.app.test_client()
        self.project_patch=patch.object(server,'current_project',return_value=self.project)
        self.project_patch.start();self.addCleanup(self.project_patch.stop)
        self.data_patch=patch.object(server,'DATA_ROOT',self.root)
        self.data_patch.start();self.addCleanup(self.data_patch.stop)

    def test_move_preserves_inode_without_copy_and_records_source(self):
        inode=self.image.stat().st_ino
        with patch('dataset_import.shutil.copy2',side_effect=AssertionError('must not copy')):
            result=import_tree(self.source,self.project,transfer='move')
        self.assertEqual((result['count'],result['shapes']),(1,1))
        dest=Path(self.project['images'])/'one.jpg'
        self.assertFalse(self.image.exists());self.assertEqual(dest.stat().st_ino,inode)
        row=json.loads((dest.parent.parent/'move-import.jsonl').read_text())
        self.assertEqual(row['source'],str(self.image));self.assertEqual(row['image'],'one.jpg')
        doc=json.loads((Path(self.project['annotations'])/'one.json').read_text())
        self.assertIsNone(doc['imageData'])

    def test_annotation_failure_restores_moved_source(self):
        original=Path.write_text
        def write(path,*args,**kwargs):
            if path.name=='one.json':raise OSError('simulated annotation write failure')
            return original(path,*args,**kwargs)
        with patch.object(Path,'write_text',new=write):
            result=import_tree(self.source,self.project,transfer='move')
        self.assertEqual(result['skipped'],1);self.assertTrue(self.image.exists())
        self.assertFalse((Path(self.project['images'])/'one.jpg').exists())

    def test_cross_mount_error_never_falls_back_to_copy(self):
        with patch.object(Path,'replace',side_effect=OSError(errno.EXDEV,'cross mount')),patch('dataset_import.shutil.copy2',side_effect=AssertionError('must not copy')):
            result=import_tree(self.source,self.project,transfer='move')
        self.assertEqual(result['skipped'],1);self.assertTrue(self.image.exists())
        self.assertFalse((Path(self.project['images'])/'one.jpg').exists())

    def test_move_api_restricts_source_and_supports_copy(self):
        with patch.dict(os.environ,{'MOVE_IMPORT_ROOT':str(self.source),'IMPORT_MODE':'move'}):
            blocked=self.client.post('/api/import/path',json={'path':str(self.root),'mode':'move'})
            self.assertEqual(blocked.status_code,400)
            ok=self.client.post('/api/import/path',json={'path':str(self.source),'mode':'move'})
            self.assertEqual(ok.status_code,200);self.assertEqual(ok.get_json()['count'],1)

    def wait_backup(self):
        for _ in range(500):
            row=project_backups.status(self.root,self.project['id'])
            if row['status'] not in {'running','queued'}:return row
            time.sleep(.01)
        self.fail('backup did not finish')

    def test_export_queues_full_backup_and_unchanged_export_reuses_archive(self):
        import_tree(self.source,self.project,transfer='move')
        response=self.client.get('/api/export/yolo.zip')
        self.assertEqual(response.status_code,200)
        row=self.wait_backup();self.assertEqual(row['status'],'complete',row)
        archive=Path(row['archive']);self.assertTrue(archive.is_relative_to(self.root/'backups'))
        with zipfile.ZipFile(archive) as z:
            self.assertIn('images/one.jpg',z.namelist());self.assertIn('annotations/one.json',z.namelist())
            self.assertIn('move-import.jsonl',z.namelist())
            self.assertIsNone(json.loads(z.read('annotations/one.json'))['imageData'])
        response.close()
        second=project_backups.schedule(self.root,self.project);row2=self.wait_backup()
        self.assertEqual(row2['archive'],str(archive));self.assertTrue(row2['reused'])
        self.assertEqual(self.client.get('/api/backup/status').get_json()['status'],'complete')

    def test_export_during_import_does_not_backup_a_partial_project(self):
        with patch.object(server, 'IMPORT_JOBS', {'active': {'project_id': self.project['id'], 'status': 'running'}}):
            response = self.client.get('/api/export/yolo.zip')
        self.assertEqual(response.status_code, 409)
        self.assertFalse((self.root/'backups').exists())

    def test_plain_copy_does_not_create_backup_automatically(self):
        import_tree(self.source,self.project)
        self.assertIsNone(project_backups.schedule(self.root,self.project))
        self.assertTrue(self.image.exists())

if __name__=='__main__':unittest.main()
