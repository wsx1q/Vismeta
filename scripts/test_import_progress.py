import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from dataset_import import import_tree
import server

class ImportProgressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.project = {'id':'progress-project', 'images':str(self.root/'out/images'),
                        'annotations':str(self.root/'out/annotations'), 'labels':str(self.root/'out/labels.json')}
        self.client = server.app.test_client()
        self.find = patch.object(server, 'current_project', return_value=self.project)
        self.find.start(); self.addCleanup(self.find.stop)
        with server.IMPORT_JOBS_LOCK:
            server.IMPORT_JOBS.clear()

    def test_progress_counts_failures_and_cap(self):
        (self.source/'classes.txt').write_text('helmet\n')
        for name in ['a','b','c']:
            Image.new('RGB',(20,10)).save(self.source/(name+'.jpg'))
            (self.source/(name+'.txt')).write_text('0 .5 .5 .2 .2' if name != 'b' else '0 2 .5 .2 .2')
        reports=[]
        result=import_tree(self.source,self.project,cap=2,progress=lambda **kw:reports.append(kw))
        self.assertEqual(reports[0]['phase'],'scanning')
        self.assertEqual(reports[-1]['processed'],2)
        self.assertEqual(reports[-1]['total'],2)
        self.assertEqual(reports[-1]['count'],1)
        self.assertEqual(reports[-1]['skipped'],1)
        self.assertEqual(result['count'],1)
        self.assertEqual(result['skipped'],2) # invalid annotation and cap notice

    def test_background_status_duplicate_and_project_isolation(self):
        started=threading.Event(); finish=threading.Event()
        def fake_import(src, project, cap, progress):
            progress(phase='importing',processed=1,total=2,count=1,skipped=0,shapes=1)
            started.set(); finish.wait(5)
            return {'count':2,'annotated':2,'shapes':2,'skipped':0,'errors':[]}
        with patch('dataset_import.import_tree', side_effect=fake_import):
            try:
                response=self.client.post('/api/import/path',json={'path':str(self.source),'async':True})
                self.assertEqual(response.status_code,202)
                job=response.get_json()['id']; self.assertTrue(started.wait(2))
                row=self.client.get('/api/import/jobs/'+job).get_json()
                self.assertEqual((row['processed'],row['total'],row['status']),(1,2,'running'))
                again=self.client.post('/api/import/path',json={'path':str(self.source),'async':True})
                self.assertEqual(again.get_json()['id'],job)
                with patch.object(server,'current_project',return_value={'id':'other'}):
                    self.assertEqual(self.client.get('/api/import/jobs/'+job).status_code,404)
            finally:
                finish.set()
            for _ in range(100):
                row=self.client.get('/api/import/jobs/'+job).get_json()
                if row['status'] != 'running': break
                time.sleep(.01)
            self.assertEqual(row['status'],'complete')
            self.assertEqual(row['result']['count'],2)

    def test_background_failure_is_reported(self):
        with patch('dataset_import.import_tree',side_effect=OSError('test failure')):
            response=self.client.post('/api/import/path',json={'path':str(self.source),'async':True})
            job=response.get_json()['id']
            for _ in range(100):
                row=self.client.get('/api/import/jobs/'+job).get_json()
                if row['status'] != 'running': break
                time.sleep(.01)
            self.assertEqual(row['status'],'failed')
            self.assertIn('test failure',row['error'])

if __name__ == '__main__': unittest.main()
