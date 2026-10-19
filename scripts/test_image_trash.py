import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import image_trash
import server

class ImageTrashTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.project={'id':'trash-test','images':str(self.root/'project/images'),'annotations':str(self.root/'project/annotations'),'labels':str(self.root/'project/labels.json')}
        for key in ['images','annotations']:Path(self.project[key]).mkdir(parents=True)
        self.client=server.app.test_client()
        p=patch.object(server,'current_project',return_value=self.project);p.start();self.addCleanup(p.stop)
        p=patch.object(server,'DATA_ROOT',self.root);p.start();self.addCleanup(p.stop)
        self.image('a.jpg');self.image('b.jpg')

    def image(self,name):
        Image.new('RGB',(20,10)).save(Path(self.project['images'])/name)
        (Path(self.project['annotations'])/(Path(name).stem+'.json')).write_text(json.dumps({'shapes':[{'label':'test','points':[[1,1],[5,5]],'shape_type':'rectangle'}],'texts':{'items':[{'answer':'kept','status':'approved'}]}}))

    def test_batch_delete_and_restore_keep_image_annotation_and_text(self):
        before={p.name:p.read_bytes() for p in Path(self.project['annotations']).iterdir()}
        response=self.client.post('/api/images/delete',json={'names':['a.jpg','b.jpg']})
        self.assertEqual(response.status_code,200);batch=response.get_json()['batch']
        self.assertEqual(self.client.get('/api/images').get_json()['total'],0)
        rows=self.client.get('/api/images/trash').get_json()['batches'];self.assertEqual(rows[0]['count'],2)
        response=self.client.post('/api/images/restore',json={'batch':batch})
        self.assertEqual(response.status_code,200);self.assertEqual(response.get_json()['count'],2)
        self.assertEqual(self.client.get('/api/images').get_json()['total'],2)
        for name,contents in before.items():self.assertEqual((Path(self.project['annotations'])/name).read_bytes(),contents)
        self.assertEqual(self.client.get('/api/images/trash').get_json()['batches'],[])

    def test_rejects_traversal_and_prevalidates_entire_batch(self):
        for names in [['../outside.jpg'],['a.jpg','missing.jpg'],['..\\outside.jpg'],[]]:
            self.assertEqual(self.client.post('/api/images/delete',json={'names':names}).status_code,400)
        self.assertTrue((Path(self.project['images'])/'a.jpg').is_file())
        self.assertEqual(image_trash.list_batches(self.project),[])

    def test_restore_conflict_does_not_overwrite(self):
        batch=image_trash.remove(self.project,['a.jpg'])['batch']
        self.image('a.jpg');current=(Path(self.project['images'])/'a.jpg').read_bytes()
        response=self.client.post('/api/images/restore',json={'batch':batch})
        self.assertEqual(response.status_code,400)
        self.assertEqual((Path(self.project['images'])/'a.jpg').read_bytes(),current)
        self.assertEqual(image_trash.list_batches(self.project)[0]['count'],1)

    def test_mid_batch_failure_rolls_back(self):
        original=Path.replace
        def replace(src,dest):
            if src.name=='b.json' and '.trash' in Path(dest).parts:raise OSError('simulated failure')
            return original(src,dest)
        with patch.object(Path,'replace',new=replace):
            with self.assertRaises(OSError):image_trash.remove(self.project,['a.jpg','b.jpg'])
        for name in ['a.jpg','b.jpg']:self.assertTrue((Path(self.project['images'])/name).is_file())
        for name in ['a.json','b.json']:self.assertTrue((Path(self.project['annotations'])/name).is_file())
        self.assertEqual(image_trash.list_batches(self.project),[])

    def test_project_isolation_and_busy_import(self):
        batch=image_trash.remove(self.project,['a.jpg'])['batch']
        other={**self.project,'images':str(self.root/'other/images'),'annotations':str(self.root/'other/annotations')}
        with patch.object(server,'current_project',return_value=other):
            self.assertEqual(self.client.get('/api/images/trash').get_json()['batches'],[])
            self.assertEqual(self.client.post('/api/images/restore',json={'batch':batch}).status_code,400)
        with patch.object(server,'IMPORT_JOBS',{'active':{'project_id':'trash-test','status':'running'}}):
            self.assertEqual(self.client.post('/api/images/delete',json={'names':['b.jpg']}).status_code,409)
            self.assertEqual(self.client.post('/api/images/restore',json={'batch':batch}).status_code,409)

    def test_shared_annotation_retained_until_last_image_is_deleted(self):
        self.image('a.png')
        image_trash.remove(self.project,['a.jpg'])
        self.assertTrue((Path(self.project['annotations'])/'a.json').is_file())
        image_trash.remove(self.project,['a.png'])
        self.assertFalse((Path(self.project['annotations'])/'a.json').exists())

if __name__=='__main__':unittest.main()
