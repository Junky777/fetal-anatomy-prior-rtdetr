import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from fetal_anatomy.rules import PRIOR_ARM_BY_KEY,apply_anatomical_prior
from fetal_anatomy.metrics import ImageRecord,detection_counts,clinical_structure_metrics,evaluate_records
from fetal_anatomy.split import create_splits
from fetal_anatomy.data import read_manifest,manifest_fingerprint
from fetal_anatomy.evaluate import evaluate_folds

class RulesAndMetrics(unittest.TestCase):
    def record(self):
        return ImageRecord(1,'synthetic','normal','3','synthetic','', (100,100),
            np.array([[10.,10.,40.,40.]]),np.array([2]),
            np.array([[10.,10.,40.,40.],[11.,11.,40.,40.],[50.,50.,60.,60.],[1.,80.,10.,90.]]),
            np.array([.9,.8,.7,.6]),np.array([2,2,3,2]))

    def test_rule_order_and_coordinates(self):
        r=self.record();keep,stats=apply_anatomical_prior(r.pred_boxes,r.pred_scores,r.pred_classes,'3',r.shape,PRIOR_ARM_BY_KEY['full_prior'])
        self.assertEqual(keep.tolist(),[0]);self.assertEqual(stats['removed_by_spatial'],1)
        self.assertEqual(stats['removed_by_plane'],1);self.assertEqual(stats['removed_by_top1'],1)
        np.testing.assert_array_equal(r.pred_boxes[keep],r.gt_boxes)

    def test_box_and_image_structure_are_different(self):
        r=self.record();_,counts=detection_counts([r],PRIOR_ARM_BY_KEY['baseline'],.25)
        self.assertEqual(counts['tp'],1);self.assertEqual(counts['fp'],3)
        clinical=clinical_structure_metrics([r],PRIOR_ARM_BY_KEY['baseline'],.25)
        heart=next(x for x in clinical if x['structure']=='Heart')
        self.assertEqual(heart['TP'],1);self.assertEqual(heart['FP'],0)
        metric,_=evaluate_records([r],PRIOR_ARM_BY_KEY['full_prior'],.25)
        self.assertEqual(metric['F1'],1.0);self.assertEqual(metric['mAP50'],1.0)

    def test_empty_predictions(self):
        r=self.record();r.pred_boxes=np.empty((0,4));r.pred_scores=np.array([]);r.pred_classes=np.array([],dtype=int)
        _,counts=detection_counts([r],PRIOR_ARM_BY_KEY['full_prior'],.25)
        self.assertEqual(counts['fn'],1);self.assertEqual(counts['fp'],0)

    def test_export_and_patient_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            evaluate_folds([self.record()],Path(td)/'ok')
            self.assertTrue((Path(td)/'ok/ablation_by_fold.csv').is_file())
            a=self.record();b=self.record();b.fold=2;b.image='another_image'
            with self.assertRaises(ValueError):evaluate_folds([a,b],Path(td)/'bad')

class Splitting(unittest.TestCase):
    def make_data(self,root):
        rows=[]
        for case in range(30):
            for plane in range(1,6):
                image=root/f'fake_{case}_{plane}.png';label=image.with_suffix('.txt')
                Image.new('L',(16,16),case).save(image);label.write_text('2 0.5 0.5 0.2 0.2\n')
                rows.append({'case_id':f'fake_{case:03}','diagnosis':'normal' if case<20 else 'tof','plane':str(plane),
                             'image':image.name,'label':label.name})
        manifest=root/'manifest.csv'
        with manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return manifest

    def test_split_once_per_patient_no_overlap_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=self.make_data(root);create_splits(manifest,root/'cv')
            test=[]
            for fold in range(1,6):
                sets={s:{r['case_id'] for r in read_manifest(root/f'cv/fold{fold}/{s}.csv')} for s in ['train','val','test']}
                self.assertFalse(sets['train']&sets['val']);self.assertFalse(sets['train']&sets['test']);self.assertFalse(sets['val']&sets['test'])
                test.extend(sets['test'])
            self.assertEqual(len(test),30);self.assertEqual(len(set(test)),30)
            rows=read_manifest(manifest);before=manifest_fingerprint(rows)
            Path(rows[0]['label']).write_text('2 0.5 0.5 0.3 0.3\n')
            self.assertNotEqual(before,manifest_fingerprint(rows))
            with self.assertRaises(FileExistsError):create_splits(manifest,root/'cv')

    def test_view_required(self):
        with tempfile.TemporaryDirectory() as td:
            m=self.make_data(Path(td));text=m.read_text().replace(',1,',',9,');m.write_text(text)
            with self.assertRaises(ValueError):read_manifest(m)

if __name__=='__main__':unittest.main()
