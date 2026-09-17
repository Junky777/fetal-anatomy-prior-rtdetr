"""Evaluate five prior arms with the original study's metric definitions."""
import argparse,json,math
from pathlib import Path
import numpy as np
from .data import read_manifest,manifest_fingerprint,write_csv,require_new_directory
from .rules import CLASS_NAMES,PRIOR_ARMS
from .metrics import ImageRecord,load_gt,evaluate_records,clinical_structure_metrics,confusion_counts,prediction_quality_counts,mean_sd_rows

def records_for_fold(manifest,cache,fold):
    rows=read_manifest(manifest);cache=Path(cache)
    meta=json.loads((cache/'metadata.json').read_text('utf-8'))
    if meta['fold']!=fold or meta['class_names']!=CLASS_NAMES or meta['conf_low']>.001:
        raise ValueError('Cache fold, class order or low-confidence limit mismatch')
    if meta['manifest_sha256']!=manifest_fingerprint(rows):raise ValueError('Cache does not match current image/label/metadata content')
    preds={}
    for line in (cache/'predictions.jsonl').open():
        r=json.loads(line)
        if r['image'] in preds:raise ValueError('Duplicate cached image')
        preds[r['image']]=r
    if set(preds)!={r['image'] for r in rows}:raise ValueError('Missing or extra cached predictions')
    records=[]
    for r in rows:
        c=preds[r['image']];shape=tuple(c['shape'])
        b=np.asarray(c['boxes'],dtype=float).reshape(-1,4);s=np.asarray(c['scores'],dtype=float);cl=np.asarray(c['classes'])
        if len(b)!=len(s) or len(s)!=len(cl) or not np.isfinite(b).all() or not np.isfinite(s).all():raise ValueError('Invalid cache arrays')
        if ((cl<0)|(cl>=16)|(cl!=cl.astype(int))).any() or ((s<0)|(s>1)).any():raise ValueError('Invalid cached classes/scores')
        gt,gc=load_gt(Path(r['label']),shape)
        records.append(ImageRecord(fold,r['case_id'],r['diagnosis'],r['plane'],r['image'],r['label'],shape,gt,gc,b,s,cl.astype(int)))
    return records

def evaluate_folds(records,out):
    folds=sorted({r.fold for r in records})
    if not records:raise ValueError('No records')
    patient_fold={};images=set()
    for r in records:
        if r.image in images:raise ValueError('Repeated test image')
        images.add(r.image)
        if r.case_id in patient_fold and patient_fold[r.case_id]!=r.fold:raise ValueError('Patient appears in multiple test folds')
        patient_fold[r.case_id]=r.fold
    out=require_new_directory(out);all_rows=[];sub=[];structure=[];matrices=[];clinical=[]
    for fold in folds:
        subset=[r for r in records if r.fold==fold]
        baseline_fp=None
        for arm in PRIOR_ARMS:
            metric,per_class=evaluate_records(subset,arm,.25)
            q=prediction_quality_counts(subset,arm,.25)
            if arm.key=='baseline':baseline_fp=metric['FP']
            metric.update(fold=fold,arm=arm.key,images=len(subset),logic_violations=q['logic_violations'],
                          FP_per100=100*metric['FP']/len(subset),FN_per100=100*metric['FN']/len(subset),
                          FP_reduction_pct=100*(1-metric['FP']/baseline_fp) if baseline_fp else float('nan'))
            all_rows.append(metric)
            structure.extend(dict(fold=fold,arm=arm.key,**row) for row in per_class)
            if arm.key in ['baseline','full_prior']:
                for group in ['normal','tof']:
                    rr=[r for r in subset if r.diagnosis==group]
                    if not rr:continue
                    result,_=evaluate_records(rr,arm,.25);sub.append(dict(fold=fold,arm=arm.key,diagnosis=group,**result))
    for arm in PRIOR_ARMS:
        if arm.key not in ['baseline','full_prior']:continue
        clinical.extend(clinical_structure_metrics(records,arm,.25))
        counts=confusion_counts(records,arm,.25)
        matrices.extend({'arm':arm.key,'reference':k[0],'prediction':k[1],'count':n} for k,n in sorted(counts.items()))
    fields=['precision','recall','F1','mAP50','mAP50_95','FP','FN','FP_per100','FN_per100','FP_reduction_pct','logic_violations']
    write_csv(out/'ablation_by_fold.csv',all_rows)
    write_csv(out/'ablation_mean_sd.csv',mean_sd_rows(all_rows,['arm'],fields))
    write_csv(out/'subgroup_by_fold.csv',sub)
    write_csv(out/'subgroup_mean_sd.csv',mean_sd_rows(sub,['arm','diagnosis'],fields[:5]))
    write_csv(out/'box_metrics_per_structure.csv',structure)
    write_csv(out/'image_structure_metrics.csv',clinical)
    write_csv(out/'confusion_counts.csv',matrices)
    (out/'evaluation_settings.json').write_text(json.dumps({'folds':folds,'confidence':.25,'iou':.5,'sd_ddof':1,
        'confidence_intervals':False,'hypothesis_tests':False,'note':'Image-structure metrics are pooled; box metrics are averaged over folds.'},indent=2),'utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--splits',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--folds',default='1,2,3,4,5');a=p.parse_args();records=[]
    folds=[int(x) for x in a.folds.split(',')]
    if len(set(folds))!=len(folds) or not set(folds).issubset(range(1,6)):raise ValueError('Unique folds 1..5 required')
    for f in folds:records+=records_for_fold(Path(a.splits)/f'fold{f}/test.csv',Path(a.cache)/f'fold{f}',f)
    evaluate_folds(records,a.out)
if __name__=='__main__':main()
