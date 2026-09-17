"""Create the study's patient-stratified outer CV and inner validation splits."""
import argparse,json
from pathlib import Path
from collections import Counter
import yaml
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from .data import read_manifest,write_csv,require_new_directory,FIELDS
from .rules import CLASS_NAMES

def create_splits(manifest,out,seed=2026):
    rows=read_manifest(manifest)
    case_group={r['case_id']:r['diagnosis'] for r in rows}
    cases=sorted(case_group);groups=[case_group[k] for k in cases]
    counts=Counter(groups)
    if set(counts)!={'normal','tof'} or min(counts.values())<5:
        raise ValueError('Five-fold stratification requires both groups and >=5 fetuses per group')
    view_count=Counter(r['case_id'] for r in rows)
    if any(n!=5 for n in view_count.values()):raise ValueError('This study requires five views per fetus')
    out=require_new_directory(out)
    assignments=[]
    for fold,(tv,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(cases,groups),1):
        tv_cases=[cases[i] for i in tv]
        tr,va=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed+fold).split(tv_cases,[case_group[k] for k in tv_cases]))
        ordered={'train':[tv_cases[i] for i in tr],'val':[tv_cases[i] for i in va],'test':[cases[i] for i in test]}
        sets={key:set(values) for key,values in ordered.items()}
        assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
        fold_dir=out/f'fold{fold}';fold_dir.mkdir()
        for split,ids in sets.items():
            selected=[r for case in ordered[split] for r in rows if r['case_id']==case]
            write_csv(fold_dir/f'{split}.csv',selected,FIELDS)
            (fold_dir/f'{split}.txt').write_text('\n'.join(r['image'] for r in selected)+'\n','utf-8')
            assignments.extend({'fold':fold,'case_id':c,'diagnosis':case_group[c],'split':split} for c in sorted(ids))
        config={s:(fold_dir/f'{s}.txt').resolve().as_posix() for s in sets}
        config.update(nc=len(CLASS_NAMES),names=CLASS_NAMES)
        (fold_dir/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False),'utf-8')
    write_csv(out/'patient_assignments.csv',assignments)
    (out/'split_settings.json').write_text(json.dumps({'seed':seed,'outer_folds':5,'inner_validation_fraction':.2,'case_order':'lexicographic'},indent=2),'utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args();create_splits(a.manifest,a.out,a.seed)
if __name__=='__main__':main()
