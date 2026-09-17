"""Train one fold with the recorded study settings; no training on import."""
import argparse,json
from pathlib import Path
import yaml
from .rules import CLASS_NAMES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['rtdetr','yolo'],default='rtdetr')
    p.add_argument('--data',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--config',default='configs/rtdetr_l.yaml');p.add_argument('--output',required=True)
    p.add_argument('--device',default=None);p.add_argument('--batch',type=int)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text('utf-8'))
    dataset=yaml.safe_load(Path(a.data).read_text('utf-8'))
    names=dataset.get('names',[])
    if isinstance(names,dict):names=[names[k] for k in sorted(names)]
    if names!=CLASS_NAMES:raise ValueError('Dataset class order must match the 16 study classes')
    if a.batch:cfg['batch']=a.batch
    elif a.family=='yolo':cfg['batch']=4
    if a.device is not None:cfg['device']=a.device
    out=Path(a.output).resolve()
    cfg.update(data=str(Path(a.data).resolve()),project=str(out.parent),name=out.name,save=True,val=True,plots=True)
    if a.dry_run:print(json.dumps(cfg,indent=2));return
    if out.exists():raise FileExistsError('Use a fresh run directory; existing experiments are never overwritten')
    if not Path(a.weights).is_file():raise FileNotFoundError('Provide locally obtained pretrained weights')
    from ultralytics import RTDETR,YOLO
    model=(RTDETR if a.family=='rtdetr' else YOLO)(a.weights)
    model.train(**cfg)
if __name__=='__main__':main()
