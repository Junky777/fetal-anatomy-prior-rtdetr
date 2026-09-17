"""Detect structures in one image with an explicitly supplied standard view."""
import argparse,json
from pathlib import Path
import numpy as np
from .rules import CLASS_NAMES,PRIOR_ARM_BY_KEY,apply_anatomical_prior

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--view',required=True,choices=['1','2','3','4','5'])
    p.add_argument('--arm',choices=list(PRIOR_ARM_BY_KEY),default='full_prior')
    p.add_argument('--output',required=True);p.add_argument('--device',default='0')
    p.add_argument('--confidence',type=float,default=.25)
    a=p.parse_args()
    if not 0<a.confidence<=1:raise ValueError('confidence must be in (0,1]')
    if not Path(a.weights).is_file() or not Path(a.image).is_file():raise FileNotFoundError('Local weights and image required')
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    from ultralytics import RTDETR
    model=RTDETR(a.weights)
    if [model.names[i] for i in range(len(model.names))]!=CLASS_NAMES:raise ValueError('Checkpoint class order mismatch')
    r=model.predict(a.image,imgsz=960,conf=min(.001,a.confidence),device=a.device,save=False,verbose=False)[0]
    b=r.boxes.xyxy.cpu().numpy();s=r.boxes.conf.cpu().numpy();c=r.boxes.cls.cpu().numpy().astype(int)
    kept,_=apply_anatomical_prior(b,s,c,a.view,r.orig_shape,PRIOR_ARM_BY_KEY[a.arm]);kept=kept[s[kept]>=a.confidence]
    data={'view':a.view,'arm':a.arm,'shape':list(r.orig_shape),'confidence':a.confidence,
          'detections':[{'class_id':int(c[i]),'structure':CLASS_NAMES[c[i]],'score':float(s[i]),'xyxy':b[i].tolist()} for i in kept],
          'task':'anatomical structure detection; not a TOF diagnosis'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,indent=2),'utf-8')
if __name__=='__main__':main()
