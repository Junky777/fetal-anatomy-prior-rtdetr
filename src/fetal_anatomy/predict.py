"""Cache unfiltered held-out detections once, then reuse for every prior arm."""
import argparse,json
from pathlib import Path
from importlib.metadata import version
from .data import read_manifest,require_new_directory,manifest_fingerprint,file_hash
from .rules import CLASS_NAMES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fold',type=int,required=True,choices=range(1,6))
    p.add_argument('--device',default='0');p.add_argument('--imgsz',type=int,default=960)
    a=p.parse_args();rows=read_manifest(a.manifest)
    if not Path(a.weights).is_file():raise FileNotFoundError(a.weights)
    out=require_new_directory(a.out)
    from ultralytics import RTDETR
    model=RTDETR(a.weights)
    names=model.names
    if [names[i] for i in range(len(names))]!=CLASS_NAMES:raise ValueError('Checkpoint class order mismatch')
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            result=model.predict(row['image'],imgsz=a.imgsz,conf=.001,device=a.device,batch=1,save=False,verbose=False)[0]
            boxes=result.boxes
            d={'image':row['image'],'shape':list(result.orig_shape),'boxes':boxes.xyxy.cpu().tolist(),
               'scores':boxes.conf.cpu().tolist(),'classes':boxes.cls.cpu().int().tolist()}
            f.write(json.dumps(d)+'\n')
    meta={'fold':a.fold,'manifest_sha256':manifest_fingerprint(rows),'weights_sha256':file_hash(a.weights),
          'imgsz':a.imgsz,'conf_low':.001,'class_names':CLASS_NAMES,
          'software':{k:version(k) for k in ['ultralytics','torch','numpy']}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),'utf-8')
if __name__=='__main__':main()
