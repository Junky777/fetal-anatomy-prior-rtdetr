"""Exploratory fixed-split native-validator comparison, separate from ablation."""
import argparse
from pathlib import Path
from .data import require_new_directory,write_csv

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','rtdetr','yolov8','yolo11','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='0');a=p.parse_args()
    for name in ['rtdetr','yolov8','yolo11']:
        if not Path(getattr(a,name)).is_file():raise FileNotFoundError(getattr(a,name))
    out=require_new_directory(a.out).resolve()
    from ultralytics import RTDETR,YOLO
    rows=[]
    for name,family,batch in [('rtdetr',RTDETR,2),('yolov8',YOLO,4),('yolo11',YOLO,4)]:
        m=family(getattr(a,name));v=m.val(data=a.data,split='test',imgsz=960,batch=batch,device=a.device,plots=False,save=False,
                                       project=str(out),name=name)
        pr,re=float(v.box.mp),float(v.box.mr)
        rows.append({'model':name,'precision':pr,'recall':re,'F1':2*pr*re/(pr+re) if pr+re else 0,
                     'mAP50':float(v.box.map50),'mAP50_95':float(v.box.map)})
    write_csv(out/'native_comparison.csv',rows)
if __name__=='__main__':main()
