"""Validated, explicit patient and view metadata for private datasets."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .rules import CLASS_NAMES

FIELDS = ['case_id', 'diagnosis', 'plane', 'image', 'label']

def read_manifest(path):
    path=Path(path).resolve()
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Manifest requires: '+', '.join(FIELDS))
        rows=list(reader)
    if not rows:raise ValueError('Empty manifest')
    seen=set();groups={};patient_views=set()
    for row in rows:
        for key in FIELDS:row[key]=row[key].strip()
        if not row['case_id']:raise ValueError('Missing case_id')
        if row['diagnosis'] not in {'normal','tof'}:raise ValueError('diagnosis must be normal or tof')
        if row['plane'] not in {'1','2','3','4','5'}:raise ValueError('Explicit plane 1..5 required')
        if row['case_id'] in groups and groups[row['case_id']]!=row['diagnosis']:
            raise ValueError('Inconsistent diagnosis for one fetus')
        groups[row['case_id']]=row['diagnosis']
        view=(row['case_id'],row['plane'])
        if view in patient_views:raise ValueError('Study manifest expects one image per fetus/view')
        patient_views.add(view)
        for key in ['image','label']:
            p=Path(row[key]);p=p if p.is_absolute() else path.parent/p
            p=p.resolve()
            if not p.is_file():raise FileNotFoundError(p)
            row[key]=p.as_posix()
        if row['image'] in seen:raise ValueError('Duplicate image')
        seen.add(row['image'])
        for line in Path(row['label']).read_text('utf-8').splitlines():
            values=line.split()
            if len(values)!=5:raise ValueError('Expected YOLO class cx cy width height')
            a=np.array([float(v) for v in values])
            if not np.isfinite(a).all() or a[0]!=int(a[0]) or not 0<=a[0]<len(CLASS_NAMES):
                raise ValueError('Invalid class or non-finite label')
            if not ((a[1:]>=0)&(a[1:]<=1)).all() or a[3]<=0 or a[4]<=0:
                raise ValueError('Invalid normalized label coordinates')
    return rows

def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(rows[0]) if rows else []
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def require_new_directory(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise FileExistsError(f'Use a new output directory: {path}')
    path.mkdir(parents=True,exist_ok=True)
    return path

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def manifest_fingerprint(rows):
    # Hash content, not just filenames, so stale caches cannot survive data edits.
    data=[{**{k:r[k] for k in FIELDS}, 'image_sha256':file_hash(r['image']),
           'label_sha256':file_hash(r['label'])} for r in rows]
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
