"""Read author-published R7 trajectories without executing author programs.

This reader intentionally handles the documented first worksheet layout of
Filo2, not arbitrary Excel workbooks. Raw source files remain immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.generative.native_learning_stream import atomic_json

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def worksheet(path):
    with ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(n.text or '' for n in item.findall('.//s:t', NS))
                       for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows = []
        for item in sheet.findall('.//s:row', NS):
            row = {}
            for cell in item.findall('s:c', NS):
                name = re.match('[A-Z]+', cell.attrib['r']).group()
                value = cell.find('s:v', NS)
                if cell.attrib.get('t') == 'inlineStr':
                    row[name] = ''.join(n.text or '' for n in cell.findall('.//s:t', NS))
                elif value is not None:
                    if cell.attrib.get('t') == 's':
                        row[name] = strings[int(value.text)]
                    else:
                        row[name] = float(value.text)
            if row:
                rows.append((int(item.attrib['r']), row))
    return rows


def trajectory_summary(records):
    counts = np.zeros(60, dtype=int)
    for item in records:
        counts[item['first_frame']:item['last_frame']+1] += 1
    mean = float(counts.mean())
    incident = [r for r in records if r['first_frame'] > 0]
    return dict(record_count=len(records), initial_preexisting=sum(r['first_frame']==0 for r in records),
        incident_records=len(incident), incident_observed_endings=sum(r['last_frame']<59 for r in incident),
        incident_right_censored=sum(r['last_frame']==59 for r in incident),
        counts=counts.tolist(), mean_count=mean, population_variance=float(counts.var()),
        fano=None if mean==0 else float(counts.var()/mean), zero_frame_fraction=float(np.mean(counts==0)),
        count_histogram={str(k):v for k,v in sorted(Counter(counts.tolist()).items())})


def parse_workbook(path, source_root):
    rows = worksheet(path)
    header = rows[0][1]
    bulb = path.parent.name.startswith('Bulbous_')
    if bulb:
        expected={'A':'Bulbous tip ID','B':'Life time (min)','C':'Between frames (x-y)'}
        if any(header.get(k)!=v for k,v in expected.items()):
            raise ValueError('Unexpected author bulb worksheet header: '+str(path))
        keys = dict(identifier='A', first='C', last='D', duration='B')
    else:
        if header.get('E')=='Start Time Step' and header.get('F')=='End Time Step':
            keys = dict(identifier='B', first='E', last='F', duration='G')
        elif header.get('B')=='Start Time Step' and header.get('C')=='End Time Step' and header.get('D')=='lifetime':
            keys = dict(identifier='A', first='B', last='C', duration='D')
        else:
            raise ValueError('Unexpected author filopodium worksheet header: '+str(path))
    records=[];mismatches=[];invalid=[];seen=set()
    for index,row in rows[1:]:
        if keys['identifier'] not in row:
            continue
        uid = str(row[keys['identifier']])
        values=[row[keys[k]] for k in ('first','last','duration')]
        if not all(np.isfinite(v) and v==int(v) for v in values):
            invalid.append(dict(excel_row=index,reason='noninteger or nonfinite coordinates',raw=values))
            continue
        first,last,duration=map(int,values)
        if uid in seen or not 0<=first<=last<=59:
            invalid.append(dict(excel_row=index,reason='duplicate ID or invalid frame bounds',
                id=uid,first=first,last=last,reported=duration))
            continue
        seen.add(uid)
        record=dict(id=uid,excel_row=index,first_frame=first,last_frame=last,
            reported_duration_minutes=duration,frame_inclusive_duration_minutes=last-first+1,
            existing_at_window_start=first==0,right_censored=last==59)
        if not bulb and keys['first']=='E':
            record['bulbous_frame_count']=int(row.get('C',0))
        records.append(record)
        if duration!=last-first+1:
            mismatches.append(dict(excel_row=index,id=uid,reported=duration,frame_derived=last-first+1))
    return dict(file=str(path.relative_to(source_root)).replace('\\','/'),
        genotype=path.parent.name.removeprefix('Bulbous_'),kind='bulb' if bulb else 'filopodium',
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records,
        duration_conflicts=mismatches,invalid_rows=invalid,
        fit_admitted=not mismatches and not invalid,**trajectory_summary(records))


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'report.json').exists() or (args.out/'observations.json').exists():
        raise FileExistsError('Do not overwrite a completed observation audit')
    manifest=json.loads((args.source/'download_manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((args.source/item['path']).read_bytes()).hexdigest()==item['sha256']
    workbooks=[parse_workbook(p,args.source) for p in sorted((args.source/'Data/Files').glob('*/*.xlsx'))]
    groups={}
    for w in workbooks:
        group=groups.setdefault(w['kind']+'/'+w['genotype'],dict(workbooks=0,records=0,
            duration_conflicts=0,invalid_rows=0,incident_records=0,incident_observed_endings=0,incident_right_censored=0))
        for name in list(group):
            if name=='workbooks':group[name]+=1
            elif name in ('duration_conflicts','invalid_rows'):group[name]+=len(w[name])
            elif name=='records':group[name]+=len(w[name])
            else:group[name]+=w[name]
    atomic_json(args.out/'observations.json',dict(source=manifest,workbooks=workbooks,
        role='Author-processed observed tracks, not model-generated trajectories'))
    report=dict(source_repository=manifest['repository'],source_commit=manifest['commit'],
        groups=groups,source_hashes_verified=True,files=len(workbooks),
        all_duration_columns_agree=not any(w['duration_conflicts'] for w in workbooks),
        bulb_preregistered_fit_admitted=False,
        scope='Format, event/censoring and count audit only; no candidate dynamics or physical individual changed')
    atomic_json(args.out/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
