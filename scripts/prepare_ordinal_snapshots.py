#!/usr/bin/env python3
"""Convert four-column TKG split timestamps and corresponding score dumps to
ordinal snapshot indices. The original inputs are never overwritten.

The manuscript defines recency/staleness in snapshot steps. The archived HVA
code subtracts integer time IDs. Running this conversion makes that subtraction
mathematically equivalent to index(t)-index(last_t) on this dataset's timeline.

Only *timestamps* from the three chronological splits define the index map:
no query answers are included in a historical feature lookup by this script.
IMPORTANT: this normalization is a protocol implementation, not proof that
older reported experiments used the same timestamp encoding.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

TRIP_KEYS=('triples','queries','quads','facts','trips')

def load_rows(path:Path)->np.ndarray:
    rows=[]
    for line in path.read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        cols=line.split()
        if len(cols)!=4:raise ValueError(f'{path}: expected four columns: {line!r}')
        rows.append([int(v) for v in cols])
    arr=np.asarray(rows,dtype=np.int64).reshape(-1,4)
    if not len(arr):raise ValueError(f'{path}: empty split')
    return arr

def normalize_splits(data_dir:Path,out_data:Path)->dict:
    splits={name:load_rows(data_dir/f'{name}.txt') for name in ['train','valid','test']}
    timeline=np.unique(np.concatenate([a[:,3] for a in splits.values()]))
    timeline.sort()
    idx={int(t):int(j) for j,t in enumerate(timeline)}
    for name,a in splits.items():
        b=a.copy()
        b[:,3]=[idx[int(t)] for t in a[:,3]]
        np.savetxt(out_data/f'{name}.txt',b,fmt='%d')
    shutil_file=data_dir/'stat.txt'
    (out_data/'stat.txt').write_bytes(shutil_file.read_bytes())
    # Guarantee train/valid/test separation isn't violated by time indexing.
    if not (splits['train'][:,3].max()<splits['valid'][:,3].min()
        and splits['valid'][:,3].max()<splits['test'][:,3].min()):
        raise ValueError('Expected strictly chronological train/valid/test timestamps')
    return idx

def convert_dump(input_path:Path,output_path:Path,idx:dict):
    with np.load(input_path,allow_pickle=False) as z:
        payload={k:np.array(z[k],copy=True) for k in z.files}
    triples=[k for k in TRIP_KEYS if k in payload]
    if len(triples)!=1:
        raise ValueError(f'{input_path}: need exactly one query key from {TRIP_KEYS}, got {triples}')
    k=triples[0]; arr=payload[k]
    if arr.ndim!=2 or arr.shape[1]<4:raise ValueError('query matrix must be [N,>=4]')
    arr=arr.astype(np.int64,copy=True)
    if any(int(t) not in idx for t in arr[:,3]):raise ValueError('score dump uses timestamps missing from split files')
    arr[:,3]=[idx[int(t)] for t in arr[:,3]]
    payload[k]=arr
    output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,**payload)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--out-root',type=Path,required=True)
    ap.add_argument('--dump',action='append',default=[],metavar='NAME=PATH',help='May repeat, e.g. valid=/.../valid.npz')
    a=ap.parse_args(argv)
    indir=a.data_root/a.dataset; outdata=a.out_root/'data'/a.dataset
    if outdata.resolve()==indir.resolve():raise ValueError('Input and output directory must differ')
    outdata.mkdir(parents=True,exist_ok=True)
    idx=normalize_splits(indir,outdata)
    for entry in a.dump:
        if '=' not in entry:raise ValueError(f'Expected NAME=PATH, got {entry!r}')
        name,path=entry.split('=',1)
        if not name.replace('_','').replace('-','').isalnum():raise ValueError('Invalid dump name')
        dest=a.out_root/'dumps'/a.dataset/f'{name}.npz'
        convert_dump(Path(path),dest,idx)
        print('prepared dump:',dest)
    # Encode mapping metadata, not experiment results.
    (outdata/'snapshot_index_map.json').write_text(
        json.dumps({'raw_timestamp_to_index':{str(k):v for k,v in idx.items()},
                    'note':'Indices sorted over all split timestamps; test labels not used as history'},indent=2),encoding='utf8')
    print('prepared indexed splits:',outdata)
if __name__=='__main__':main()
