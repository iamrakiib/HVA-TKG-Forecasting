#!/usr/bin/env python3
"""Fail-closed test of the paper's object-only ORIGINAL-quadruple evaluation.

Do not use a 10,000-query diagnostic sample as the canonical test split unless
it is the same set of original quadruples; this validator checks identity.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from prepare_ordinal_snapshots import load_rows

def validate(split_file:Path,score_dump:Path,stat_file:Path):
    vals=stat_file.read_text(encoding='utf8').split(); num_entities=int(vals[0])
    expected=load_rows(split_file)
    with np.load(score_dump,allow_pickle=False) as z:
        if 'scores' not in z: raise ValueError(f'{score_dump}: missing scores array')
        key=next((k for k in ('triples','queries','quads','facts','trips') if k in z),None)
        if not key:raise ValueError('dump missing query quadruples')
        scores=np.asarray(z['scores']); got=np.asarray(z[key],dtype=np.int64)
        if 'entity' in z:
            branch=str(np.asarray(z['entity']).reshape(-1)[0]).lower()
            if branch not in ('object','tail','o'):raise ValueError('Object-only protocol: subject branch is not permitted')
    if scores.shape!=(len(got),num_entities):raise ValueError(f'wrong score shape {scores.shape}, expected ({len(got)},{num_entities})')
    if got.shape!=(len(expected),4):raise ValueError(f'query shape {got.shape}, expected {expected.shape}')
    if Counter(map(tuple,got))!=Counter(map(tuple,expected)):
        raise ValueError('Score dump does not contain exactly the original split quadruples')
    if np.any((got[:,2]<0)|(got[:,2]>=num_entities)):raise ValueError('gold ID outside entity vocabulary')
    if not np.isfinite(scores).all():raise ValueError('score array has NaN/inf values')
    print(f'PASS: {split_file.name} object-only score dump matches original split (N={len(got)})')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--valid-dump',type=Path,required=True);p.add_argument('--test-dump',type=Path,required=True)
    a=p.parse_args(); d=a.data_root/a.dataset
    validate(d/'valid.txt',a.valid_dump,d/'stat.txt')
    validate(d/'test.txt',a.test_dump,d/'stat.txt')
if __name__=='__main__':main()
