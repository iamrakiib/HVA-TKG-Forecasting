#!/usr/bin/env python3
"""Create *new* candidate-budget and diagnostics plots from run-generated CSVs.
No paper data points, labels or numeric result arrays are supplied here.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['budget','groups'],required=True)
    p.add_argument('--input-csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();df=pd.read_csv(a.input_csv)
    fig,ax=plt.subplots(figsize=(6,4))
    if a.kind=='budget':
        for col in ['gold_recall_at_K']:
            if col not in df:raise ValueError(f'Missing {col} in measured sweep')
        ax.plot(df['K'],df['gold_recall_at_K'],marker='o')
        ax.set_xlabel('Candidate budget K');ax.set_ylabel('Gold candidate recall at K')
    else:
        for col in ['group','base_Hits@1','adjusted_Hits@1']:
            if col not in df:raise ValueError(f'Missing {col} in measured diagnostics')
        x=range(len(df));w=0.35
        ax.bar([v-w/2 for v in x],df['base_Hits@1'],width=w,label='Backbone')
        ax.bar([v+w/2 for v in x],df['adjusted_Hits@1'],width=w,label='HVA')
        ax.set_xticks(list(x),list(df['group']));ax.legend();ax.set_ylabel('Hits@1')
    fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=250);plt.close(fig)
if __name__=='__main__':main()
