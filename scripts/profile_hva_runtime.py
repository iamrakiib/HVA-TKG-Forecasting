#!/usr/bin/env python3
"""Measured HVA lookup/adapter/total latency, without backbone inference.

Requires an actually trained adapter checkpoint from run_hva_from_scores.py.
Writes a newly computed JSON file outside the code repository.
Profile device, warmup, repeats, batch size, cache and copy assumptions.
"""
from __future__ import annotations
import argparse,sys,json,time,platform,tracemalloc,pickle
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone',choices=['tirgn','regcn','cygnet','cenet','temp'],required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--test-dump',type=Path,required=True)
    p.add_argument('--adapter-checkpoint',type=Path,required=True)
    p.add_argument('--mode',choices=['exact_only','dual_branch'],default='exact_only')
    p.add_argument('--k',type=int,default=100);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--stale-threshold',type=int,default=10)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--out-json',type=Path,required=True)
    args=p.parse_args()
    if ROOT in args.out_json.resolve().parents:raise ValueError('Write measurements outside source repo')
    sys.path.insert(0,str(ROOT/'backbone_integrations'/args.backbone))
    from hva.history_utils import (read_stat,read_split_arrays,load_score_dump,augment_with_inverse,
                                   canonicalize_queries,build_histories,score_only_topk,scatter_topk_back)
    from hva.hva_model import HistoryValidityAdapter,build_hva_features
    dev=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    num_entities,num_rel=read_stat(args.data_root/args.dataset)
    splits=read_split_arrays(args.data_root/args.dataset)
    scores,triples,entity=load_score_dump(args.test_dump)
    if entity!='object':raise ValueError('Manuscript reports original-object queries only')
    queries=canonicalize_queries(triples,entity,num_rel)
    if scores.shape[1]!=num_entities:raise ValueError('Unexpected vocabulary length')
    hist=build_histories(augment_with_inverse(np.concatenate([splits['train'],splits['valid']]),num_rel))
    model=HistoryValidityAdapter(max(2*num_rel,int(np.max(queries[:,1])+1)),mode=args.mode,stale_threshold=args.stale_threshold).to(dev)
    # Never silently fall back to an untrained adapter.
    checkpoint=torch.load(args.adapter_checkpoint,map_location=dev,weights_only=True)
    model.load_state_dict(checkpoint,strict=True);model.eval()
    times={'lookup_feature_ms':[],'adapter_score_ms':[],'complete_hva_ms':[]}
    batch_count=0
    if dev.type=='cuda':torch.cuda.reset_peak_memory_stats(dev)
    tracemalloc.start()
    with torch.inference_mode():
        for start in range(0,len(queries),args.batch_size):
            stop=min(start+args.batch_size,len(queries)); n=stop-start
            x=torch.tensor(scores[start:stop],dtype=torch.float32,device=dev)
            q=np.array(queries[start:stop],copy=True);q[:,2]=-1
            rel=torch.as_tensor(queries[start:stop,1],dtype=torch.long,device=dev)
            ids=score_only_topk(x,args.k);topscores=x.gather(1,ids)
            def once(record:bool):
                synchronize(dev);a=time.perf_counter()
                feat=build_hva_features(q,ids,hist,dev,args.mode,args.stale_threshold)
                synchronize(dev);b=time.perf_counter()
                adjusted,delta=model(topscores,rel,feat)
                synchronize(dev);c=time.perf_counter()
                merged=scatter_topk_back(x,ids,adjusted)
                synchronize(dev);d=time.perf_counter()
                # Fully-routed timing includes top-K and history feature construction.
                synchronize(dev);t=time.perf_counter()
                ids2=score_only_topk(x,args.k)
                feat2=build_hva_features(q,ids2,hist,dev,args.mode,args.stale_threshold)
                score2=x.gather(1,ids2)
                a2,_=model(score2,rel,feat2)
                merged2=scatter_topk_back(x,ids2,a2)
                synchronize(dev);t2=time.perf_counter()
                if record:
                    times['lookup_feature_ms'].append((b-a)*1000/n)
                    times['adapter_score_ms'].append((c-b)*1000/n)
                    times['complete_hva_ms'].append((t2-t)*1000/n)
            for _ in range(args.warmup):once(False)
            for _ in range(args.repeats):once(True)
            batch_count+=1
    _,peak_py=tracemalloc.get_traced_memory();tracemalloc.stop()
    report={
        'scope':'additional HVA only; excludes backbone inference',
        'dataset':args.dataset,'backbone':args.backbone,'mode':args.mode,'topk':args.k,
        'n_queries':int(len(queries)),'timed_batches':batch_count,'batch_size':args.batch_size,
        'warmup':args.warmup,'repetitions_per_batch':args.repeats,
        'device':str(dev),'device_name':torch.cuda.get_device_name(dev) if dev.type=='cuda' else platform.processor(),
        'python':platform.python_version(),'torch':torch.__version__,
        'latency_ms_per_query':{name:{'median':float(np.median(vals)),'mean':float(np.mean(vals))} for name,vals in times.items()},
        'python_tracemalloc_peak_mib':peak_py/(1024**2),
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(dev)/(1024**2) if dev.type=='cuda' else None,
        'history_cache_serialized_mib_estimate':len(pickle.dumps(hist,protocol=4))/(1024**2),
        'memory_note':'Python tracked memory excludes many native allocations; serialized cache size is approximate, not exact process RSS.',
        'timing_note':'Per-query latencies are derived from measured batch durations; total includes top-K selection and score scatter, and need not equal lookup+adapter.'}
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
    print('Wrote measurement report:',args.out_json)
if __name__=='__main__':main()
