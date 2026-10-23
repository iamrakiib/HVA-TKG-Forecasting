#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from revision_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--dump',required=True); ap.add_argument('--split',choices=['valid','test'],required=True); ap.add_argument('--out',default='results/input_audit.json'); a=ap.parse_args()
ne,nr,sp,h,filt=context(a.data_root,a.dataset,a.split); scores,tr,ent=load_score_dump(a.dump); q=canonicalize_queries(tr,ent,nr)
res={'dataset':a.dataset,'split':a.split,'entity_branch':ent,'score_shape':list(scores.shape),'query_shape':list(q.shape),'num_entities_stat':ne,'num_relations_stat':nr,'score_entity_dim_matches_stat':bool(scores.shape[1]==ne),'dataset_split_rows':int(len(sp[a.split])),'dump_rows':int(len(q)),'row_count_matches_split':bool(len(q)==len(sp[a.split])),'finite_fraction':float(np.isfinite(scores).mean()),'gold_id_in_range':bool(np.all((q[:,2]>=0)&(q[:,2]<scores.shape[1])))}
save_json(res,a.out); print(json.dumps(res,indent=2))
