#!/usr/bin/env python3
"""Execute the manuscript's HVA experiment matrix on ORIGINAL backbone score dumps.

Plan format: a JSON object with 'jobs', each job holding backbone, dataset,
seed, data_root, valid_dump and test_dump (absolute paths recommended).
No paper metrics or hand-transcribed result values appear in this repository.

A job trains HVA using validation score dumps only; it may also run the
comparison controls, dual-branch, and five-backbone feature ablations.
Every subprocess saves new computed outputs outside the repository.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BB=('tirgn','regcn','cygnet','cenet','temp')
DS=('ICEWS14','ICEWS18','YAGO')
SEEDS=(0,99,12345,2026,3407)

def jobs_from_file(path):
    cfg=json.loads(Path(path).read_text(encoding='utf8'))
    jobs=cfg.get('jobs')
    if not isinstance(jobs,list) or not jobs:raise ValueError('Plan must contain nonempty jobs list')
    for j in jobs:
        for key in ('backbone','dataset','seed','data_root','valid_dump','test_dump'):
            if key not in j:raise ValueError(f'Missing {key} in job')
        if j['backbone'] not in BB or j['dataset'] not in DS or int(j['seed']) not in SEEDS:
            raise ValueError(f'Invalid backbone/dataset/seed in {j}')
        for key in ('data_root','valid_dump','test_dump'):
            if not Path(j[key]).exists():raise FileNotFoundError(f'{key}={j[key]}')
    return jobs

def assert_ordinal_timestamps(job):
    """Verify input IDs match the paper's snapshot-index time coordinate.

    This does not claim past score dumps used the same transformation; it
    prevents a NEW reproduction run from silently treating hour units as
    ordinal snapshot distances. Missing real-world snapshots must first be
    mapped by scripts/prepare_ordinal_snapshots.py.
    """
    from prepare_ordinal_snapshots import load_rows
    d = Path(job['data_root']) / job['dataset']
    arrays = [load_rows(d / f'{name}.txt') for name in ('train','valid','test')]
    unique = np.unique(np.concatenate([a[:, 3] for a in arrays]))
    if len(unique) > 1 and np.any(np.diff(unique) != 1):
        raise ValueError(
            f"{job['dataset']}: time IDs are not consecutive ordinal snapshots. "
            "Use scripts/prepare_ordinal_snapshots.py on COPIES of inputs, "
            "or pass --allow-unverified-time-units for explicitly "
            "documented archival behaviour. No timestamps were changed."
        )

def make_commands(job,output_root,dual,ablations,controls,budgets,exact_recency=False,rhvc_config=None,posthoc=False,profile=False):
    b=job['backbone']; d=job['dataset'];seed=int(job['seed'])
    dest=output_root/b/d/str(seed); m=ROOT/'backbone_integrations'/b
    c=[sys.executable,'-m','hva.run_hva_from_scores','--dataset',d,
       '--data-root',str(Path(job['data_root']).resolve()),
       '--valid-dump',str(Path(job['valid_dump']).resolve()),
       '--test-dump',str(Path(job['test_dump']).resolve()),
       '--seed',str(seed),'--mode','exact_only','--topk','100','--eval-topk','100',
       '--save-adjusted-scores','--out-dir',str(dest/'exact_only')]
    result=[(m,c)]
    if dual:
        new=c.copy();new[new.index('exact_only')]='dual_branch';new[-1]=str(dest/'dual_branch')
        result.append((m,new))
    if budgets:
        for k in (20,50,200,500):
            new=c.copy();new[new.index('--topk')+1]=str(k);new[new.index('--eval-topk')+1]=str(k)
            new[-1]=str(dest/f'exact_only_k{k}')
            result.append((m,new))
    if ablations:
        folder=ROOT/'analysis/revision_tools/03_HVA_ABLATION_CPU'
        cmd=[sys.executable,'run_hva_ablation_cpu.py',
             '--dataset',d,'--data-root',str(Path(job['data_root']).resolve()),
             '--valid-dump',str(Path(job['valid_dump']).resolve()),
             '--test-dump',str(Path(job['test_dump']).resolve()),
             '--seed',str(seed),'--topk','100','--out-dir',str(dest/'feature_ablation'),
             '--variants','full,no_occurrence,no_recency,no_frequency,no_stale',
             '--save-adjusted']
        result.append((folder,cmd))
    if controls:
        folder=ROOT/'analysis/revision_tools/02_STRONG_CONTROLS_CPU'
        for script in ('rfs_tune_apply.py','linear_same_features.py'):
            cmd=[sys.executable,script,'--dataset',d,
                 '--data-root',str(Path(job['data_root']).resolve()),
                 '--valid-dump',str(Path(job['valid_dump']).resolve()),
                 '--test-dump',str(Path(job['test_dump']).resolve()),
                 '--topk','100','--out-dir',str(dest/script[:-3])]
            # Same-feature Linear has a stochastic training loop. RFS is a
            # deterministic validation-grid search and has no --seed option.
            if script == 'linear_same_features.py':
                cmd.extend(['--seed', str(seed)])
            result.append((folder,cmd))
    if exact_recency:
        # Manuscript's deterministic additive Exact Recency reference: alpha=.10, lambda=.10.
        cmd=[sys.executable,'-m','hva.run_exact_recency','--dataset',d,
             '--data-root',str(Path(job['data_root']).resolve()),
             '--dump',str(Path(job['test_dump']).resolve()),'--split','test',
             '--alpha','0.10','--lambda-decay','0.10','--out-dir',str(dest/'exact_recency')]
        result.append((m,cmd))
    if rhvc_config:
        # This is the paper-described broad RHVC history, NOT the archived exact-only runner.
        cmd=[sys.executable,str(ROOT/'scripts/run_rhvc_broad_history.py'),
             '--backbone',b,'--dataset',d,
             '--data-root',str(Path(job['data_root']).resolve()),
             '--valid-dump',str(Path(job['valid_dump']).resolve()),
             '--test-dump',str(Path(job['test_dump']).resolve()),
             '--params-json',str(Path(rhvc_config).resolve()),
             '--out-dir',str(dest/'rhvc_broad'), '--save-adjusted-scores']
        result.append((ROOT,cmd))
    if posthoc:
        cmd=[sys.executable,str(ROOT/'scripts/run_posthoc_audits.py'),
             '--dataset',d,'--data-root',str(Path(job['data_root']).resolve()),
             '--base-dump',str(Path(job['test_dump']).resolve()),
             '--adjusted-dump',str(dest/'exact_only'/'test_adjusted_scores.npz'),
             '--out-root',str(dest/'diagnostics'),'--execute']
        result.append((ROOT,cmd))
        cmd=[sys.executable,str(ROOT/'scripts/coverage_identity_audit.py'),
             '--dataset',d,'--data-root',str(Path(job['data_root']).resolve()),
             '--base-dump',str(Path(job['test_dump']).resolve()),
             '--adjusted-dump',str(dest/'exact_only'/'test_adjusted_scores.npz'), '--k','100']
        result.append((ROOT,cmd))
    if profile:
        cmd=[sys.executable,str(ROOT/'scripts/profile_hva_runtime.py'),
             '--backbone',b,'--dataset',d,
             '--data-root',str(Path(job['data_root']).resolve()),
             '--test-dump',str(Path(job['test_dump']).resolve()),
             '--adapter-checkpoint',str(dest/'exact_only'/'hva_adapter.pt'),
             '--k','100','--batch-size','128',
             '--out-json',str(dest/'runtime'/'k100.json')]
        result.append((ROOT,cmd))
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--include-dual',action='store_true')
    p.add_argument('--include-feature-ablations',action='store_true')
    p.add_argument('--include-controls',action='store_true')
    p.add_argument('--include-budget-sensitivity',action='store_true')
    p.add_argument('--include-exact-recency',action='store_true')
    p.add_argument('--rhvc-params-json',type=Path,
                   help='Actual RHVC broader-history coefficients from original experiment; no invented defaults')
    p.add_argument('--include-posthoc',action='store_true',
                   help='Run actual-query coverage, diagnostic and case-extraction scripts after HVA')
    p.add_argument('--include-runtime-profile',action='store_true')
    p.add_argument('--include-hri',action='store_true',
                   help='Run one deterministic HRI-inspired reference per dataset; not five-seed HRI reconstruction')
    p.add_argument('--hri-tie-policy',choices=('legacy_strict','expected_uniform'),default='legacy_strict')
    p.add_argument('--require-complete-matrix',action='store_true')
    p.add_argument('--allow-unverified-time-units', action='store_true',
                   help='Opt out of manuscript snapshot-index preflight for archival raw-timestamp runs')
    p.add_argument('--execute',action='store_true',help='Otherwise only prints all commands')
    a=p.parse_args(argv)
    if ROOT in a.output_root.resolve().parents or a.output_root.resolve()==ROOT:
        raise ValueError('Output directory must be outside the source repository')
    jobs=jobs_from_file(a.plan)
    if a.rhvc_params_json is not None and not a.rhvc_params_json.is_file():
        raise FileNotFoundError(a.rhvc_params_json)
    if len({(j['backbone'],j['dataset'],int(j['seed'])) for j in jobs})!=len(jobs):
        raise ValueError('Duplicate backbone-dataset-seed jobs')
    if a.require_complete_matrix:
        wanted={(b,d,s) for b in BB for d in DS for s in SEEDS}
        got={(j['backbone'],j['dataset'],int(j['seed'])) for j in jobs}
        if got!=wanted:raise ValueError(f'Incomplete five-backbone × three-dataset × five-seed matrix: missing {len(wanted-got)}')
    for job in jobs:
        if not a.allow_unverified_time_units:
            assert_ordinal_timestamps(job)
        check=[sys.executable,str(ROOT/'scripts/validate_object_score_dumps.py'),
               '--dataset',job['dataset'],'--data-root',str(Path(job['data_root']).resolve()),
               '--valid-dump',str(Path(job['valid_dump']).resolve()),
               '--test-dump',str(Path(job['test_dump']).resolve())]
        commands=[(ROOT,check)]+make_commands(job,a.output_root.resolve(),a.include_dual,a.include_feature_ablations,a.include_controls,a.include_budget_sensitivity,a.include_exact_recency,a.rhvc_params_json,a.include_posthoc,a.include_runtime_profile)
        for cwd,cmd in commands:
            print('\n$', ' '.join(str(x) for x in cmd),'\n  cwd=',cwd,flush=True)
            if a.execute:subprocess.run(cmd,cwd=cwd,check=True)
    if a.include_hri:
        # The standalone recurrence model does not take a random seed: do not
        # report deterministic clones as a five-seed experimental distribution.
        for ds in sorted({j['dataset'] for j in jobs}):
            job=next(j for j in jobs if j['dataset']==ds)
            cmd=[sys.executable,str(ROOT/'scripts/run_hri_inspired.py'),
                 '--dataset',ds,'--data-root',str(Path(job['data_root']).resolve()),
                 '--tie-policy',a.hri_tie_policy,
                 '--out-json',str(a.output_root.resolve()/'hri_reference'/f'{ds}.json')]
            if a.allow_unverified_time_units:cmd.append('--allow-unverified-time-units')
            print('\n$', ' '.join(map(str,cmd)), '\n  cwd=',ROOT,flush=True)
            if a.execute:subprocess.run(cmd,cwd=ROOT,check=True)
    print('Complete; no pre-entered results were used.')
if __name__=='__main__':main()
