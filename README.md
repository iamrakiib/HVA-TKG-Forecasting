# History Validity Adapter (HVA)

Source code supplement for *Learning Candidate-Level History Validity Correction for Temporal Knowledge Graph Forecasting*.

HVA learns a ranking-stage correction over frozen temporal knowledge graph backbone scores. This release supplies adapter overlays for TiRGN, RE-GCN, CyGNet, CENET, and TeMP, together with comparison controls, ablations, diagnostics, and statistical analysis tools.

## Installation

Use an environment with Python, pip, and PyTorch support for your hardware:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

Tests use small synthetic inputs to check implementation behavior. They do not validate benchmark results. Dependencies are currently unpinned; the original experiment environment is not supplied.

## Required inputs

Provide `data/<DATASET>/{train,valid,test,stat}.txt`. Split files contain integer columns `subject relation object timestamp`. `stat.txt` contains entity and original relation counts.

For each frozen backbone checkpoint, provide validation and test `.npz` dumps containing `scores [N,E]` (full candidate scores), `triples [N,4]` (original split quadruples in score-row order), and optionally `entity=['object']` (object prediction is the default).

The matrix uses original object-prediction queries. Install each upstream backbone separately and follow its license and citation requirements. Backbone training and general score-export pipelines are outside this release.

```bash
python scripts/validate_object_score_dumps.py \
    --dataset ICEWS14 --data-root /path/to/data \
    --valid-dump /path/to/valid.npz --test-dump /path/to/test.npz
```

Historical distance is interpreted in ordinal snapshot steps. If stored times use other units, prepare copies of both splits and dumps:

```bash
python scripts/prepare_ordinal_snapshots.py \
    --dataset ICEWS14 --data-root /path/to/raw_data \
    --out-root /path/to/prepared_inputs \
    --dump valid=/path/to/valid.npz --dump test=/path/to/test.npz
```

Use `prepared_inputs/data` and `prepared_inputs/dumps/ICEWS14/{valid,test}.npz` afterward. This does not establish the timestamp encoding used in historical experiments. The launcher checks consecutive time IDs by default; `--allow-unverified-time-units` permits explicitly documented archived raw-time behavior.

## Run HVA and comparisons

```bash
python scripts/run_experiment_matrix.py \
    --plan /path/to/plan.json --output-root /path/to/run_artifacts \
    --include-dual --include-feature-ablations --include-controls \
    --include-budget-sensitivity --include-exact-recency
```

Add `--execute` to run. Outputs must be outside the source repository. `--require-complete-matrix` checks all five backbones, three datasets (`ICEWS14`, `ICEWS18`, `YAGO`), and five configured seeds (`0`, `99`, `12345`, `2026`, `3407`). This trains adapters from frozen scores. Confirm settings against original run records before comparing table values.

HVA defaults: K=100, relation embedding dimension 16, two hidden layers of width 64, dropout 0.10, AdamW learning rate 1e-3, batch size 64, at most 12 epochs, patience 3, pairwise margin 0.20, pairwise weight 0.25, residual penalty 1e-4, and staleness threshold 10. Adapter fitting uses a chronological split of the validation dump; test data is reserved for evaluation.

RHVC requires a complete coefficient configuration;

## Diagnostics and runtime

```bash
python scripts/run_posthoc_audits.py \
    --dataset ICEWS14 --data-root /path/to/data \
    --base-dump /path/to/base_test.npz \
    --adjusted-dump /path/to/test_adjusted_scores.npz \
    --out-root /path/to/audit_artifacts --execute
```

This runs input, top-K, history-group, age, rank-transition, and query-case analyses. `scripts/coverage_identity_audit.py` checks weighted coverage consistency using the same dataset, base and adjusted dumps, and `--k 100`.




## License

Author-owned contributions are released under the [MIT License](LICENSE). Third-party code and external backbone implementations retain their respective licenses. Upstream source attribution and redistribution permissions still need to be established where applicable.
