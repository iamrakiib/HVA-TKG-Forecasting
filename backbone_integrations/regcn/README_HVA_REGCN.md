# RE-GCN score adapter overlay

This directory supplies ranking-stage adapters for frozen RE-GCN scores. It contains HVA exact-only, HVA dual-branch, Exact Recency, and a wrapper for the current broad-history RHVC runner. Upstream backbone training source, checkpoints, and general score-export pipelines are not included.

From this directory, run HVA with original object-prediction validation and test dumps:

```bash
python -m hva.run_hva_from_scores \
    --dataset ICEWS14 --data-root /path/to/data \
    --valid-dump /path/to/valid.npz --test-dump /path/to/test.npz \
    --seed 0 --mode exact_only --topk 100 --eval-topk 100 \
    --save-adjusted-scores --out-dir /path/to/hva_outputs
```

Use `--mode dual_branch` for the broader-feature variant. Run the input validators and timestamp preparation described in the [main README](../../README.md) first. Write outputs outside the source repository. The root matrix launcher supplies the same protocol across backbones.

Top-K membership depends only on frozen backbone scores. Features use facts strictly before the query time. Validation uses training history; test uses training plus validation history. Scores outside top-K are preserved.

The RHVC wrapper forwards to a later broad-history reimplementation and requires explicit coefficients. See [RHVC configuration](../../docs/RHVC_CONFIG_SCHEMA.md) and [provenance limitations](../../docs/EXPERIMENT_COVERAGE.md). The supplied code does not independently establish reproduction of historical numerical tables.
