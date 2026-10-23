# RHVC configuration

The runner requires either `--params-json` for a fixed configuration or `--grid-json` for validation-MRR selection. No historical coefficients are supplied.

```bash
python scripts/rhvc_write_config_template.py --out /path/to/rhvc_parameters.json
```

Fill every `null` from the intended configuration. A blank template is not executable. The object must contain exactly these fields:

| Field | Meaning |
|---|---|
| `gamma` | Combined correction scale |
| `topk` | Positive candidate budget |
| `stale_threshold` | Nonnegative age threshold |
| `lambda_decay` | Exponential recency decay |
| `exact_occurrence`, `exact_frequency`, `exact_recency`, `exact_stale` | Exact-history weights |
| `subject_occurrence`, `subject_frequency`, `subject_recency`, `subject_stale` | Subject-candidate history weights |
| `relation_occurrence`, `relation_frequency`, `relation_recency`, `relation_stale` | Relation-candidate history weights |

Weights multiply occurrence, `log1p(count)`, `exp(-lambda_decay * age)`, and staleness. Staleness weights are subtracted. The three history contributions are summed and multiplied by `gamma` for baseline-selected top-K candidates only.

For a validation grid, supply the same fields with nonempty lists of values; the runner evaluates their Cartesian product. Record the grid and selected coefficients. Validation selection does not establish historical settings.

```bash
python scripts/run_rhvc_broad_history.py \
    --backbone tirgn --dataset ICEWS14 --data-root /path/to/data \
    --valid-dump /path/to/valid.npz --test-dump /path/to/test.npz \
    --params-json /path/to/rhvc_parameters.json \
    --out-dir /path/to/rhvc_outputs --save-adjusted-scores
```
