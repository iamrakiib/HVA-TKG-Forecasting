# Analysis run order

These tools consume dataset integer-ID files and frozen backbone scores. They compute new outputs; original experiment outputs are not included.

## Inputs

For each backbone and seed, supply validation and test dumps with `scores [N,E]`, `triples [N,4]`, and optional `entity=['object']`. Supply `train.txt`, `valid.txt`, `test.txt`, and `stat.txt` under the dataset directory. Retain original seeds, query populations, and timestamp units in run metadata.

## Order

1. `01_CPU_AUDIT_AND_MECHANISM`: input audits, top-K coverage, diagnostic partitions, history age, transitions, and query cases.
2. `02_STRONG_CONTROLS_CPU`: same-feature deterministic RFS and learned linear controls.
3. `03_HVA_ABLATION_CPU`: adapter feature ablations and exact-versus-dual comparisons.
4. `04_STATISTICS_AND_TABLES`: paired seed statistics and query-level bootstrap/permutation.

See each directory's README and script `--help` for input arguments. Write computed outputs outside the source repository. Statistics require actual per-seed or per-query measurements, with paired populations matched across methods.
