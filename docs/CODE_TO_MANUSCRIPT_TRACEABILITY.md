# Code-to-method traceability

This maps methods named by the supplied documentation to current entry points. The submitted manuscript was unavailable, so this is not an independent claim-by-claim manuscript verification.

| Method or analysis | Location |
|---|---|
| HVA exact-only and dual-branch | `backbone_integrations/<backbone>/hva/hva_model.py`, `run_hva_from_scores.py` |
| Causal history, score-only top-K, original-score scatter | Each overlay's `hva/history_utils.py` |
| CE, pairwise margin, residual regularization | Each overlay's `hva/run_hva_from_scores.py` |
| Filtered ranking and diagnostics | Each overlay's `hva/metrics.py` |
| Exact Recency | Each overlay's `hva/run_exact_recency.py` |
| RHVC broad-history replacement | `scripts/run_rhvc_broad_history.py`; overlay RHVC files are wrappers |
| HRI-inspired recurrence and tie sensitivity | `scripts/run_hri_inspired.py`, `scripts/hri_tie_audit.py` |
| Same-feature RFS and learned linear controls | `analysis/revision_tools/02_STRONG_CONTROLS_CPU/` |
| Feature ablations and exact/dual slices | `analysis/revision_tools/03_HVA_ABLATION_CPU/` |
| Top-K, history groups, age, transitions, query cases | `analysis/revision_tools/01_CPU_AUDIT_AND_MECHANISM/` |
| Paired seed tests and query bootstrap | `analysis/revision_tools/04_STATISTICS_AND_TABLES/` |
| Matrix, input validation, timestamp preparation | `scripts/run_experiment_matrix.py`, `validate_object_score_dumps.py`, `prepare_ordinal_snapshots.py` |
| Coverage, audits, runtime, figures | `scripts/coverage_identity_audit.py`, `run_posthoc_audits.py`, `profile_hva_runtime.py`, `plot_study_figures.py` |

See [provenance notes](EXPERIMENT_COVERAGE.md) for the distinction between functionality and historical reproducibility.
