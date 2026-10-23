# Pack 02: Strong reviewer controls, CPU only

This pack does **not** train a TKG backbone. It consumes frozen VALID/TEST score dumps.

Controls:
- RFS: deterministic same-feature correction on base top-K, tuned on VALID only.
- Linear same-feature residual: four learned weights over `[occurrence, recency, log_frequency, stale]`, trained on VALID only with the same candidate scope as HVA.

Feature definitions match the core HVA code:
`recency = 1/(1+gap)`, `log_frequency = log1p(freq_before_t)`, `stale = 1[gap > threshold]`.
