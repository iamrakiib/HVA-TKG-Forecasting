# Pack 03: HVA feature ablations on CPU

This pack preserves the user's core HVA definitions:
- exact features: `[seen, 1/(1+gap), log1p(freq), stale]`
- top-K selected from base scores only
- relation embedding + centered score + gap-from-top1 + rank context
- residual correction `base + tanh(gamma_raw) * delta`
- CE + pairwise margin + delta regularization training objective

New code only adds controlled switches that zero one input component at a time. The `full` variant is intended to reproduce the same core logic, but full GPU reruns were not performed here.

No backbone is trained. The small adapter is learned on VALID score dumps and can run on CPU.
