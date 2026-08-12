# Screening vLLM 0.27 for the drain/concurrency deadlock

## RETRACTED 2026-08-12: the first four verdicts were instrument error, not results

The cells `eager`, `cg_piecewise`, `cg_none` and `breakable_cg` were published here as WEDGED.
They were not. Every one of them was declared dead at **exactly** the harness's 180 s
per-request timeout:

```
eager         trigger 09:42:49 -> verdict 09:45:49
cg_piecewise  trigger 10:00:26 -> verdict 10:03:26
cg_none       trigger 10:17:33 -> verdict 10:20:33
breakable_cg  trigger 10:28:01 -> verdict 10:31:01
```

Four engines do not die identically to the second. That is a timeout firing. The trigger
treated "request did not return in 180 s" as proof of a wedge, and a C=6 storm of 1200-token
prompts legitimately exceeds that on slower configurations, `--enforce-eager` most of all
since it removes CUDA graphs and runs several times slower by design. The harness measured my
own impatience and called it a deadlock.

**The conclusion drawn from those cells, that CUDA graphs are eliminated as the cause, is
withdrawn.** It may still be true. It is not supported by this data.

Raw invalid ledger kept as `screen027-INVALID-timeout-predicate.jsonl` rather than deleted.

## What the harness does now

A timeout is a signal, never a verdict. On any timeout the harness reads the server's own
counters and watches `generation_tokens_total` for 180 s:

- counter still climbing -> the engine is ALIVE and merely slow; the cell continues, and the
  slowness is recorded as a note on the result
- counter frozen, or metrics unreachable -> wedged, with the reason recorded

Per-request timeout raised to 600 s so that slow configurations are measured rather than
guillotined. Re-running the full matrix against the corrected harness; results below as they
land.

## Verdicts

(campaign restarting; table regenerates as cells complete)
