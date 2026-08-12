# Screening vLLM 0.27 for the drain/concurrency deadlock

Live ledger. This file is regenerated as each cell finishes, including the cells that
disprove my own hypotheses. Raw JSONL: `screen027.jsonl`. Harness: `wedge_trigger.py`
(C=6 storm, then a 27K-token cold prefill, then drain, then probe, 3 cycles) and
`screen_027.sh` (boots each cell unattended, one variable changed from the control).

**Control:** v0.27.0 + DeepGEMM `2fd67329` + #51920 workaround + #51538's `47f6574`,
GLM-5.2 TP=4, FLASHINFER_MLA_SPARSE_SM120, fp8_ds_mla, ctx 200000, seqs 6, MTP k=2.
The control survives light single-turn traffic across idle gaps and dies under real load.

| cell | variable | verdict | died at |
|---|---|---|---|
| `eager` | --enforce-eager (no CUDA graphs at all) | **WEDGED** | cycle 1 / storm |

## Findings so far

**CUDA graphs are not the cause.** `--enforce-eager` removes graph replay entirely and
the engine still hangs, and it hangs *earlier*: the control survives the storm and dies on the
following drain, while eager died inside the storm itself on cycle 1. That eliminates the
graph-pointer-staleness branch and sharpens the trigger: the operative condition is concurrent
load, not the idle transition I originally described. Eager also makes every step slower, so it
spends longer in the dangerous region, which fits a race that scales with time under concurrency.

Cells still to run are listed in the table as they land. Nothing here is filtered:
a hypothesis that dies gets published the same as one that survives.
