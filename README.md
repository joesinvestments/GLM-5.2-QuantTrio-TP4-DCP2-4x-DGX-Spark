# GLM-5.2 (QuantTrio Int4-Int8) — TP=4 on 4× DGX Spark, tuned for a real agent workload

This is my production GLM-5.2 serving config on a 4-node NVIDIA DGX Spark (GB10, sm_121a)
cluster over 200G RoCEv2, and the measured tuning window that produced it.

**The premise, up front:** I didn't tune this for benchmark screenshots. I built and tuned it
completely around my own agent's (Hermes) coding workloads — thousands of real requests a day,
and every decision below was measured at that traffic shape. As far as I know I'm the only
person building around that specific premise, and it changes the answers. A config tuned at
`max_num_seqs=1` on hand-picked prompts and a config tuned under a live agent are different
animals.

## Current champion: 44.6 tok/s single-stream decode — one env flag (2026-08-10, same day, correction)

Hours after publishing the section below, the kernel investigation it called for landed, and
it **inverts my own headline finding**. The quantized-draft slowness was not the quantized
compute itself — it was Marlin's default small-batch reduce path (`atomic-add off`, fp32
global reduction), at its worst at exactly the draft's tiny-M GEMM shapes, amplified by the
draft's 4 sequential forwards per step. The fix is one environment variable:

```
VLLM_MARLIN_USE_ATOMIC_ADD=1
```

Full 2×2, single-stream cold decode (repeats in `window-data/kernel-exp*.json`):

| draft config | no flag | with flag |
|---|---|---|
| bf16 (unquantized) | 21.1 tok/s @ 38.5% | 19.4 @ 32.9% (no effect) |
| quantized probabilistic | 6.5 @ 61.4% | **44.6 @ 56.8%** (repeats 42.4 / 46.9) |

The flag's entire effect lives in the quantized-draft path. So the drafter advice comes full
circle: **quantize the draft (`quantization:"compressed-tensors"`) AND set the flag** — either
alone is a regression from the other pairing, which is why half the community measures the
"fix" as a win and I measured it as a loss yesterday. Both were right on their own half of
the matrix. Outputs verified coherent and non-empty on repeated runs.

Production config: the challenger image + quantized probabilistic k=4 draft + the flag.
Two-day progression on the same weights and fleet, every step one named change:
**5.7 → 14.7 → 21.1 → 44.6 tok/s (7.8×)**.

**Content-dependence disclosure (added same day, after live agent cross-check):** 44.6 is a
real but content-favorable number — the probe's short-summary outputs draft-accept at
56–61%. On realistic structured-prose generations (250-word analysis prompts) acceptance
runs 33–36% and decode lands at **~19–21 tok/s single-stream**, which my agent's first real
turn through the stack independently confirmed (21.2 tok/s client-measured). Temperature is
not the variable (temp 0 vs 1.0 measured within noise of each other). The 2×2 above is
unaffected — every cell was probed with the identical battery, so the relative conclusions
stand — but if you're comparing MY headline against YOUR prose workload, ~20 is the honest
sustained number and 44+ is the high-acceptance ceiling. I criticized other repos' peaks
for exactly this; same standard applies to mine.

## Previous champion: the legacy stack at 21.1 tok/s (2026-08-10, superseded same day)

After the v4 window below, I rebuilt the *other* GLM stack — the no-DCP "legacy" lineage
(eugr/spark-vllm-docker base at vLLM `ab666069` + the ciprianveg/CosmicRaisins Triton
sparse-MLA mods + the indexer MTP-overhang patch) — and probed it head-to-head against v4
with the identical battery. It won decisively and is now my production config:

| config | C=1 decode tok/s | cold prefill tok/s | accept | verdict |
|---|---|---|---|---|
| v4 (DCP2 stack, tuned — see below) | 14.7 | 282–345 | 72% | superseded |
| **legacy challenger v1** | **21.1** | 282 storm / ~900 C=1 | 38.5% | **champion** |
| + quantized probabilistic draft | 6.5 | 375 | 61.4% | rejected |
| + quantized greedy draft | 7.6 | 351 | 44.1% | rejected |

**Why I'm calling these the best numbers we've seen on this hardware for this workload:**
every published 4×-Spark GLM figure I've checked is either measured at `max_num_seqs=1` on
hand-picked high-acceptance content, counts SSE chunks (undercounts by the acceptance
factor), or measures warm cache and calls it prefill. These are cold, cache-busted,
server-counter-verified numbers at a serving config (12-seq class), cross-checked against
18 hours of live agent traffic — and the progression is measured, not vibes: 5.7 tok/s at
bring-up → 14.7 after the context-sizing window → 21.1 on the rebuilt stack. Same weights,
same fleet, ~3.7× in two days, every step attributable to one named change.

### The finding as measured BEFORE the atomic-add discovery (kept for the record — see correction above)

The community's standard advice for GLM/DeepSeek MTP on these boxes — mine included, until
today — is that a draft config without `quantization:"compressed-tensors"` silently loads a
degraded drafter, and adding it is a pure win (acceptance jumps ~38% → ~61%; I reproduced
exactly that). What nobody measured: on this tree/hardware the properly-quantized w8a16
draft block runs **~3× slower per step**. Net decode: 21.1 tok/s "broken" vs 6.5–7.6
"fixed". I isolated the variables (greedy vs probabilistic, quantized vs not — see
`window-data/challenger*.json`): the cost is the quantized draft *compute*, not the
sampler. **The naive bf16 drafter wins.** If you've applied the popular fix, measure your
end-to-end decode — you may be paying 3× step time for acceptance you can't cash.

Open kernel question with a big prize: if the w8a16 draft path can be made fast
(Marlin/atomic-add config?), 61% acceptance at ~120 ms steps arithmetic says ~30 tok/s.

### Also corrected along the way

The reference repo's README says the DCP branch includes the July head-padding natively —
it does not (I checked the fork source at the pin *and* the branch tip; padding is still
64). The +36–45% prefill numbers were measured on the legacy stack. If you're on the DCP
stack waiting for that prefill by rebuilding, you're rebuilding the wrong tree — the padding
(and those prefill numbers) live in the legacy lineage only. That correction is what
redirected this campaign, and the ~900 tok/s C=1 cold prefill above is that win, realized.

Launch machinery for the champion is in `legacy-stack/` — including
`resolve_gid_and_launch.sh`, which resolves the RoCEv2 GID index dynamically at every boot
and refuses to launch on disagreement. My GID moved 3→4 *during this campaign*; if yours is
hardcoded, your next reboot is a dice roll.

## The v4 config (DCP2 stack — superseded for my workload, kept for >200K sessions)

- **Stack:** local-inference-lab/vllm @e232d26 + PR#72 + draft-quant packed mapping + b12x,
  plus two patches of mine (below). Weights: QuantTrio Int4-Int8Mix, unpruned, 256 experts.
- **Parallelism:** TP=4, `--distributed-executor-backend mp`, **DCP2** (decode context
  parallel), interleave 1, attention `B12X_MLA_SPARSE`, moe `flashinfer_cutlass`.
- **Spec decode:** adaptive MTP ladder **k=2/4/5** (CosmicRaisins controller), probabilistic
  draft sampling, `quantization: compressed-tensors` on the draft (if you skip that field your
  drafter silently loads garbage and acceptance craters). NOTE the stack-dependence: on THIS
  DCP tree the quantized draft is fast and correct (72% accept at these speeds); on the
  legacy ab666069 tree it is a 3× step-time loss — see the champion section. Measure on
  YOUR tree; neither answer transfers.
- **Shape:** `--max-model-len 131072 --max-num-seqs 12 --max-num-batched-tokens 2048`,
  gmu 0.90, kv `fp8_ds_mla`, cudagraph FULL_AND_PIECEWISE capture 72.
- **Fabric:** dual-rail RoCEv2, NCCL 2.30.4, RoCE GID index resolved **dynamically at every
  boot** and verified identical across all nodes — GID indexes are NOT stable across boots
  and hardcoding one cost me a 10-hour outage on another model.

Full recipe: [`recipe/glm-dcp2-v4-speed128k-adaptive.yaml`](recipe/glm-dcp2-v4-speed128k-adaptive.yaml)

## Measured results (window 2026-08-10, all cells probed identically)

Probes: C=12 storm of cold 1.2K-token prompts (my real burst shape), segmented C=1 cold
decode, deep cold prefill. Cold means cache-busted — if your "prefill" number is over ~1000
tok/s on this hardware you are measuring your prefix cache, not your prefill.

| Cell | Config | C=1 decode tok/s | cold prefill tok/s | accept | verdict |
|---|---|---|---|---|---|
| 0 | 262K ctx / mnbt 1024 | 5.7 | 282 | 65.6% | stable baseline |
| 1 | + upstream July Triton kernel set | — | — | — | **crashed fleet on deep prefill** |
| 1b | + head-pad 64→16 only | — | — | — | **wedged under C=12 storm** |
| 2 | 262K ctx / mnbt 2048 | — | — | — | **crashed on deep prefill (indexer law)** |
| **3b** | **131K ctx / mnbt 2048** | **14.7 (+158%)** | **345 (+23%)** | **72.2%** | **window winner (since superseded, above)** |

### The two findings worth stealing

**1. The indexer law.** The DSA sparse-indexer scratch scales with `max_model_len ×
max_num_batched_tokens`. On GB10 at TP=4/DCP2 the survivable product is ≈ **262144 × 1024**.
Exceed it and the engine dies on the first deep cold prefill — not at boot, not on short
prompts, so you'll ship it and find out later. All the working community configs I've checked
respect this product without saying so; mine crashed twice proving it.

**2. Context you don't use costs decode speed you do use.** Past moderate depth GLM's decode
step is dominated by the sparse indexer, and the indexer working set follows the context
window. Halving `max-model-len` 262K→131K (with mnbt rescaled along the constant product)
took single-stream decode from 5.7 to **14.7 tok/s** on identical hardware, weights and
drafter. My live traffic's biggest prompt is ~2K tokens; I was paying a 2.6× decode tax for
window I never touched. Size the window to the workload you actually have, and revert the day
your workload changes — for me that's one launcher run back to the 262K recipe.

### Negative results (so you don't burn the day I burned)

- **The July head-padding prefill kernels (+36-45% upstream) do not patch onto an older tree.**
  The full 10-file set crashes on deep prefill and the padding change alone wedges under
  concurrency: the b12x glue that accepts 16-head alignment postdates my image snapshot.
  If you want that win, rebuild the image from the current upstream tree. There is no shortcut;
  I tried both.
- **mnbt above the indexer product = delayed crash**, see above.
- Adaptive-k works, but note nobody publishing it (including upstream) has shipped an
  adaptive-vs-fixed ablation at concurrency. Mine holds k=5 under real traffic with a smooth
  per-position decay (85/72/58/49/42%), which is what a healthy drafter looks like. If your
  positions collapse (60/28/18…), your draft weights didn't map — fix that before touching k.

## My patches (in `patches/`)

- **`fix-indexer-mtp-overhang.patch`** — the indexer's expanded block-table has no headroom
  for MTP draft spill when `max_model_len % (block_size × cp) == 0`; at ≥3 concurrent
  requests the engine crashes. My exact production shape. Community-reported first; this is
  the anchored patch for the e232d26 tree.
- **`launcher-rank-verification`** — my launcher refuses to declare the fleet SERVING unless
  all 4 ranks are verified running. The head node happily answers `/v1/models` with a dead
  worker, and the first collective then kills the engine with an NCCL retry storm. A
  container name-conflict race handed me a 3/4-rank "healthy" fleet exactly once, which is
  once more than acceptable.

## Operational notes that matter more than the config

- **Streaming clients only.** A non-streaming client with a timeout turns every timeout into
  a zombie request that decodes to full budget while the client retries. Aggregate throughput
  moved ~3× on my other model the day the agent switched to streaming.
- **GLM's chat template deletes prior-turn reasoning.** That rewrites history every turn and
  invalidates the server prefix cache mid-conversation. For multi-turn agent sessions send
  `{"chat_template_kwargs":{"clear_thinking":false}}` — my workload is heavily
  prefix-cache-dependent and this is a real TTFT lever.
- **Thinking eats max_tokens.** GLM will happily spend your entire output budget on reasoning
  and return an empty answer with HTTP 200. Budget ~2× your expected answer, or disable
  thinking on latency-sensitive lanes (`enable_thinking:false`).
- Drop page caches before launch: 405GB of weights through the page cache will trip vLLM's
  free-memory guard at gmu 0.90 on unified memory.

## Credit where due

The stack rides on CosmicRaisins' glm-5.2-gb10 work (fork pin, DCP2 recipes, adaptive-MTP
controller). The indexer-overhang bug and several env caps come out of the tonyd2wild /
0xdfi / drowzeys 4×-Spark lineage — I verified everything against my own fleet before
adopting, and their READMEs are worth your time. Errors here are mine.
