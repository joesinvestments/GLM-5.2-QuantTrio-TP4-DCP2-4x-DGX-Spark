# GLM-5.2 (QuantTrio Int4-Int8) — TP=4 + DCP2 on 4× DGX Spark, tuned for a real agent workload

This is my production GLM-5.2 serving config on a 4-node NVIDIA DGX Spark (GB10, sm_121a)
cluster over 200G RoCEv2, and the measured tuning window that produced it.

**The premise, up front:** I didn't tune this for benchmark screenshots. I built and tuned it
completely around my own agent's (Hermes) coding workloads — thousands of real requests a day,
and every decision below was measured at that traffic shape. As far as I know I'm the only
person building around that specific premise, and it changes the answers. A config tuned at
`max_num_seqs=1` on hand-picked prompts and a config tuned under a live agent are different
animals.

## The v4 config (current production)

- **Stack:** local-inference-lab/vllm @e232d26 + PR#72 + draft-quant packed mapping + b12x,
  plus two patches of mine (below). Weights: QuantTrio Int4-Int8Mix, unpruned, 256 experts.
- **Parallelism:** TP=4, `--distributed-executor-backend mp`, **DCP2** (decode context
  parallel), interleave 1, attention `B12X_MLA_SPARSE`, moe `flashinfer_cutlass`.
- **Spec decode:** adaptive MTP ladder **k=2/4/5** (CosmicRaisins controller), probabilistic
  draft sampling, `quantization: compressed-tensors` on the draft (if you skip that field your
  drafter silently loads garbage and acceptance craters — this is the single most common bug
  in community GLM/DeepSeek MTP configs).
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
| **3b** | **131K ctx / mnbt 2048** | **14.7 (+158%)** | **345 (+23%)** | **72.2%** | **winner, in production** |

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
