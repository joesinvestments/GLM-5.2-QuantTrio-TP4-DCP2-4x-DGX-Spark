#!/usr/bin/env python3
"""One window cell's probe battery at the MEASURED production shape (2026-08-10 18h profile:
mean prompt 960, 48% in 1-2K band, mean output 145, bursts). Metrics-delta methodology:
server-side prefill/decode second sums are the clock, client wall is the check."""
import json, time, random, sys, urllib.request, concurrent.futures as cf

BASE = "http://192.168.1.16:8210"
CELL = sys.argv[1]
NONCE = int(time.time())  # cache-buster: seeds must differ across runs or "cold" prefill hits the prefix cache

def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    out = {}
    for pfx in ["vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count",
                "vllm:request_decode_time_seconds_sum", "vllm:prompt_tokens_total",
                "vllm:generation_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_accepted_tokens_total", "vllm:request_success_total"]:
        out[pfx] = sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines()
                       if l.startswith(pfx) and not l.startswith("#"))
    return out

def req(prompt_tokens, out_tokens, seed):
    rnd = random.Random(seed + NONCE)
    # incompressible pseudo-corpus -> genuinely cold prefill, no cache assist
    words = " ".join(f"v{rnd.randint(0,10**9)}" for _ in range(int(prompt_tokens*0.75)))
    body = json.dumps({"model": "glm-5.2-quanttrio", "max_tokens": out_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Data block: " + words + "\nSummarize the pattern in one sentence."}]}).encode()
    t0 = time.monotonic()
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", body,
        {"Content-Type": "application/json"}), timeout=600)
    j = json.loads(r.read())
    return time.monotonic() - t0, j["usage"]["completion_tokens"]

results = {"cell": CELL}
# ── probe B': production storm, C=12, 1.2K prompts ──
m0 = metrics(); t0 = time.monotonic()
with cf.ThreadPoolExecutor(12) as ex:
    walls = list(ex.map(lambda i: req(1200, 150, 1000+i), range(12)))
storm_wall = time.monotonic() - t0; m1 = metrics()
d = lambda k: m1[k] - m0[k]
results["storm"] = {
    "wall_s": round(storm_wall, 1), "errors": 12 - len(walls),
    "prefill_toks_per_s": round(d("vllm:prompt_tokens_total") / max(d("vllm:request_prefill_time_seconds_sum"), 1e-9), 1),
    "agg_decode_toks_per_s": round(d("vllm:generation_tokens_total") / max(storm_wall, 1e-9), 1),
    "acceptance_pct": round(100 * d("vllm:spec_decode_num_accepted_tokens_total") / max(d("vllm:spec_decode_num_draft_tokens_total"), 1), 1)}
time.sleep(3)
# ── probe C1: clean single-stream decode, 3 runs ──
m0 = metrics(); c1 = []
for i in range(3):
    w, ct = req(1000, 300, 2000+i); c1.append((w, ct))
m1 = metrics()
dec_s = m1["vllm:request_decode_time_seconds_sum"] - m0["vllm:request_decode_time_seconds_sum"]
gen = m1["vllm:generation_tokens_total"] - m0["vllm:generation_tokens_total"]
results["c1"] = {"decode_toks_per_s": round(gen / max(dec_s, 1e-9), 1),
    "prefill_toks_per_s": round((m1["vllm:prompt_tokens_total"]-m0["vllm:prompt_tokens_total"]) /
        max(m1["vllm:request_prefill_time_seconds_count"]*0 + (m1["vllm:request_prefill_time_seconds_sum"]-m0["vllm:request_prefill_time_seconds_sum"]), 1e-9), 1),
    "acceptance_pct": round(100 * (m1["vllm:spec_decode_num_accepted_tokens_total"]-m0["vllm:spec_decode_num_accepted_tokens_total"]) /
        max(m1["vllm:spec_decode_num_draft_tokens_total"]-m0["vllm:spec_decode_num_draft_tokens_total"], 1), 1)}
# ── probe D: 30K deep single (reference point, not the current regime) ──
m0 = metrics(); w, ct = req(30000, 200, 3000); m1 = metrics()
results["deep30k"] = {"wall_s": round(w, 1),
    "prefill_toks_per_s": round((m1["vllm:prompt_tokens_total"]-m0["vllm:prompt_tokens_total"]) /
        max(m1["vllm:request_prefill_time_seconds_sum"]-m0["vllm:request_prefill_time_seconds_sum"], 1e-9), 1)}
print(json.dumps(results))
