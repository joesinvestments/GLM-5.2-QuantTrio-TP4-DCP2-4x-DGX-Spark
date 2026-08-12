#!/usr/bin/env python3
"""Load-then-drain wedge trigger for GLM on the GX10 fleet.

PREDICATE HISTORY (both failures are in here on purpose):
  v1 treated a 180 s request timeout as proof of a wedge. Every cell then "wedged" at exactly
     180 s, because a C=6 storm legitimately exceeds that on slower configs. Four invalid
     verdicts published and retracted.
  v2 judged on counters, but "counters frozen + running=0" is ALSO what a healthy idle engine
     looks like. It happened to be right on the control only because a human checked
     separately with a fresh request.
  v3 (this one) never infers death from silence. After any timeout it ASKS THE ENGINE TO WORK:
     a small fresh completion. Serving it means alive-but-slow. Failing it too means wedged.
     An idle engine passes this trivially, so idleness can never be mistaken for a wedge.
"""
import json, sys, time, random, urllib.request, concurrent.futures as cf
BASE="http://192.168.1.16:8210"; LABEL=sys.argv[1]
CYCLES=int(sys.argv[2]) if len(sys.argv)>2 else 3
IDLE=int(sys.argv[3]) if len(sys.argv)>3 else 120
REQ_TIMEOUT=600      # slow is not dead
FRESH_TIMEOUT=120    # a small request the engine must be able to serve
NONCE=int(time.time())

def counters():
    try:
        t=urllib.request.urlopen(BASE+"/metrics",timeout=15).read().decode()
        g=lambda n: sum(float(l.rsplit(' ',1)[1]) for l in t.splitlines()
                        if l.startswith(n) and not l.startswith('#'))
        return g("vllm:generation_tokens_total"), g("vllm:num_requests_running")
    except Exception:
        return None, None

def turn(ptok, otok, seed, timeout=REQ_TIMEOUT):
    rnd=random.Random(seed+NONCE)
    words=" ".join(f"v{rnd.randint(0,10**9)}" for _ in range(int(ptok*0.75)))
    body=json.dumps({"model":"glm-5.2-quanttrio","max_tokens":otok,
        "chat_template_kwargs":{"enable_thinking":False},
        "messages":[{"role":"user","content":"Data: "+words+"\nSummarize in one sentence."}]}).encode()
    t0=time.monotonic()
    urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",body,
        {"Content-Type":"application/json"}), timeout=timeout).read()
    return time.monotonic()-t0

def engine_serves():
    """THE predicate: can the engine still do work? Two attempts, generous timeout."""
    for attempt in (1,2):
        try:
            s=turn(40, 8, 900+attempt, timeout=FRESH_TIMEOUT)
            return True, f"fresh completion in {s:.1f}s (attempt {attempt})"
        except Exception as e:
            last=type(e).__name__
            g,run=counters()
            if g is None: return False, "metrics unreachable and no completion"
    return False, f"fresh completion failed twice ({last}), running={run}"

def phase(name, fn):
    try:
        fn(); return None
    except Exception as e:
        ok, why = engine_serves()
        if ok: return ("SLOW", f"{name}: {type(e).__name__} but {why}")
        return ("WEDGED", f"{name}: {type(e).__name__}, then {why}")

for c in range(1, CYCLES+1):
    notes=[]
    phases=(("storm", lambda: [f.result() for f in [cf.ThreadPoolExecutor(6).submit(turn,1200,150,c*100+i) for i in range(6)]]),
            ("deep-prefill", lambda: turn(20000,100,c*7)),
            ("drain-idle", lambda: time.sleep(IDLE)),
            ("post-drain", lambda: turn(50,10,999,timeout=FRESH_TIMEOUT)))
    for nm, fn in phases:
        r = phase(nm, fn)
        if r and r[0]=="WEDGED":
            print(json.dumps({"label":LABEL,"cycle":c,"verdict":"WEDGED","detail":r[1],"notes":notes}), flush=True)
            sys.exit(0)
        if r: notes.append(r[1])
    print(json.dumps({"label":LABEL,"cycle":c,"verdict":"OK","notes":notes}), flush=True)
print(json.dumps({"label":LABEL,"verdict":"SURVIVED","cycles":CYCLES}), flush=True)
