#!/usr/bin/env python3
"""The REAL trigger, learned 2026-08-12: light idle does not wedge this engine; LOAD then
DRAIN does. Each cycle: concurrent storm + a deep prefill, let the batch drain, idle, then
probe. Reports the cycle at which the engine stops answering, or SURVIVED.
Usage: wedge_trigger.py <label> [cycles] [idle_s]"""
import json, sys, time, random, urllib.request, concurrent.futures as cf
BASE="http://192.168.1.16:8210"; LABEL=sys.argv[1]
CYCLES=int(sys.argv[2]) if len(sys.argv)>2 else 3
IDLE=int(sys.argv[3]) if len(sys.argv)>3 else 120
NONCE=int(time.time())

def turn(ptok, otok, seed, timeout=180):
    rnd=random.Random(seed+NONCE)
    words=" ".join(f"v{rnd.randint(0,10**9)}" for _ in range(int(ptok*0.75)))
    body=json.dumps({"model":"glm-5.2-quanttrio","max_tokens":otok,
        "chat_template_kwargs":{"enable_thinking":False},
        "messages":[{"role":"user","content":"Data: "+words+"\nSummarize in one sentence."}]}).encode()
    t0=time.monotonic()
    urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",body,
        {"Content-Type":"application/json"}), timeout=timeout).read()
    return time.monotonic()-t0

def alive(timeout=90):
    try: return round(turn(50, 10, 999, timeout), 1)
    except Exception as e: return type(e).__name__

for c in range(1, CYCLES+1):
    phase="storm"
    try:
        with cf.ThreadPoolExecutor(6) as ex:      # C=6, production ceiling
            list(ex.map(lambda i: turn(1200,150,c*100+i), range(6)))
        phase="deep-prefill"
        turn(20000, 100, c*7)                      # ~27K real tokens
        phase="drain-idle"
        time.sleep(IDLE)
        phase="post-drain probe"
        r=alive()
        if isinstance(r, str):
            print(json.dumps({"label":LABEL,"cycle":c,"verdict":"WEDGED","phase":phase,"err":r}), flush=True); sys.exit(0)
        print(json.dumps({"label":LABEL,"cycle":c,"verdict":"OK","post_drain_s":r}), flush=True)
    except Exception as e:
        print(json.dumps({"label":LABEL,"cycle":c,"verdict":"WEDGED","phase":phase,"err":type(e).__name__}), flush=True)
        sys.exit(0)
print(json.dumps({"label":LABEL,"verdict":"SURVIVED","cycles":CYCLES}), flush=True)
