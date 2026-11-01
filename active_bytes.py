#!/usr/bin/env python3
"""active_bytes.py — estimate weight bytes READ PER DECODED TOKEN for a GGUF model.

Usage: python3 active_bytes.py model.gguf      (needs: pip install gguf numpy)

Dense model: every weight is read once per token, so active bytes ~= file bytes
(minus the embedding table, of which only one row is read).
MoE model: routed-expert tensors are scaled by experts_used / expert_count;
shared experts, attention, norms, router and lm_head are read in full.

This is a lower bound on traffic: it ignores KV cache / recurrent state reads,
activations, and any expert weights re-read across the batch.
"""
import re, sys
from collections import Counter
from gguf import GGUFReader

r = GGUFReader(sys.argv[1])
arch = None
fields = {f.name: f for f in r.fields.values()}
def val(key):
    f = fields.get(key)
    return None if f is None or not f.data else f.parts[f.data[0]][0]

arch = bytes(fields["general.architecture"].parts[fields["general.architecture"].data[0]]).decode()
n_exp = val(f"{arch}.expert_count")
n_used = val(f"{arch}.expert_used_count")

cat = Counter()
for t in r.tensors:
    n, b = t.name, int(t.n_bytes)
    if re.search(r"ffn_(gate|up|down)_exps", n): cat["routed_experts"] += b
    elif "shexp" in n: cat["shared_experts"] += b
    elif n.startswith("token_embd"): cat["token_embd (1 row read/token)"] += b
    elif n.startswith("output."): cat["lm_head"] += b
    else: cat["attention+norms+router+other"] += b

total = sum(cat.values())
frac = (int(n_used) / int(n_exp)) if n_exp else 1.0
active = sum(v for k, v in cat.items() if not k.startswith("token_embd") and k != "routed_experts")
active += cat["routed_experts"] * frac
print(f"arch={arch} experts={n_exp} used={n_used}")
for k, v in cat.most_common():
    print(f"  {k:32s} {v/1e9:8.3f} GB")
print(f"file total        {total/1e9:.3f} GB")
print(f"active per token  {active/1e9:.3f} GB  (routed experts x {frac:.4f})")
