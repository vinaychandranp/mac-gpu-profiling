#!/usr/bin/env python3
"""siglip_bench.py — image-encoder throughput for SigLIP 2 on Apple Silicon (PyTorch MPS).

Usage: python3 siglip_bench.py <hf-model-id> [--batches 1,8,32] [--dtypes fp32,fp16] [--secs 8]

For each (dtype, batch) it runs the VISION tower only on random pixel tensors at the
model's native resolution, synchronizing MPS around a timed loop of ~--secs seconds,
and writes phase windows so macmon samples can be sliced (same format as bench.sh).

It also runs a tiny zero-shot sanity check on real images (--images) so we know the
model is doing something sensible, not just running fast.

Output: runs/<ts>-<label>/{env.txt, phases.jsonl, results.json, macmon.jsonl, zeroshot.txt}
"""
import argparse, datetime as dt, json, os, platform, subprocess, time
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--batches", default="1,8,32")
ap.add_argument("--dtypes", default="fp32,fp16")
ap.add_argument("--secs", type=float, default=8.0)
ap.add_argument("--images", nargs="*", default=[])
ap.add_argument("--zeroshot-only", action="store_true")
ap.add_argument("--labels", default="a city park with trees,a cat,a plate of food,a motorcycle,a living room with plants,a screenshot of text")
a = ap.parse_args()

label = a.model.split("/")[-1]
here = Path(__file__).resolve().parent
out = here / "runs" / f"{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}-{label}"
out.mkdir(parents=True)

dev = torch.device("mps")
t0 = time.time()
model = AutoModel.from_pretrained(a.model)
proc = AutoProcessor.from_pretrained(a.model)
load_s = time.time() - t0
vis = model.vision_model
n_vis = sum(p.numel() for p in vis.parameters())
n_all = sum(p.numel() for p in model.parameters())
cfg = model.config.vision_config
res = cfg.image_size

(out / "env.txt").write_text("\n".join([
    f"date_utc: {now()}",
    f"machine: {platform.machine()} / {subprocess.getoutput('sysctl -n machdep.cpu.brand_string')}",
    f"os: macOS {platform.mac_ver()[0]}",
    f"torch: {torch.__version__}",
    f"model: {a.model}",
    f"vision_params: {n_vis}",
    f"total_params: {n_all}",
    f"image_size: {res}  patch: {cfg.patch_size}  tokens_per_image: {(res // cfg.patch_size) ** 2}",
    f"load_seconds: {load_s:.1f}",
]) + "\n")
print((out / "env.txt").read_text())

mm = subprocess.Popen(["macmon", "pipe", "-i", "500"], stdout=open(out / "macmon.jsonl", "w"),
                      stderr=subprocess.DEVNULL)
results = []
if a.zeroshot_only:
    a.dtypes = ""
try:
    time.sleep(4)
    for dname in [d for d in a.dtypes.split(",") if d]:
        dtype = {"fp32": torch.float32, "fp16": torch.float16}[dname]
        v = vis.to(dev, dtype=dtype).eval()
        for bs in [int(b) for b in a.batches.split(",")]:
            x = torch.randn(bs, 3, res, res, device=dev, dtype=dtype)
            with torch.inference_mode():
                for _ in range(3):  # warmup
                    v(pixel_values=x)
                torch.mps.synchronize()
                start = now(); t = time.time(); n = 0
                while time.time() - t < a.secs:
                    v(pixel_values=x); n += 1
                    if n % 4 == 0:
                        torch.mps.synchronize()
                torch.mps.synchronize()
                el = time.time() - t
            end = now()
            r = {"dtype": dname, "batch": bs, "iters": n, "seconds": round(el, 3),
                 "images_per_s": round(n * bs / el, 2), "ms_per_batch": round(1000 * el / n, 2)}
            results.append(r); print(r)
            with open(out / "phases.jsonl", "a") as f:
                f.write(json.dumps({"phase": f"{dname}-b{bs}", "start": start, "end": end}) + "\n")
            time.sleep(3)
finally:
    mm.terminate()
(out / "results.json").write_text(json.dumps(results, indent=2))

# zero-shot sanity check (fp32, full model)
if a.images:
    from PIL import Image
    labels = a.labels.split(",")
    m = model.to(dev, dtype=torch.float32).eval()
    lines = []
    for path in a.images:
        img = Image.open(path).convert("RGB")
        inp = proc(text=labels, images=img, padding="max_length", max_length=64, return_tensors="pt").to(dev)
        with torch.inference_mode():
            p = torch.sigmoid(m(**inp).logits_per_image)[0].cpu()
        top = sorted(zip(labels, p.tolist()), key=lambda z: -z[1])[:3]
        lines.append(f"{Path(path).name}: " + " | ".join(f"{l} {s:.3f}" for l, s in top))
    (out / "zeroshot.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
print("done ->", out)
