#!/usr/bin/env python3
"""analyze.py — turn one or more bench.sh run dirs into a results table.

Usage: python3 analyze.py runs/<dir> [runs/<dir> ...]

For each phase it reports llama-bench tokens/s, and the MEDIAN of the macmon
samples that fall inside that phase's wall-clock window (first 1 s dropped to
skip model load / ramp): GPU busy, GPU MHz, GPU W, RAM W, total W, ANE W.

Derived numbers (both are approximations, stated in the README):
  decode GB/s   = tg tokens/s x weight bytes read per token
                  dense: whole file; MoE: pass --active-bytes
  prefill TFLOP/s = 2 x params x pp tokens/s   (ignores attention FLOPs)
"""
import argparse, datetime as dt, json, statistics as st, sys
from pathlib import Path

PEAK_GBPS = 546.0  # M4 Max (40-core GPU) unified memory bandwidth, Apple spec


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def load(run):
    run = Path(run)
    env = dict(l.split(": ", 1) for l in (run / "env.txt").read_text().splitlines() if ": " in l)
    phases = [json.loads(l) for l in (run / "phases.jsonl").read_text().splitlines()]
    mm = [json.loads(l) for l in (run / "macmon.jsonl").read_text().splitlines() if l.strip()]
    for m in mm:
        m["_t"] = ts(m["timestamp"])
    out = {"run": run.name, "env": env, "phases": {}}
    for p in phases:
        a, b = ts(p["start"]) + dt.timedelta(seconds=1), ts(p["end"])
        win = [m for m in mm if a <= m["_t"] <= b]
        bench = json.loads((run / f"{p['phase']}.json").read_text())
        r = bench[0]
        med = lambda f: round(st.median(f(m) for m in win), 2) if win else None
        out["phases"][p["phase"]] = {
            "tok_s": round(r["avg_ts"], 2), "tok_s_sd": round(r["stddev_ts"], 2),
            "n_params": r.get("model_n_params"), "model_bytes": r.get("model_size"),
            "samples": len(win),
            "gpu_busy": med(lambda m: m["gpu_usage"][1]),
            "gpu_mhz": med(lambda m: m["gpu_usage"][0]),
            "gpu_W": med(lambda m: m["gpu_power"]),
            "ram_W": med(lambda m: m["ram_power"]),
            "all_W": med(lambda m: m["all_power"]),
            "ane_W": med(lambda m: m["ane_power"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--active-bytes", type=float, default=None,
                    help="bytes of weights read per decoded token (MoE); default = model file bytes")
    ap.add_argument("--active-params", type=float, default=None,
                    help="params used per token for the prefill FLOP estimate (MoE)")
    a = ap.parse_args()
    for run in a.runs:
        r = load(run)
        print(f"\n## {r['run']}\n{r['env'].get('model','')}")
        print("| phase | tok/s | GPU busy | GPU MHz | GPU W | RAM W | total W | ANE W | derived |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, p in r["phases"].items():
            d = ""
            if name == "tg" and p["model_bytes"]:
                gb = p["tok_s"] * (a.active_bytes or p["model_bytes"]) / 1e9
                d = f"{gb:.0f} GB/s = {100*gb/PEAK_GBPS:.0f}% of {PEAK_GBPS:.0f}"
            if name == "pp" and p["n_params"]:
                tf = 2 * (a.active_params or p["n_params"]) * p["tok_s"] / 1e12
                d = f"{tf:.1f} TFLOP/s (2·N·tok/s)"
            print(f"| {name} | {p['tok_s']} ± {p['tok_s_sd']} | {p['gpu_busy']} | {p['gpu_mhz']} | "
                  f"{p['gpu_W']} | {p['ram_W']} | {p['all_W']} | {p['ane_W']} | {d} |")
        (Path(run) / "summary.json").write_text(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
