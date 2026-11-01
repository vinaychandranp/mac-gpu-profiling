# mac-gpu-profiling

Prefill vs decode for a dense and an MoE LLM on an Apple M4 Max, measured with
`llama-bench` and sampled with [macmon](https://github.com/vladkens/macmon)
(no sudo). Everything needed to check or rerun the numbers is in this repo.

## Machine and software

| | |
|---|---|
| machine | Mac Studio (Mac16,9), Apple M4 Max, 40-core GPU |
| memory | 48 GB unified, ~546 GB/s peak (Apple spec) |
| OS | macOS 26.6.2 (25G83) |
| llama.cpp | 0.8.0 (build 11960, commit 8642541b2), Homebrew, Metal, `-ngl 99` |
| macmon | 0.10.0, `macmon pipe -i 500` (2 Hz) |

## Models

| model | type | file | active weights read per decoded token |
|---|---|---|---|
| [Qwen3.8-27B](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) Q4_K_M | dense, 64 layers, 3:1 linear/full attention | 18.96 GB | **18.25 GB** (all but the embedding table) |
| [Flint](https://huggingface.co/RecursiveSI/flint-int4-gguf) int4 Q4_1 | MoE, 49 layers, 256 routed experts, top-6, + shared experts | 25.99 GB | **1.66 GB** (routed experts × 6/256 + everything else) |

Active bytes come from `active_bytes.py`, which reads tensor sizes out of the GGUF.
SHA-256 of each file is in `runs/*/env.txt`.

## Results (batch 1, pp = 512-token prompt, tg = 128 generated tokens)

Phase metrics are **medians of the macmon samples inside each phase's wall-clock
window** (first second dropped).

| model | phase | tok/s | GPU busy | GPU W | **RAM W** | derived |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | prefill | 252.6 ± 0.1 | 100% | **59.3** | 2.9 | 2·N·tok/s ≈ **13.6 TFLOP/s** |
| Qwen3.8-27B | decode | 22.7 ± 0.0 | 100% | **40.2** | **18.6** | 22.7 × 18.25 GB ≈ **415 GB/s = 76% of peak** |
| Flint | prefill | 1798 ± 5 | 100% | 51.5 | 5.9 | 2·N_active·tok/s ≈ 9.5 TFLOP/s (N_active ≈ 2.65B, rough) |
| Flint | decode | 117.1 ± 1.5 | 100% | **19.9** | 9.5 | 117 × 1.66 GB ≈ **194 GB/s = 36% of peak** |

Reference points: an 8192² fp32 MPS matmul loop ran at 13.4 TFLOP/s, GPU 100% @ 1578 MHz,
~66 W. ANE power was 0.0 W in every phase (llama.cpp's Metal backend does not use it).

## What it shows

1. **Dense decode is memory-bound, and you can see it in the power rails.** "GPU busy"
   reads 100% in both phases, because stalled cycles count as busy. The split is
   in the power: prefill is GPU-heavy (59 W GPU, 3 W DRAM); decode drops GPU power by
   a third and DRAM power rises ~6× (40 W / 19 W). Qwen decode reaches ~76% of peak
   bandwidth, about what a well-tuned memory-bound kernel gets.
2. **Dense prefill sits at the compute ceiling.** 13.6 TFLOP/s by 2·N·tok/s, the same
   as the plain fp32 matmul test.
3. **MoE decode is 5× faster but NOT bandwidth-bound.** Flint reads ~11× fewer bytes per
   token than Qwen, yet decodes only ~5× faster, using ~36% of peak bandwidth, with
   *lower* GPU and DRAM power than dense decode. At batch 1 something other than raw
   bytes dominates: scattered expert gathers, small matmuls (expert FFN width 512),
   routing and per-layer launch/sync overhead. Which one is the open question, and the
   interesting one for expert prefetching.

## Caveats (please poke at these)

- The machine was in light interactive use during the runs (a person using it, a
  download in the background during one Qwen run). Repeat runs agreed within ~2% for
  decode; Flint prefill differed 1610 vs 1798 t/s between a 3-rep and a 25-rep run.
- "Active bytes" is a **lower bound** on traffic: it ignores KV cache / recurrent
  state, activations and the one embedding row. For Flint it also assumes 6 distinct
  experts per layer per token, read once.
- Prefill TFLOP/s uses 2·N·tok/s, which ignores attention FLOPs (small at 512 tokens).
  Flint's N_active ≈ 2.65B is converted from active bytes at ~5 bits/weight: rough.
- macmon's `all_power` does not include `ram_power`; both are reported as-is.
- Batch 1 only. MoE behaviour at larger batches (more distinct experts per step) is
  the obvious next experiment, along with speculative decoding (both models ship
  draft/MTP heads).

## Reproduce

```sh
brew install llama.cpp macmon
./bench.sh /path/to/model.gguf label            # PP=512 TG=128 REPS=3 by default
python3 analyze.py runs/<dir> --active-bytes <bytes-per-token>
pip install gguf numpy && python3 active_bytes.py /path/to/model.gguf
```

The Flint run used `REPS=25` so prefill lasted long enough for macmon to sample.

`runs/<dir>/` holds `env.txt` (machine, versions, model sha256), `phases.jsonl` (UTC
windows), `pp.json` / `tg.json` (llama-bench output), `macmon.jsonl` (raw 2 Hz
samples) and `summary.json`.
