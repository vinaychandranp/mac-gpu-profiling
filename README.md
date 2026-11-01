# mac-gpu-profiling

Prefill vs decode for a dense and an MoE LLM on an Apple M4 Max, measured with
`llama-bench` and sampled with [macmon](https://github.com/vladkens/macmon)
(no sudo). Everything needed to check or rerun the numbers is in this repo.

## Machine and software

| | |
|---|---|
| machine | Mac Studio (Mac16,9), Apple M4 Max, 40-core GPU |
| memory | 48 GB unified, ~546 GB/s peak (Apple spec) |
| GPU fp32 peak | 40 cores × 128 ALUs × 2 (FMA) × 1.578 GHz (clock measured by macmon) ≈ **16.2 TFLOP/s** |
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
| Qwen3.8-27B | prefill | 252.6 ± 0.1 | 100% | **59.3** | 2.9 | 2·N_matmul·tok/s ≈ **12.3–12.9 TFLOP/s** (see note) |
| Qwen3.8-27B | decode | 22.7 ± 0.0 | 100% | **40.2** | **18.6** | 22.7 × 18.25 GB ≈ **415 GB/s = 76% of peak** |
| Flint | prefill | 1798 ± 5 | 100% | 51.5 | 5.9 | 2·N_active·tok/s ≈ 9.5 TFLOP/s (N_active ≈ 2.65B, rough) |
| Flint | decode | 117.1 ± 1.5 | 100% | **19.9** | 9.5 | 117 × 1.66 GB ≈ **194 GB/s = 36% of peak** |

Reference points: an 8192² fp32 MPS matmul loop ran at 13.4 TFLOP/s, GPU 100% @ 1578 MHz,
~66 W, i.e. **83% of the 16.2 TFLOP/s fp32 peak**. ANE power was 0.0 W in every phase (llama.cpp's Metal backend does not use it).

## What it shows

1. **Dense decode is memory-bound, and you can see it in the power rails.** "GPU busy"
   reads 100% in both phases, because stalled cycles count as busy. The split is
   in the power: prefill is GPU-heavy (59 W GPU, 3 W DRAM); decode drops GPU power by
   a third and DRAM power rises ~6× (40 W / 19 W). Qwen decode reaches ~76% of peak
   bandwidth, about what a well-tuned memory-bound kernel gets.
2. **Dense prefill is compute-bound.** ~12.3–12.9 TFLOP/s = **76–80% of the 16.2 TFLOP/s
   fp32 peak**, just under the plain fp32 matmul test (13.4, 83%), as expected once
   dequantization, norms, activations and launch gaps are added.
   *Note on N:* the naive 2·26.9B·tok/s gives 13.6 TFLOP/s, which would beat the plain
   matmul. That is a counting error: the token embedding (248,320 × 5,120 = 1.27B params)
   is a lookup with no FLOPs (→ 25.6B, 12.9 TFLOP/s), and the output head (another 1.27B)
   likely runs only for the last prompt token in llama-bench's pp test (→ 24.4B,
   12.3 TFLOP/s). Attention FLOPs, ignored here, add a little back.
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

## Vision encoders: SigLIP 2 (PyTorch MPS, `siglip_bench.py`)

Vision tower only, random pixels at native resolution, ~8 s timed loop per config.
TFLOP/s = 2 × vision params × patch tokens × images/s (ignores attention FLOPs,
which add ~4–6% here). % is of the 16.2 TFLOP/s fp32 peak.

| model | vision params | tokens/img | config | images/s | ≈TFLOP/s | GPU W | RAM W |
|---|---|---|---|---|---|---|---|
| siglip2-base-patch16-224 | 93M | 196 | fp16, batch 1 | 149 | 5.4 (33%) | 24.7 | 2.1 |
| | | | fp16, batch 32 | **325** | 11.8 (73%) | 54.8 | 2.7 |
| siglip2-giant-opt-patch16-384 | 1,164M | 576 | fp16, batch 1 | 8.1 | 10.8 (67%) | 43.6 | 1.9 |
| | | | fp16, batch 32 | **9.6** | 12.9 (79%) | 61.4 | 6.5 |

fp32 is ~10–20% slower throughout (full table in `runs/*siglip*/results.json`).

- **Compute-bound, like LLM prefill:** low RAM power, high GPU power. The giant model
  reaches ~79% of fp32 peak, close to LLM prefill and the plain matmul.
- **Small models at batch 1 are overhead-bound:** base at batch 1 uses only ~33% of
  peak (7 ms/image is mostly launch/framework cost); batching to 32 more than doubles
  throughput. The giant is big enough to keep the GPU fed even at batch 1.
- **fp16 ≈ fp32 speed:** the gain is small, consistent with the GPU's fp16 and fp32 ALU
  rates being similar on this chip; fp16 mainly saves memory and power.
- Zero-shot sanity check (`zeroshot.txt`): both models rank a park photo, an interior
  with plants, and a book page correctly among six labels.
