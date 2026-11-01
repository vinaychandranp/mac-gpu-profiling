#!/usr/bin/env bash
# bench.sh — profile LLM prefill and decode on Apple Silicon with llama.cpp + macmon.
#
# Usage: ./bench.sh <model.gguf> [label]
#
# Runs prefill (pp) and decode (tg) as SEPARATE llama-bench invocations so each
# phase has its own wall-clock window, and samples macmon (no sudo) at 2 Hz the
# whole time. Everything lands in runs/<timestamp>-<label>/:
#   env.txt        machine, OS, llama.cpp version, model size + sha256
#   phases.jsonl   {"phase","start","end"} in UTC, for slicing the macmon log
#   pp.json        llama-bench JSON for prefill
#   tg.json        llama-bench JSON for decode
#   macmon.jsonl   raw macmon samples (power, GPU busy/freq, RAM, temps)
#
# Then: python3 analyze.py runs/<dir>
set -euo pipefail

MODEL="${1:?usage: ./bench.sh <model.gguf> [label]}"
LABEL="${2:-$(basename "$MODEL" .gguf)}"
PP="${PP:-512}"      # prompt tokens for the prefill test
TG="${TG:-128}"      # generated tokens for the decode test
REPS="${REPS:-3}"
IDLE_S="${IDLE_S:-5}" # idle seconds before/between/after phases (baseline)

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/runs/$(date -u +%Y%m%dT%H%M%SZ)-$LABEL"
mkdir -p "$OUT"

now() { python3 -c 'import datetime;print(datetime.datetime.now(datetime.timezone.utc).isoformat())'; }

{
  echo "date_utc: $(now)"
  echo "machine: $(sysctl -n hw.model) / $(sysctl -n machdep.cpu.brand_string)"
  echo "gpu: $(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2" cores"}')"
  echo "memory_bytes: $(sysctl -n hw.memsize)"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "llama.cpp: $(llama-bench --version 2>&1 | grep -i '^version' || true)"
  echo "macmon: $(macmon --version 2>&1 | head -1)"
  echo "model: $MODEL"
  echo "model_bytes: $(stat -f %z "$MODEL")"
  echo "model_sha256: $(shasum -a 256 "$MODEL" | cut -d' ' -f1)"
  echo "params: PP=$PP TG=$TG REPS=$REPS ngl=99"
} > "$OUT/env.txt"
cat "$OUT/env.txt"

macmon pipe -i 500 > "$OUT/macmon.jsonl" 2>/dev/null &
MM=$!
trap 'kill $MM 2>/dev/null || true' EXIT

phase() { # name, then llama-bench args
  local name="$1"; shift
  local s; s="$(now)"
  llama-bench -m "$MODEL" -ngl 99 -r "$REPS" -o json "$@" > "$OUT/$name.json" 2>"$OUT/$name.log"
  printf '{"phase":"%s","start":"%s","end":"%s"}\n' "$name" "$s" "$(now)" >> "$OUT/phases.jsonl"
  sleep "$IDLE_S"
}

sleep "$IDLE_S"
phase pp -p "$PP" -n 0
phase tg -p 0 -n "$TG"

kill $MM 2>/dev/null || true
echo "done -> $OUT"
