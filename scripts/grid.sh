#!/usr/bin/env bash
# Run the drift x verbosity grid for one model and collect the summary lines.
#   scripts/grid.sh anthropic:claude-haiku-5-5 7
set -u
MODEL="${1:-anthropic:claude-haiku-5-5}"
SEED="${2:-7}"
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY="python"
OUT="runs/grid-$(date +%Y%m%d-%H%M%S)-$SEED"
mkdir -p "$OUT"
for d in 0 1 2 3; do
  for v in 3 2 1 0; do
    f="$OUT/d${d}-v${v}"
    "$PY" -m tiny_api_agent.eval --spec examples/library.json --tasks examples/library_tasks.json \
      --model "$MODEL" --verbosity "$v" --drift "$d" --seed "$SEED" --out "$f.jsonl" > "$f.log" 2>&1
    echo "d=$d v=$v rc=$? | $(grep -E '^model=' "$f.log")"
    grep -E '^(clean|repair-only)' "$f.log" | sed 's/^/        /'
  done
done
echo "grid -> $OUT"
