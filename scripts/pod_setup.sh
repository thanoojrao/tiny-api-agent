#!/usr/bin/env bash
# Turn a fresh Linux GPU box into a working eval/training environment, then start vLLM.
#
#   git clone <repo> && cd tiny-api-agent
#   cp .env.example .env   # fill in HF_TOKEN
#   scripts/pod_setup.sh [hf-model-id]
#
# Safe to re-run: every step skips itself when already done. Nothing here persists beyond
# the pod; weights come from the Hub, outputs go to git or the Hub, the pod is terminated after.
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${VLLM_MAX_LEN:-8192}"       # longest prompt+completion the server will accept
LOG="runs/vllm.log"
cd "$(dirname "$0")/.."

echo "== 1/5 uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== 2/5 python env (train extra: torch, vllm, trl, peft)"
uv sync --extra train --extra dev

echo "== 3/5 weights: $MODEL"
[ -f .env ] && set -a && . ./.env && set +a
uv run hf download "$MODEL" >/dev/null   # idempotent; lands in the HF cache

echo "== 4/5 vllm serve on :$PORT"
if curl -sf "http://localhost:$PORT/v1/models" | grep -q "\"$MODEL\""; then
  echo "already serving $MODEL"
else
  mkdir -p runs
  nohup uv run vllm serve "$MODEL" --port "$PORT" --dtype bfloat16 \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.85 >"$LOG" 2>&1 &
  echo "pid $! -> $LOG"
  for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" | grep -q "\"$MODEL\""; then break; fi
    if ! kill -0 $! 2>/dev/null; then echo "vllm died, see $LOG"; tail -20 "$LOG"; exit 1; fi
    sleep 5
  done
  curl -sf "http://localhost:$PORT/v1/models" | grep -q "\"$MODEL\"" || { echo "vllm not up after 10 min, see $LOG"; exit 1; }
fi

echo "== 5/5 smoke: one task through the adapter"
uv run python -m tiny_api_agent.eval --spec examples/library.json --tasks examples/library_tasks.json \
  --model "vllm:$MODEL" --limit 1 --verbosity 3

cat <<MSG

ready. baseline grid (seed 7, same as the Haiku run):
  scripts/grid.sh vllm:$MODEL 7
stop the server:  pkill -f "vllm serve"
MSG
