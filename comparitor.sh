#!/bin/bash

set -x
set -e

PORTS=(11435 11436 11437 11438)
# PORTS=(11435)
COMPARITORS_PER_DEVICE=4

SERVER_PIDS=()
COMPARITOR_PIDS=()

cleanup() {
    echo "Shutting down zombie comparitors..."
    for pid in "${COMPARITOR_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    echo "Shutting down Ollama servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

wait_for_server() {
    local port=$1
    local max_wait=60
    local waited=0
    until curl -s -o /dev/null "http://127.0.0.1:${port}/"; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$max_wait" ]; then
            echo "Server on port ${port} did not become ready within ${max_wait}s" >&2
            return 1
        fi
    done
}


# --- Launch one Ollama server per GPU ---
for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    CUDA_VISIBLE_DEVICES="${i}" \
      GGML_VK_VISIBLE_DEVICES=-1 \
      OLLAMA_HOST="127.0.0.1:${port}" \
      OLLAMA_NUM_PARALLEL="${COMPARITORS_PER_DEVICE}" \
      OLLAMA_MULTIUSER_CACHE=true \
      ollama serve > "ollama_gpu${i}_port${port}.log" 2>&1 &
    SERVER_PIDS+=($!)
    wait_for_server "$port"
    echo "Server ready: GPU ${i}, port ${port}, pid ${SERVER_PIDS[-1]}"
done

echo "All ${#PORTS[@]} Ollama servers are up."

# --- Launch comparitor jobs: N per device, all running concurrently ---
for port in "${PORTS[@]}"; do
    for n in $(seq 1 "$COMPARITORS_PER_DEVICE"); do
        PYTHONUNBUFFERED=1 uv run comparitor.py compare --rater llm --prompt prompts/emotionality.md \
            --llm-host "http://127.0.0.1:${port}" --llm-model gemma4:e4b-it-q8_0 \
            /mnt/data/cevaw/corpus_1998_to_2025_v8.parquet \
            > "comparitor_port${port}_${n}.log" 2>&1 &
        COMPARITOR_PIDS+=($!)
    done
done

echo "Launched ${#COMPARITOR_PIDS[@]} comparitor jobs, waiting for completion..."

# Wait for every comparitor job individually so a mid-list failure
# under `set -e` isn't masked (plain `wait pid1 pid2 ...` only
# reports the LAST pid's exit status).
FAILED=0
for pid in "${COMPARITOR_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Comparitor job pid ${pid} failed" >&2
        FAILED=1
    fi
done

echo "All comparitor jobs finished."
exit "$FAILED"
