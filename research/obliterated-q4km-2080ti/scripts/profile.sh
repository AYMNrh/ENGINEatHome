#!/bin/bash
# usage: profile.sh TAG START DEPTH(0=no mtp)   -- nsys CUDA trace of one agent turn, serialized by gpu.lock
cd ~/llm/experiments/obliterated-opt
TAG=$1; START=$2; D=$3
exec 9>gpu.lock; flock -n 9 || { echo "GPU lock busy"; exit 1; }
[ "$(systemctl is-active engineathome-api.service)" = "active" ] && { echo gateway active; exit 1; }
SPEC=""; [ "$D" != "0" ] && SPEC="--spec-type draft-mtp --spec-draft-n-max $D --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-ubatch 64"
nsys profile -o raw/prof_$TAG --force-overwrite true -t cuda --sample=none --cpuctxsw=none --stats=false \
  ~/llm/llama.cpp/build/bin/llama-server --model ~/llm/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf --ctx-size 65536 --host 127.0.0.1 --port 18090 --parallel 1 -ngl 99 --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 -b 512 -ub 128 --threads 8 --fit off --no-context-shift --cache-ram 0 --no-webui --jinja --no-warmup $SPEC > raw/server_prof_$TAG.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:18090/health | grep -q ok && break; sleep 2; done
python3 scripts/agentsim2.py 18090 $START 1 256 prof_$TAG raw/ledger_prof.jsonl - greedy
P=$(pgrep -x llama-server); kill $P; wait
nsys stats -r cuda_gpu_kern_sum --format csv raw/prof_$TAG.nsys-rep > raw/prof_${TAG}_kern.csv 2>/dev/null
ls -la raw/prof_$TAG*
