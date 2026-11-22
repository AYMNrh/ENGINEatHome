#!/bin/bash
# usage: kgpu.sh script.sh [args]  -> runs under gpuexec (Swift-safe), output to sessions/<n>.log
cd ~/llm/experiments/inkling-kernels; mkdir -p sessions
LOG=sessions/$(date +%H%M%S)-$(basename $1 .sh).log
systemd-run --user --collect --unit=kgpu-$$ -p StandardOutput=append:$HOME/llm/experiments/inkling-kernels/$LOG -p StandardError=append:$HOME/llm/experiments/inkling-kernels/$LOG /usr/bin/python3 -u $HOME/llm/experiments/inkling-kernels/gpuexec.py $HOME/llm/experiments/inkling-kernels/$1 "${@:2}" >/dev/null
echo $LOG
