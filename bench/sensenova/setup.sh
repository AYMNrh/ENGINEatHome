#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime=${1:-"$HOME/models/sensenova-runtime"}
cuda=${CUDA_ROOT:-/usr/local/cuda-13.3}
revision=fad174ecd8bf22746a626c58e67e193f095aaa46
model_revision=5cf786036589460866c676ab515bb7f4dd5790a4
model=SenseNova-Loop-9B-Q4_K_M.gguf
model_sha=983a5899fe29c67a73953f40aa0219044aae80868f2de077f6adb004697b79da

if [[ -e "$runtime" ]]; then
    printf 'Choose a new runtime directory. This path already exists: %s\n' "$runtime" >&2
    exit 2
fi
test -x "$cuda/bin/nvcc"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

git clone --branch sensenova-loop --single-branch https://github.com/GuideboardLabs/llama.cpp-sensenova-loop.git "$runtime"
git -C "$runtime" switch --detach "$revision"
git -C "$runtime" apply --check "$here/evidence/sensenova-mhc-cuda.patch"
git -C "$runtime" apply "$here/evidence/sensenova-mhc-cuda.patch"

cmake -S "$runtime" -B "$runtime/build-cu133" \
    -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
    -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" -DCUDAToolkit_ROOT="$cuda" \
    -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build "$runtime/build-cu133" --target llama-server -j "${BUILD_JOBS:-12}"

curl --fail --location --retry 2 \
    "https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF/resolve/$model_revision/$model" \
    --output "$runtime/$model.part"
printf '%s  %s\n' "$model_sha" "$runtime/$model.part" | sha256sum --check
mv -- "$runtime/$model.part" "$runtime/$model"

mkdir "$runtime/256k-baseline-lib"
cp -L -- "$runtime/build-cu133/bin/libggml-cuda.so.0" "$runtime/256k-baseline-lib/libggml-cuda.so.0"
sha256sum "$runtime/256k-baseline-lib/libggml-cuda.so.0"
