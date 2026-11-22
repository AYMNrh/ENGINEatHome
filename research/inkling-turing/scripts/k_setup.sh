set -e
R=~/llm/experiments/inkling-kernels
cd $R
if [ ! -d llama.cpp ]; then
  git clone refs/speed-fork llama.cpp
  cd llama.cpp
  git checkout -b turing-iq3-opt
  git config user.name "kernel-experiments"; git config user.email "kernel-experiments@localhost"
else
  cd llama.cpp
fi
git log --oneline -1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_FLAGS="-lineinfo" > $R/cmake-config.log 2>&1
echo CONFIG_DONE
cmake --build build -j 10 --target llama-bench llama-server llama-perplexity test-backend-ops > $R/build.log 2>&1
echo BUILD_DONE rc=$?
