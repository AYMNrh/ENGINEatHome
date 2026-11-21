cd ~/llm/experiments/inkling-kernels
cp mmvq-iq.cuh llama.cpp/ggml/src/ggml-cuda/mmvq-iq.cuh
cd llama.cpp
cmake --build build -j 12 --target llama-bench llama-server test-backend-ops 2>&1 | grep -E "error|Error|Linking CXX executable bin/(llama-bench|llama-server)" | head -20
echo build-done
