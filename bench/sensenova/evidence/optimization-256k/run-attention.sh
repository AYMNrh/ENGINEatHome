set -e
r=/mnt/f/MODELS/SenseNova-Loop-9B/optimization-256k
reference=/mnt/f/MODELS/SenseNova-Loop-9B/optimization/baseline-262144/results.json
cd ~/models/sensenova-runtime
python3 - "$r" <<'PY'
import json, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1])
deadline = time.monotonic() + 2200
while time.monotonic() < deadline:
    try:
        main = json.loads((out / "mmq-j64/results.json").read_text())
        trace = json.loads((out / "nsys-mmq-j64/results.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        main = trace = {}
    if len(main.get("runs", [])) == 6 and len(trace.get("runs", [])) == 1 and subprocess.run(
            ["pgrep", "-x", "llama-server"], stdout=subprocess.DEVNULL).returncode == 1:
        break
    time.sleep(5)
else:
    raise SystemExit("MMQ candidate did not finish")
PY
python3 <<'PY'
from pathlib import Path
mmq = Path("ggml/src/ggml-cuda/mmq.cu")
s = mmq.read_text()
first = s.index("\n\n        // Keep one full K reduction per tile.")
last = s.index("\n    }\n    GGML_ASSERT(J_best > 0);", first)
mmq.write_text(s[:first]+s[last:])

path = Path("ggml/src/ggml-cuda/convert.cu")
s = path.read_text()
first = s.index("template <bool need_check>\nstatic __global__ void dequantize_block_q8_0_f16")
last = s.index("\ntemplate<typename dst_t>",first)
code = s[first:last].replace("template <bool need_check>", "template <bool need_check, int warps = 1>")
code = code.replace("    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;\n    const int * x0 = ((int *) vx) + blockIdx.x * nint;",
"""    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t tile = int64_t(blockIdx.x)*warps + warp;
    const int64_t i0 = CUDA_Q8_0_NE_ALIGN*tile;
    if constexpr (warps > 1) {
        if (i0 >= k) {
            return;
        }
    }
    const int * x0 = ((int *) vx) + tile*nint;""")
code = code.replace("    __shared__ int vals[nint];", "    __shared__ int tiles[warps][nint];\n    int * vals = tiles[warp];")
code = code.replace("ix0 + threadIdx.x", "ix0 + lane").replace("2*threadIdx.x", "2*lane")
code = code.replace("threadIdx.x % (QK8_0/2)", "lane % (QK8_0/2)").replace("iy/2 + threadIdx.x", "iy/2 + lane")
code = code.replace("    __syncthreads();","""    if constexpr (warps == 1) {
        __syncthreads();
    } else {
        __syncwarp();
    }""")
s=s[:first]+code+s[last:]
point="    if (k % CUDA_Q8_0_NE_ALIGN == 0) {\n        const bool need_check = false;"
assert s.count(point)==1
s=s.replace(point,"""    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // Co-schedule four independent tiles for long, aligned flash-attention K/V conversion.
        static const bool group_warps = getenv("GGML_CUDA_LONG_Q8_CONVERT_WARP4") != nullptr;
        if (group_warps && k >= (1LL << 26)) {
            constexpr int warps = 4;
            dequantize_block_q8_0_f16<false, warps>
                <<<(num_blocks + warps - 1)/warps, warps*WARP_SIZE, 0, stream>>>(vx, y, k);
            return;
        }
#endif
        const bool need_check = false;""")
path.write_text(s)
PY
git diff --check
git diff -- ggml/src/ggml-cuda/convert.cu > "$r/q8-convert-warp4.patch"
cmake --build build-cu133 --target ggml-cuda -j8 > "$r/build-warp4.log" 2>&1
mkdir -p 256k-warp4-lib
cp -L -- build-cu133/bin/libggml-cuda.so.0 256k-warp4-lib/libggml-cuda.so.0
cp -- 256k-baseline-lib/libggml-cuda.so.0 build-cu133/bin/libggml-cuda.so.0.28.0
export LD_LIBRARY_PATH="$HOME/models/sensenova-runtime/256k-warp4-lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib"
unset GGML_CUDA_FUSED_Q4K_J64
export GGML_CUDA_LONG_Q8_CONVERT_WARP4=1
python3 -u "$r/benchmark.py" 262144 "$r/q8-convert-warp4" --frozen-prompt "$r/frozen-prompt.json" --reference "$reference"
nsys=/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
python3 -u "$r/benchmark.py" 262144 "$r/nsys-warp4" --frozen-prompt "$r/frozen-prompt.json" --reps 1 --no-probs --cold-only --reference "$reference" --nsys "$nsys"
"$nsys" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --output "$r/nsys-warp4/summary" "$r/nsys-warp4/trace.nsys-rep"
