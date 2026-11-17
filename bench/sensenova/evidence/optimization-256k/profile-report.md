# SenseNova Loop 9B Q4_K_M at 256K

RTX 5070 Ti, WSL2, CUDA 13.3, llama.cpp fork fad174e with the original CUDA mHC port. This Nsight Systems trace covers a warmup and a filled 261,632-token cold request with 256 greedy outputs. Q8 K and V, microbatch 1,024, and CUDA Graphs were active. The traced output matched all 256 frozen token IDs. Profiled scheduling changes timing. Compare unprofiled request latencies in the benchmark JSON files.

| Kernel | GPU time | Calls | Representative grid / block | Registers per thread | Resource-derived occupancy ceiling |
|---|---:|---:|---|---:|---:|
| FP16 MMA FlashAttention, head size 256 | 170.050 s | 3,612 | 140×1×1 / 32×4×1 | 246 | 16.7% |
| Fused Q4_K MMQ, J=128 | 24.688 s | 14,190 | 8×192×1 / 32×8×1 | 254 | 16.7% |
| Dense Q4_K MMQ, J=128 | 21.666 s | 48,504 | 256×1×1 / 32×8×1 | 251 | 16.7% |
| Dense Q6_K MMQ, J=128 | 12.028 s | 12,642 | 256×1×1 / 32×8×1 | 255 | 16.7% |
| Q8-to-FP16 KV conversion | 5.405 s | 7,280 | 130,816×1×1 / 32×1×1 | 39 | 50.0% |

The occupancy column bounds residency from 1,536 threads, 65,536 registers, 100 KiB shared memory and 24 CTAs per SM. Achieved occupancy and per-kernel DRAM bandwidth are unavailable. Nsight Compute returned ERR_NVGPUCTRPERM on the Windows host.

The FFN fused case uses 4,096 input channels, 12,288 output channels, and 1,024 query tokens per full microbatch. A J=128 CTA covers 64 output channels and 128 token columns. Prefill uses Q4_K/Q6_K MMQ with Q8_1 activations. Decode uses MMVQ. No GEMM kernels attributed to cuBLAS appeared in this trace. The baseline recorded 2,272 CUDA Graph launch calls and 16 graph instantiations. FlashAttention and its helpers consumed 177.255 summed GPU kernel seconds. DeltaNet and mHC consumed 3.883 and 6.881 seconds respectively.

[Raw top five and launch shapes](nsys-top5.csv)
