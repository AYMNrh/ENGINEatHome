import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

src, work, out = map(Path, sys.argv[1:4])
variant = sys.argv[4]
cache = src / "build-cu133/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir"
flags = dict(line.split(" = ", 1) for line in (cache / "flags.make").read_text().splitlines()
             if " = " in line)
includes = shlex.split((cache / "includes_CUDA.rsp").read_text().replace(str(src), str(work)))
unit = work / "ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu"
command = ["/usr/local/cuda-13.3/bin/nvcc", "-forward-unknown-to-host-compiler",
           *shlex.split(flags["CUDA_DEFINES"]), *includes, *shlex.split(flags["CUDA_FLAGS"]),
           "-Xptxas=-v", "-x", "cu", "-c", str(unit), "-o", str(out / (variant + ".o"))]
if variant == "original":
    (out / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
else:
    original = json.loads((out / "compile-command.json").read_text())
    assert original[:-1] == command[:-1]
with (out / (variant + "-ptxas.log")).open("w") as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=work)
