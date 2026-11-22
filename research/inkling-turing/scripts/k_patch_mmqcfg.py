import re,sys
p="llama.cpp/ggml/src/ggml-cuda/mmq-config-ampere.cuh"
s=open(p).read()
types=["NVFP4"]
n=0
for t in types:
    for J in (24,32,40,48,64):
        pat=re.compile(r"CASE\(GGML_TYPE_%s, 256, 1, 128, +%d, (\w+), MMQ_ITER_K, true, false\);"%(t,J))
        m=pat.findall(s)
        assert len(m)==1,(t,J,m)
        s=pat.sub(lambda mm:"CASE(GGML_TYPE_%s, 128, 2,  64, %3d, %s, MMQ_ITER_K, true, false); // inkling-turing: 2 blocks/SM"%(t,J,mm.group(1)),s)
        n+=1
open(p,"w").write(s); print("patched",n)
