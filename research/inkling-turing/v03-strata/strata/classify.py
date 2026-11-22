import json,struct,numpy as np
R="results"
def last_row(p):
    with open(p,"rb") as f:
        n,V=struct.unpack("<ii",f.read(8)); f.seek(8+4*n+4*(n-1)*V); return np.frombuffer(f.read(4*V),dtype=np.float32)
rows=[]
for i in range(12):
    m=json.load(open("%s/case%d.json"%(R,i))) if __import__("os").path.exists("%s/case%d.json"%(R,i)) else None
    if m is None: 
        import subprocess; subprocess.run(["./venv/bin/python","mkcase.py",str(i)],capture_output=True); m=json.load(open("%s/case%d.json"%(R,i)))
    g=last_row("%s/ref_gpu_%d.bin"%(R,i)); a,b=m["off_tok"],m["on_tok"]
    top=int(np.argmax(g)); gap=float(g[a]-g[b]); rank_on=int((g>g[b]).sum())
    rows.append((i,m["d"],a,b,round(gap,3),top==a,rank_on,round(float(np.sort(g)[-1]-np.sort(g)[-2]),3)))
    print("case %2d d=%2d off=%d on=%d  logit(off)-logit(on)=%8.3f  M1-token-is-argmax=%s  rank of spec-token=%d  top1-top2=%.3f"%rows[-1])
gaps=[r[4] for r in rows]
print("spec token within 1.5 logits of M=1 token (near-tie):",sum(abs(g)<1.5 for g in gaps),"| 1.5-5:",sum(1.5<=abs(g)<5 for g in gaps),"| >5 (genuine):",sum(abs(g)>=5 for g in gaps),"| spec token better than M1 token:",sum(g<0 for g in gaps))
