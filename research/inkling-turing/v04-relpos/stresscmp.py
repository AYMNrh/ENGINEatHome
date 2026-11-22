import json,sys
a,b=sys.argv[1],sys.argv[2]
A=json.load(open("stress_%s.json"%a))["results"]; B=json.load(open("stress_%s.json"%b))["results"]
bad=0
for x,y in zip(A,B):
    if x.get("tokens")!=y.get("tokens") or x.get("lp")!=y.get("lp"): bad+=1; print("  DIFF req",x["i"],x["n_prompt"],x.get("multi"))
print(a,"vs",b,":",len(A),"requests,",bad,"differ; prefix-reuse requests:",sum(1 for r in B if r.get("cache_n")), "multi-turn:",sum(1 for r in B if r.get("multi")))
