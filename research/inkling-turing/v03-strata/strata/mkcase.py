import json,sys
i=int(sys.argv[1]); P=json.load(open("results/div_prompts.json")); off=json.load(open("results/div_off1.json")); on=json.load(open("results/div_on1.json"))
o,n=off[i]["tokens"],on[i]["tokens"]; d=next(k for k,(x,y) in enumerate(zip(o,n)) if x!=y)
seq=P[i]+o[:d]
open("results/case%d.ids"%i,"w").write(",".join(map(str,seq))); json.dump(dict(i=i,d=d,off_tok=o[d],on_tok=n[d],plen=len(P[i]),n=len(seq)),open("results/case%d.json"%i,"w")); print(i,d,o[d],n[d],len(seq))
