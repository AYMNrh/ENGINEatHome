import csv,sys,collections,re
rows=list(csv.reader(open(sys.argv[1])))
h=rows[1]; data=rows[2:]
iS=h.index("Source"); iW=h.index("Warp Stall Sampling (All Samples)"); iE=h.index("Instructions Executed")
tot_s=sum(int(r[iW] or 0) for r in data); tot_e=sum(int(r[iE] or 0) for r in data)
op=collections.Counter(); ops=collections.Counter()
seg=collections.defaultdict(lambda:[0,0,collections.Counter()]); k=0
for r in data:
    src=r[iS].strip(); m=re.match(r"(@!?U?P\w+\s+)?([A-Z0-9_]+)(\.[\w.]+)?",src)
    o=m.group(2) if m else src[:8]
    e=int(r[iE] or 0); s=int(r[iW] or 0)
    op[o]+=e; ops[o]+=s
    seg[k][0]+=e; seg[k][1]+=s; seg[k][2][o]+=e
    if o=="BAR": k+=1
print("total inst %.1fM  samples %d"%(tot_e/1e6,tot_s))
print("by opcode (inst%, stall%):", ", ".join("%s %.0f/%.0f"%(o,100*op[o]/tot_e,100*ops[o]/tot_s) for o,_ in op.most_common(16)))
print("segments between BAR (inst%, stall%, top ops):")
for k,(e,s,c) in sorted(seg.items()):
    if e*100/tot_e>1 or s*100/tot_s>1: print("  seg%d %.1f%% / %.1f%%  %s"%(k,100*e/tot_e,100*s/tot_s,", ".join("%s:%d"%(o,100*v/e) for o,v in c.most_common(6))))
