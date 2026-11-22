import json,subprocess,threading,time,os
SA=os.path.expanduser("~/llm/experiments/strata-ab")
ids=json.load(open(SA+"/prompt_8192.json"))["a"]
p=subprocess.Popen([SA+"/bin13/strata-inkling","--serve","--native",SA+"/pack/Inkling-2-Mini-SQ-15.strata","--max-context","16384","--parallel","1","--draft","dspark","--no-anti-loop"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(SA+"/results/append_server.log","w"),text=True,bufsize=1,env=dict(os.environ,LD_LIBRARY_PATH=SA+"/bin13"))
lines=[]
threading.Thread(target=lambda:[lines.append((time.time(),l.rstrip())) for l in p.stdout],daemon=True).start()
def wait(pred,mark,to=120):
    e=time.time()+to
    while time.time()<e:
        for t,l in lines[mark:]:
            if pred(l): return t,l
        time.sleep(.05)
    return None,"timeout"
wait(lambda s:s.startswith("READY"),0,200)
out=[]
def gen(prompt,tag):
    mark=len(lines); ts=time.time()
    p.stdin.write("BGEN 0 64 slot_read=0 temperature=0 top_k=1 %s\n"%",".join(map(str,prompt))); p.stdin.flush()
    t,l=wait(lambda s:s.startswith("BDONE") or s=="BADM 0 0",mark)
    new=lines[mark:]; toks=[int(x.split()[1]) for _,x in new if x.startswith("T ")]+[int(x.split()[2]) for _,x in new if x.startswith("BT ")]
    first=next((tt for tt,x in new if x.startswith("T ")),None); adm=next((x for _,x in new if x.startswith("DONE ")),"")
    r=dict(tag=tag,prompt_len=len(prompt),ttft_s=round(first-ts,3) if first else None,admit=adm,resume=[x for _,x in new if x.startswith("RESUME")],final=l,n=len(toks)); print(r,flush=True); out.append(r); return toks
t1=gen(ids,"cold8k")
t2=gen(ids+t1[:-1]+list(range(1000,1040)),"append_turn_reuse")   # exact extension of previous prompt+generation
json.dump(out,open(SA+"/results/append.json","w"),indent=1)
p.stdin.write("QUIT\n"); p.stdin.flush()
try: p.wait(20)
except Exception: p.kill()
