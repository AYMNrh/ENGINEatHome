import sys; sys.path.insert(0,"~/llm/experiments/inkling-kernels/pilot")
import pilot_long as L, json
from tokenizers import Tokenizer
tk=Tokenizer.from_file("pack/tokenizer.json")
files=L.build(7,260,"7340")
text=L.dump(files)
ids=tk.encode(text).ids
print("total ids",len(ids),"chars",len(text))
tail=tk.encode("\n\n# Task: explain in plain English what the first function in settings.py and the scheduler do, then list three risks.\n").ids
other=tk.encode("\n\n# Task: write a unit test for weighted_mean in reports/aggregate.py covering empty input and zero weights.\n").ids
for n in (8192,32768,131072):
    if len(ids)<n: print("too short",n); continue
    a=ids[:n-len(tail)]+tail
    b=ids[:n-len(tail)]+other[:len(tail)] if len(other)>=len(tail) else ids[:n-len(other)]+other
    json.dump(dict(a=a,b=b,n=n,shared=n-len(tail)),open("prompt_%d.json"%n,"w"))
    print(n,len(a),len(b),"shared prefix",n-len(tail))
