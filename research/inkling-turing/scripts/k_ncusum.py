import csv, sys, glob, io, re
KEEP = ['gpu__time_duration.sum','dram__throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','launch__registers_per_thread','launch__grid_size','launch__block_size','launch__shared_mem_per_block_dynamic','launch__shared_mem_per_block_static','launch__occupancy_limit_registers','launch__occupancy_limit_shared_mem','smsp__issue_active.avg.pct_of_peak_sustained_active','sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active','l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum','l1tex__data_pipe_lsu_wavefronts_mem_shared.sum','smsp__inst_executed.sum','l1tex__t_sector_hit_rate.pct']
for f in sorted(glob.glob(sys.argv[1])):
    txt = open(f, errors='ignore').read()
    i = txt.find('"ID"')
    if i < 0:
        print(f.split('/')[-1], 'NO DATA'); continue
    rows = list(csv.reader(io.StringIO(txt[i:])))
    hdr, units, data = rows[0], rows[1], rows[2:]
    d = data[-1]                                   # second profiled instance (past warm-up)
    print('=====', f.split('/')[-1], re.sub(r'\(const.*', '', d[hdr.index('Kernel Name')])[:90])
    vals = {k: d[hdr.index(k)] for k in hdr}
    out = []
    for k in KEEP:
        if k in vals: out.append(f'{k.split(".")[0].replace("launch__","").replace("smsp__","").replace("l1tex__","")[:34]}={vals[k]}{units[hdr.index(k)] if units[hdr.index(k)] in ("%","usecond","msecond","nsecond") else ""}')
    st = sorted(((float(vals[k].replace(',', '')), k) for k in hdr if 'warp_issue_stalled' in k and vals[k].replace(',', '').replace('.', '').isdigit()), reverse=True)[:5]
    print('   ', ' | '.join(out))
    print('    top stalls:', ', '.join(f'{k.split("stalled_")[1].replace("_per_warp_active.pct","")}={v:.1f}%' for v, k in st))
