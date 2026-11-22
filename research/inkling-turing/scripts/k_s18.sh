# SASS-level source counters for the J=48 MoE MMQ kernels (bin-iqB)
R=~/llm/experiments/inkling-kernels
B=$R/bin-iqB
M=$R/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
for T in 40 18 22; do
  sudo -n env LD_LIBRARY_PATH=$B GGML_CUDA_DISABLE_GRAPHS=1 ncu --kernel-name-base demangled -k "regex:mul_mat_q<\(ggml_type\)$T, \(int\)48" -s 3 -c 1 --section SourceCounters --section WarpStateStats --section LaunchStats --section Occupancy --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis -f -o $R/prof/src_iqB_$T $B/llama-bench -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512 -p 512 -n 0 -d 0 -r 1 -o md > /dev/null 2>$R/prof/src_iqB_$T.err
  sudo -n ncu -i $R/prof/src_iqB_$T.ncu-rep --page source --print-source sass --csv > $R/prof/src_iqB_$T.sass.csv 2>/dev/null
  sudo -n ncu -i $R/prof/src_iqB_$T.ncu-rep --page details > $R/prof/src_iqB_$T.details.txt 2>/dev/null
  echo "t$T: $(wc -l < $R/prof/src_iqB_$T.sass.csv) sass lines; $(grep -m1 "Duration" $R/prof/src_iqB_$T.details.txt)"
done
sudo -n chown "$USER" $R/prof/src_iqB_* 
