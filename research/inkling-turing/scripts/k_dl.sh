cd ~/llm/experiments/inkling-kernels
mkdir -p model
~/llm/hf-venv/bin/hf download unsloth/Inkling-2-Mini-GGUF Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file --local-dir ./model > dl.log 2>&1
F=model/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file
sha256sum $F > model/mainline.sha256
ln -sf Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf
echo DL_DONE >> dl.log
