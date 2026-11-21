p = '/home/marvin/llm/experiments/inkling-kernels/llama.cpp/tests/test-backend-ops.cpp'
s = open(p).read()
if 'test_flash_attn_ext_ink' not in s:
    cls = '''
// turing-iq3-opt: Inkling 2 decode attention shape (8 KV heads x GQA 3, per-head relative-bias masks, sinks, q8_0 KV);
// INK_FA_NMSE overrides the tolerance so that kernels can be bracketed against the CPU reference
struct test_flash_attn_ext_ink : public test_flash_attn_ext {
    using test_flash_attn_ext::test_flash_attn_ext;
    double max_nmse_err() override {
        const char * env = getenv("INK_FA_NMSE");
        return env ? atof(env) : 5e-4;
    }
    std::string vars() override {
        return test_flash_attn_ext::vars() + ",ink=1";
    }
};
'''
    anchor = '// mask with 2 heads and 2 batches, broadcast over the Q heads and batches'
    assert s.count(anchor) == 1
    s = s.replace(anchor, cls + anchor, 1)
    reg = '    test_cases.emplace_back(new test_flash_attn_ext(72, 64, 4, {1, 1}, 256, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));'
    assert s.count(reg) == 1
    add = reg + '''
    for (int64_t kv : {4096, 8192, 32768}) {
        test_cases.emplace_back(new test_flash_attn_ext_ink(128, 128, 8, {3, 1}, kv, 1, true, true, 0, 0, GGML_PREC_F32,
            GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 1, 2, 3}, true, false, 0, 0, 24));
    }'''
    s = s.replace(reg, add)
    open(p, 'w').write(s)
print('patched test')
