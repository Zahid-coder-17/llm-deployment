with open(r'embeddra-edge\backends\custom_cpp\hypergqe_engine.cpp', 'r', encoding='utf-8') as f:
    content = f.read()

# Add include if not present
if '#include <unordered_set>' not in content:
    content = '#include <unordered_set>\n' + content

target = """    // Repetition penalty
    if (rep_penalty != 1.0f && !generated_tokens_.empty()) {
        for (int prev_tok : generated_tokens_) {
            if (prev_tok >= 0 && prev_tok < config_.vocab_size) {
                if (work_logits[prev_tok] > 0.0f) {
                    work_logits[prev_tok] /= rep_penalty;
                } else {
                    work_logits[prev_tok] *= rep_penalty;
                }
            }
        }
    }"""

replacement = """    // Calibrated sliding-window repetition penalty (last 32 tokens, deduplicated)
    if (rep_penalty != 1.0f && !generated_tokens_.empty()) {
        int window_size = 32;
        int start_idx = std::max(0, static_cast<int>(generated_tokens_.size()) - window_size);
        std::unordered_set<int> seen;
        for (size_t i = start_idx; i < generated_tokens_.size(); ++i) {
            int prev_tok = generated_tokens_[i];
            if (prev_tok >= 0 && prev_tok < config_.vocab_size && seen.insert(prev_tok).second) {
                if (work_logits[prev_tok] > 0.0f) {
                    work_logits[prev_tok] /= rep_penalty;
                } else {
                    work_logits[prev_tok] *= rep_penalty;
                }
            }
        }
    }"""

assert target in content, "Target not found in hypergqe_engine.cpp!"
new_content = content.replace(target, replacement)
with open(r'embeddra-edge\backends\custom_cpp\hypergqe_engine.cpp', 'w', encoding='utf-8') as f:
    f.write(new_content)
print("SUCCESSFULLY PATCHED hypergqe_engine.cpp!")
