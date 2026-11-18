#include "embeddra_runtime.h"
#include "../../backends/custom_cpp/hypergqe_engine.h"
#include "../hardware/hardware_detector.h"
#include <chrono>
#include <iostream>

namespace embeddra {
namespace runtime {

class EmbeddraRuntime::Impl {
public:
    Impl() : is_initialized_(false), stop_flag_(false) {}

    bool initialize() {
        hardware_info_ = hardware::HardwareDetector::detect();
        hardware::HardwareDetector::printDeviceInfo(hardware_info_);
        engine_ = std::make_unique<backend::HyperGQEEngine>();
        is_initialized_ = true;
        return true;
    }

    bool loadModel(const std::string& model_path) {
        if (!is_initialized_) {
            if (!initialize()) return false;
        }
        return engine_->loadModel(model_path);
    }

    void generate(const std::string& prompt, const GenerationConfig& config, TokenCallback callback) {
        (void)prompt;
        std::vector<int> dummy_tokens = {128000};
        generateWithTokens(dummy_tokens, config, callback);
    }

    void generateWithTokens(const std::vector<int>& prompt_tokens, const GenerationConfig& config, TokenCallback callback) {
        if (!engine_ || !engine_->isLoaded()) {
            std::cerr << "[EmbeddraRuntime] Cannot generate: model is not loaded.\n";
            return;
        }

        stop_flag_ = false;
        engine_->resetKVCache();

        auto start_time = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point first_token_time;
        bool has_first_token = false;

        std::vector<int> tokens_to_prefill = prompt_tokens.empty() ? std::vector<int>{128000} : prompt_tokens;
        int current_pos = 0;

        // Prefill prompt (skip expensive 128k LM head until final prefill token)
        const float* logits = nullptr;
        size_t prefill_count = tokens_to_prefill.size();
        for (size_t i = 0; i < prefill_count; ++i) {
            if (stop_flag_) break;
            bool is_last = (i + 1 == prefill_count);
            logits = engine_->forward(tokens_to_prefill[i], current_pos++, is_last);
        }

        // Autoregressive generation
        int tokens_generated = 0;
        int next_token = 0;
        int last_token = -1;
        int prev_last_token = -2;
        int prev_3_token = -3;

        // Calibrated repetition penalty: 1.05 for conversation (temp > 0.05), 1.0 for agent tool actions
        float rep_penalty = (config.temperature > 0.05f) ? 1.05f : 1.0f;
        while (tokens_generated < config.max_tokens && !stop_flag_) {
            if (!logits) break;
            next_token = engine_->sample(logits, config.temperature, config.top_p, config.top_k, rep_penalty);

            // Break degenerate repetition loops (same token repeated 3+ times consecutively)
            if (tokens_generated >= 3 && next_token == last_token && next_token == prev_last_token) {
                break;
            }

            if (!has_first_token) {
                first_token_time = std::chrono::high_resolution_clock::now();
                has_first_token = true;
            }

            tokens_generated++;
            bool is_eos = (next_token == config.eos_token_id || next_token == 128001 || next_token == 128009);

            // Stream token to listener (skip emitting raw EOS token text, just signal is_final)
            if (callback) {
                callback("", next_token, (tokens_generated >= config.max_tokens || is_eos));
            }

            if (is_eos) {
                break;
            }

            prev_3_token = prev_last_token;
            prev_last_token = last_token;
            last_token = next_token;

            // Next forward step
            logits = engine_->forward(next_token, current_pos++);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        last_stats_.tokens_generated = tokens_generated;
        last_stats_.total_generation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        if (has_first_token) {
            last_stats_.time_to_first_token_ms = std::chrono::duration<double, std::milli>(first_token_time - start_time).count();
        }
        if (last_stats_.total_generation_time_ms > 0) {
            last_stats_.tokens_per_second = (tokens_generated * 1000.0) / last_stats_.total_generation_time_ms;
        }
    }

    void stopGeneration() {
        stop_flag_ = true;
    }

    void unloadModel() {
        if (engine_) {
            engine_->unloadModel();
        }
    }

    void shutdown() {
        unloadModel();
        is_initialized_ = false;
    }

    bool isModelLoaded() const {
        return engine_ && engine_->isLoaded();
    }

    GenerationStats getLastStats() const {
        return last_stats_;
    }

    std::string getDeviceBackendInfo() const {
        if (engine_) {
            return engine_->getActiveBackendName();
        }
        return "Uninitialized";
    }

    void setBackendMode(int mode) {
        if (engine_) {
            engine_->setBackendMode(static_cast<backend::HyperGQEEngine::BackendMode>(mode));
        }
    }

private:
    bool is_initialized_;
    std::atomic<bool> stop_flag_;
    hardware::DeviceInfo hardware_info_;
    std::unique_ptr<backend::HyperGQEEngine> engine_;
    GenerationStats last_stats_;
};

EmbeddraRuntime::EmbeddraRuntime() : pimpl_(std::make_unique<Impl>()) {}
EmbeddraRuntime::~EmbeddraRuntime() = default;

bool EmbeddraRuntime::initialize() { return pimpl_->initialize(); }
bool EmbeddraRuntime::loadModel(const std::string& model_path) { return pimpl_->loadModel(model_path); }
void EmbeddraRuntime::generate(const std::string& prompt, int max_tokens, TokenCallback callback) {
    GenerationConfig config;
    config.max_tokens = max_tokens;
    pimpl_->generate(prompt, config, callback);
}
void EmbeddraRuntime::generateWithConfig(const std::string& prompt, const GenerationConfig& config, TokenCallback callback) {
    pimpl_->generate(prompt, config, callback);
}
void EmbeddraRuntime::generateWithTokens(const std::vector<int>& prompt_tokens, const GenerationConfig& config, TokenCallback callback) {
    pimpl_->generateWithTokens(prompt_tokens, config, callback);
}
void EmbeddraRuntime::stopGeneration() { pimpl_->stopGeneration(); }
void EmbeddraRuntime::unloadModel() { pimpl_->unloadModel(); }
void EmbeddraRuntime::shutdown() { pimpl_->shutdown(); }
bool EmbeddraRuntime::isModelLoaded() const { return pimpl_->isModelLoaded(); }
GenerationStats EmbeddraRuntime::getLastStats() const { return pimpl_->getLastStats(); }
std::string EmbeddraRuntime::getDeviceBackendInfo() const { return pimpl_->getDeviceBackendInfo(); }
void EmbeddraRuntime::setBackendMode(int mode) { pimpl_->setBackendMode(mode); }

} // namespace runtime
} // namespace embeddra

