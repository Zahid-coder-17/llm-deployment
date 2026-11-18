#include <unordered_set>
#include "hypergqe_engine.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_EER(...) __android_log_print(ANDROID_LOG_INFO, "HyperGQEEngine", __VA_ARGS__)
#else
#define LOG_EER(...)
#endif

namespace embeddra {
namespace backend {

static inline float silu(float x) {
    if (x <= -12.0f) return 0.0f;
    if (x >= 12.0f) return x;
    return x / (1.0f + std::exp(-x));
}

// Fast IEEE 754 half-precision (FP16) to single-precision (FP32) conversion
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;

    if (exp == 0x1F) { // Inf/NaN
        exp = 0xFF;
    } else if (exp == 0) { // Subnormal or Zero
        if (mant != 0) {
            // Normalize
            exp = 0x70 + 1;
            while ((mant & 0x00800000) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x007FFFFF;
        }
    } else {
        exp = exp + (127 - 15);
    }

    uint32_t u = sign | (exp << 23) | mant;
    float result;
    std::memcpy(&result, &u, sizeof(float));
    return result;
}

HyperGQEEngine::HyperGQEEngine(const EngineConfig& config)
    : config_(config), is_loaded_(false), fd_(-1), mmap_base_(nullptr), mmap_size_(0) {
    initBuffers();
}

HyperGQEEngine::~HyperGQEEngine() {
    unloadModel();
}

void HyperGQEEngine::initBuffers() {
    hidden_state_.resize(config_.hidden_size, 0.0f);
    norm_state_.resize(config_.hidden_size, 0.0f);
    q_state_.resize(config_.hidden_size, 0.0f);
    k_state_.resize(config_.num_kv_heads * config_.head_dim, 0.0f);
    v_state_.resize(config_.num_kv_heads * config_.head_dim, 0.0f);
    router_logits_.resize(16, 0.0f);
    attn_out_.resize(config_.hidden_size, 0.0f);
    gate_state_.resize(config_.intermediate_size, 0.0f);
    up_state_.resize(config_.intermediate_size, 0.0f);
    ffn_act_.resize(config_.intermediate_size, 0.0f);
    ffn_out_.resize(config_.hidden_size, 0.0f);
    logits_.resize(config_.vocab_size, 0.0f);
    q_group_buf_.resize(config_.num_kv_heads * config_.head_dim, 0.0f);
    q_aux_buf_.resize(2 * config_.head_dim, 0.0f);
    cat_heads_buf_.resize(768, 0.0f);
    attn_scores_buf_.resize(config_.max_seq_len, 0.0f);
    w_d_buf_.resize(config_.hidden_size * 16, 0.0f);
    w_u_buf_.resize(16 * config_.head_dim, 0.0f);

    mem_pool_ = std::make_unique<memory::MemoryPool>(64 * 1024 * 1024);
    kv_cache_ = std::make_unique<memory::KVCacheBuffer>(
        config_.num_layers,
        config_.num_kv_heads,
        config_.head_dim,
        config_.max_seq_len
    );
    layers_.resize(config_.num_layers);

    gpu_engine_ = std::make_unique<GPUEngine>();
    if (gpu_engine_->initialize()) {
        LOG_EER("HyperGQEEngine: Initialized GPU Acceleration [%s]", gpu_engine_->getDeviceName().c_str());
    } else {
        LOG_EER("HyperGQEEngine: GPU not available, falling back to CPU execution.");
    }
    // Use high-throughput persistent work-stealing ARM NEON pool by default to avoid OpenCL sync stalls
    backend_mode_ = BACKEND_CPU_8CORE;
}

bool HyperGQEEngine::isGPUActive() const {
    return gpu_engine_ && gpu_engine_->isSupported();
}

std::string HyperGQEEngine::getGPUBackendName() const {
    if (isGPUActive()) {
        return gpu_engine_->getDeviceName();
    }
    return "CPU (Fallback)";
}

void HyperGQEEngine::setBackendMode(BackendMode mode) {
    backend_mode_ = mode;
}

HyperGQEEngine::BackendMode HyperGQEEngine::getBackendMode() const {
    return backend_mode_;
}

std::string HyperGQEEngine::getActiveBackendName() const {
    if (backend_mode_ == BACKEND_GPU && isGPUActive()) {
        return "GPU: " + gpu_engine_->getDeviceName();
    } else if (backend_mode_ == BACKEND_CPU_8CORE) {
        return "CPU: 8-Core ARM NEON (4x Cortex-A78 + 4x Cortex-A55)";
    } else if (backend_mode_ == BACKEND_HYBRID) {
        return "Hybrid: Mali-G68 GPU + 8-Core ARM NEON";
    }
    return "CPU (Scalar Fallback)";
}

void HyperGQEEngine::resetKVCache() {
    if (kv_cache_) {
        kv_cache_->reset();
    }
    generated_tokens_.clear();
}

bool HyperGQEEngine::loadModel(const std::string& model_path) {
    std::cout << "[HyperGQEEngine] Attempting to load model from: " << model_path << "\n";

    // Auto-detect matching .eer file if given .pt path
    std::string path_to_open = model_path;
    if (model_path.find(".pt") != std::string::npos) {
        std::string eer_candidate = model_path.substr(0, model_path.find_last_of('.')) + ".eer";
        std::ifstream test_f(eer_candidate);
        if (test_f.good()) {
            path_to_open = eer_candidate;
        } else {
            // Also check standard location
            std::string alt = "/sdcard/Android/data/com.embeddra.edge/files/maria_1b.eer";
            std::ifstream test_alt(alt);
            if (test_alt.good()) path_to_open = alt;
        }
    }
    LOG_EER("HyperGQEEngine: Loading model from %s", path_to_open.c_str());

#if defined(__linux__) || defined(__ANDROID__)
    fd_ = open(path_to_open.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::cerr << "[HyperGQEEngine] Cannot open file: " << path_to_open << "\n";
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    mmap_size_ = sb.st_size;

    mmap_base_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_base_ == MAP_FAILED) {
        std::cerr << "[HyperGQEEngine] mmap failed for size: " << mmap_size_ << "\n";
        close(fd_);
        fd_ = -1;
        return false;
    }
#else
    std::ifstream f(path_to_open, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    mmap_size_ = f.tellg();
    f.seekg(0, std::ios::beg);
    mmap_base_ = malloc(mmap_size_);
    if (!mmap_base_) return false;
    f.read(static_cast<char*>(mmap_base_), mmap_size_);
#endif

    const char* base = static_cast<const char*>(mmap_base_);

    // Check Header
    if (std::memcmp(base, "EER1", 4) != 0) {
        std::cerr << "[HyperGQEEngine] Invalid magic header. Expected EER1.\n";
        unloadModel();
        return false;
    }

    uint32_t num_tensors = *reinterpret_cast<const uint32_t*>(base + 44);
    uint32_t index_size = *reinterpret_cast<const uint32_t*>(base + 48);

    const char* index_ptr = base + 52;
    const char* index_end = index_ptr + index_size;

    tensor_map_.clear();
    while (index_ptr < index_end) {
        uint16_t name_len = *reinterpret_cast<const uint16_t*>(index_ptr);
        uint8_t num_dims = *reinterpret_cast<const uint8_t*>(index_ptr + 2);
        index_ptr += 3;

        std::string name(index_ptr, name_len);
        index_ptr += name_len;

        std::vector<int> shape(num_dims);
        for (int d = 0; d < num_dims; ++d) {
            shape[d] = static_cast<int>(*reinterpret_cast<const uint32_t*>(index_ptr));
            index_ptr += 4;
        }

        uint8_t dtype = *reinterpret_cast<const uint8_t*>(index_ptr);
        uint64_t offset = *reinterpret_cast<const uint64_t*>(index_ptr + 1);
        uint64_t size_bytes = *reinterpret_cast<const uint64_t*>(index_ptr + 9);
        index_ptr += 17;

        TensorRef ref;
        ref.name = name;
        ref.shape = shape;
        ref.dtype = dtype;
        ref.offset = offset;
        ref.size_bytes = size_bytes;
        ref.data_ptr = static_cast<const void*>(base + offset);

        tensor_map_[name] = ref;
    }

    std::cout << "[HyperGQEEngine] Parsed " << tensor_map_.size() << " tensors from .eer index.\n";
    LOG_EER("Parsed %zu tensors from .eer index", tensor_map_.size());

    // Bind Core Pointers
    if (tensor_map_.count("embed.weight")) {
        embed_weight_ = static_cast<const uint16_t*>(tensor_map_["embed.weight"].data_ptr);
    }
    if (tensor_map_.count("norm.weight")) {
        final_norm_weight_ = static_cast<const uint16_t*>(tensor_map_["norm.weight"].data_ptr);
    }

    // Bind Per-layer Pointers
    int bound_count = 0;
    for (int l = 0; l < config_.num_layers; ++l) {
        std::string p_blocks = "blocks." + std::to_string(l) + ".";
        std::string p_layers = "layers." + std::to_string(l) + ".";
        auto& lp = layers_[l];

        auto get_ptr = [&](const std::string& suffix) -> const void* {
            if (tensor_map_.count(p_blocks + suffix)) return tensor_map_[p_blocks + suffix].data_ptr;
            if (tensor_map_.count(p_layers + suffix)) return tensor_map_[p_layers + suffix].data_ptr;
            return nullptr;
        };

        lp.attn_norm_w = static_cast<const uint16_t*>(get_ptr("attn_norm.weight"));
        lp.q_proj_w = static_cast<const uint16_t*>(get_ptr("attn.q_proj.weight"));
        lp.k_proj_w = static_cast<const uint16_t*>(get_ptr("attn.k_proj.weight"));
        lp.v_proj_w = static_cast<const uint16_t*>(get_ptr("attn.v_proj.weight"));
        lp.router_w = static_cast<const uint16_t*>(get_ptr("attn.router.weight"));
        lp.expert_emb = static_cast<const uint16_t*>(get_ptr("attn.expert_embeddings"));
        lp.sel_mlp_0_w = static_cast<const uint16_t*>(get_ptr("attn.selection_mlp.0.weight"));
        lp.sel_mlp_0_b = static_cast<const uint16_t*>(get_ptr("attn.selection_mlp.0.bias"));
        lp.sel_mlp_2_w = static_cast<const uint16_t*>(get_ptr("attn.selection_mlp.2.weight"));
        lp.sel_mlp_2_b = static_cast<const uint16_t*>(get_ptr("attn.selection_mlp.2.bias"));
        lp.hyper_w_d = static_cast<const uint16_t*>(get_ptr("attn.hyper_W_D.weight"));
        lp.hyper_w_d_b = static_cast<const uint16_t*>(get_ptr("attn.hyper_W_D.bias"));
        lp.hyper_w_u = static_cast<const uint16_t*>(get_ptr("attn.hyper_W_U.weight"));
        lp.hyper_w_u_b = static_cast<const uint16_t*>(get_ptr("attn.hyper_W_U.bias"));
        lp.q_shared_w = static_cast<const uint16_t*>(get_ptr("attn.q_shared.weight"));
        lp.o_proj_w = static_cast<const uint16_t*>(get_ptr("attn.o_proj.weight"));

        lp.ffn_norm_w = static_cast<const uint16_t*>(get_ptr("ffn_norm.weight"));

        const void* gate = get_ptr("ffn.gate.weight");
        if (!gate) gate = get_ptr("ffn.gate_proj.weight");
        lp.gate_proj_w = static_cast<const uint16_t*>(gate);

        const void* up = get_ptr("ffn.up.weight");
        if (!up) up = get_ptr("ffn.up_proj.weight");
        lp.up_proj_w = static_cast<const uint16_t*>(up);

        const void* down = get_ptr("ffn.down.weight");
        if (!down) down = get_ptr("ffn.down_proj.weight");
        lp.down_proj_w = static_cast<const uint16_t*>(down);

        if (lp.q_proj_w && lp.k_proj_w && lp.v_proj_w && lp.gate_proj_w && lp.down_proj_w) {
            bound_count++;
        }
    }

    std::cout << "[HyperGQEEngine] All " << bound_count << " of " << config_.num_layers << " HyperGQE layers bound to physical memory.\n";
    LOG_EER("All %d of %d HyperGQE layers bound to physical memory", bound_count, config_.num_layers);
    is_loaded_ = true;
    return true;
}

void HyperGQEEngine::unloadModel() {
    if (mmap_base_) {
#if defined(__linux__) || defined(__ANDROID__)
        munmap(mmap_base_, mmap_size_);
#else
        free(mmap_base_);
#endif
        mmap_base_ = nullptr;
    }
    if (fd_ >= 0) {
#if defined(__linux__) || defined(__ANDROID__)
        close(fd_);
#endif
        fd_ = -1;
    }
    tensor_map_.clear();
    is_loaded_ = false;
}

void HyperGQEEngine::computeRMSNorm(float* out, const float* in, const uint16_t* weight, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_sq += in[i] * in[i];
    }
    float variance = sum_sq / size;
    float inv_std = 1.0f / std::sqrt(variance + eps);
    for (int i = 0; i < size; ++i) {
        float w = weight ? fp16_to_fp32(weight[i]) : 1.0f;
        out[i] = in[i] * inv_std * w;
    }
}

void HyperGQEEngine::computeRoPE(float* q, float* k, int pos, int num_q_heads, int num_kv_heads, int dim) {
    for (int h = 0; h < num_q_heads; ++h) {
        float* q_head = q + h * dim;
        for (int i = 0; i < dim / 2; ++i) {
            float freq = 1.0f / std::pow(config_.rope_theta, (2.0f * i) / dim);
            float angle = pos * freq;
            float cos_v = std::cos(angle);
            float sin_v = std::sin(angle);

            float q0 = q_head[i];
            float q1 = q_head[i + dim / 2];
            q_head[i] = q0 * cos_v - q1 * sin_v;
            q_head[i + dim / 2] = q0 * sin_v + q1 * cos_v;
        }
    }

    if (k && num_kv_heads > 0) {
        for (int h = 0; h < num_kv_heads; ++h) {
            float* k_head = k + h * dim;
            for (int i = 0; i < dim / 2; ++i) {
                float freq = 1.0f / std::pow(config_.rope_theta, (2.0f * i) / dim);
                float angle = pos * freq;
                float cos_v = std::cos(angle);
                float sin_v = std::sin(angle);

                float k0 = k_head[i];
                float k1 = k_head[i + dim / 2];
                k_head[i] = k0 * cos_v - k1 * sin_v;
                k_head[i + dim / 2] = k0 * sin_v + k1 * cos_v;
            }
        }
    }
}

static void gemv_neon_rows(float* out, const uint16_t* weight, const float* in, int start_r, int end_r, int N) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    int r = start_r;
    int end_r2 = start_r + ((end_r - start_r) & ~1);
    for (; r < end_r2; r += 2) {
        const uint16_t* row0 = weight + (size_t)r * N;
        const uint16_t* row1 = weight + (size_t)(r + 1) * N;

        float32x4_t sum0_r0 = vdupq_n_f32(0.0f);
        float32x4_t sum1_r0 = vdupq_n_f32(0.0f);
        float32x4_t sum0_r1 = vdupq_n_f32(0.0f);
        float32x4_t sum1_r1 = vdupq_n_f32(0.0f);

        int c = 0;
        int N8 = N & ~7;
        for (; c < N8; c += 8) {
            __builtin_prefetch(row0 + c + 64, 0, 0);
            __builtin_prefetch(row1 + c + 64, 0, 0);

            float16x8_t w16_0 = vld1q_f16(reinterpret_cast<const float16_t*>(row0 + c));
            float16x8_t w16_1 = vld1q_f16(reinterpret_cast<const float16_t*>(row1 + c));

            float32x4_t w32_lo_0 = vcvt_f32_f16(vget_low_f16(w16_0));
            float32x4_t w32_hi_0 = vcvt_f32_f16(vget_high_f16(w16_0));
            float32x4_t w32_lo_1 = vcvt_f32_f16(vget_low_f16(w16_1));
            float32x4_t w32_hi_1 = vcvt_f32_f16(vget_high_f16(w16_1));

            float32x4_t in_lo = vld1q_f32(in + c);
            float32x4_t in_hi = vld1q_f32(in + c + 4);

            sum0_r0 = vmlaq_f32(sum0_r0, w32_lo_0, in_lo);
            sum1_r0 = vmlaq_f32(sum1_r0, w32_hi_0, in_hi);

            sum0_r1 = vmlaq_f32(sum0_r1, w32_lo_1, in_lo);
            sum1_r1 = vmlaq_f32(sum1_r1, w32_hi_1, in_hi);
        }

        float32x4_t total_vec0 = vaddq_f32(sum0_r0, sum1_r0);
        float32x4_t total_vec1 = vaddq_f32(sum0_r1, sum1_r1);
        float sum_r0 = vaddvq_f32(total_vec0);
        float sum_r1 = vaddvq_f32(total_vec1);

        for (; c < N; ++c) {
            float in_c = in[c];
            sum_r0 += fp16_to_fp32(row0[c]) * in_c;
            sum_r1 += fp16_to_fp32(row1[c]) * in_c;
        }
        out[r] = sum_r0;
        out[r + 1] = sum_r1;
    }

    for (; r < end_r; ++r) {
        const uint16_t* row = weight + (size_t)r * N;
        float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
        float32x4_t sum_vec1 = vdupq_n_f32(0.0f);

        int c = 0;
        int N8 = N & ~7;
        for (; c < N8; c += 8) {
            float16x8_t w16 = vld1q_f16(reinterpret_cast<const float16_t*>(row + c));
            float32x4_t w32_lo = vcvt_f32_f16(vget_low_f16(w16));
            float32x4_t w32_hi = vcvt_f32_f16(vget_high_f16(w16));

            float32x4_t in_lo = vld1q_f32(in + c);
            float32x4_t in_hi = vld1q_f32(in + c + 4);

            sum_vec0 = vmlaq_f32(sum_vec0, w32_lo, in_lo);
            sum_vec1 = vmlaq_f32(sum_vec1, w32_hi, in_hi);
        }

        float32x4_t total_vec = vaddq_f32(sum_vec0, sum_vec1);
        float sum = vaddvq_f32(total_vec);

        for (; c < N; ++c) {
            sum += fp16_to_fp32(row[c]) * in[c];
        }
        out[r] = sum;
    }
#else
    for (int r = start_r; r < end_r; ++r) {
        const uint16_t* row = weight + (size_t)r * N;
        float sum = 0.0f;
        for (int c = 0; c < N; ++c) {
            sum += fp16_to_fp32(row[c]) * in[c];
        }
        out[r] = sum;
    }
#endif
}

class PersistentGemvPool {
public:
    static PersistentGemvPool& instance() {
        static PersistentGemvPool pool(4);
        return pool;
    }

    void run(float* out, const uint16_t* weight, const float* in, int M, int N) {
        if (M < 256) {
            gemv_neon_rows(out, weight, in, 0, M, N);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            out_ = out;
            weight_ = weight;
            in_ = in;
            M_ = M;
            N_ = N;
            active_workers_ = num_threads_ - 1;
            generation_++;
        }
        cv_start_.notify_all();

        // Main thread executes slice 0 directly (zero overhead)
        int rows_per_worker = (M + num_threads_ - 1) / num_threads_;
        int end0 = std::min(rows_per_worker, M);
        gemv_neon_rows(out, weight, in, 0, end0, N);

        std::unique_lock<std::mutex> lock(mtx_);
        cv_done_.wait(lock, [this]() { return active_workers_ == 0; });
    }

private:
    explicit PersistentGemvPool(int num_workers) : num_threads_(num_workers) {
        workers_.reserve(num_threads_ - 1);
        for (int i = 1; i < num_threads_; ++i) {
            workers_.emplace_back([this, i]() { workerLoop(i); });
        }
    }

    ~PersistentGemvPool() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
            generation_++;
        }
        cv_start_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void workerLoop(int thread_idx) {
        uint64_t my_gen = 0;
        while (true) {
            float* my_out = nullptr;
            const uint16_t* my_weight = nullptr;
            const float* my_in = nullptr;
            int my_M = 0;
            int my_N = 0;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_start_.wait(lock, [this, my_gen]() { return stop_ || generation_ != my_gen; });
                if (stop_) return;
                my_gen = generation_;
                my_out = out_;
                my_weight = weight_;
                my_in = in_;
                my_M = M_;
                my_N = N_;
            }

            int rows_per_worker = (my_M + num_threads_ - 1) / num_threads_;
            int start = thread_idx * rows_per_worker;
            int end = std::min(start + rows_per_worker, my_M);
            if (start < my_M) {
                gemv_neon_rows(my_out, my_weight, my_in, start, end, my_N);
            }

            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (--active_workers_ == 0) {
                    cv_done_.notify_one();
                }
            }
        }
    }

    int num_threads_;
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    uint64_t generation_ = 0;
    int active_workers_ = 0;
    float* out_ = nullptr;
    const uint16_t* weight_ = nullptr;
    const float* in_ = nullptr;
    int M_ = 0;
    int N_ = 0;
};

// Matrix-Vector Multiplication: out [M] = W [M, N] * in [N]
void HyperGQEEngine::gemv_fp16(float* out, const uint16_t* weight, const float* in, int M, int N) {
    if (!weight) {
        std::memset(out, 0, M * sizeof(float));
        return;
    }

    if (backend_mode_ == BACKEND_GPU) {
        if (gpu_engine_ && gpu_engine_->isSupported()) {
            gpu_engine_->gemv_fp16(out, weight, in, M, N);
            return;
        }
    } else if (backend_mode_ == BACKEND_HYBRID) {
        if (gpu_engine_ && gpu_engine_->isSupported() && (static_cast<size_t>(M) * N >= 500000)) {
            gpu_engine_->gemv_fp16(out, weight, in, M, N);
            return;
        }
    }

    // Persistent work-stealing ARM NEON execution (zero thread spawn overhead)
    PersistentGemvPool::instance().run(out, weight, in, M, N);
}

const float* HyperGQEEngine::forward(int token_id, int pos, bool compute_logits) {
    if (!is_loaded_) return nullptr;

    // 1. Embedding lookup
    if (embed_weight_ && token_id >= 0 && token_id < config_.vocab_size) {
        const uint16_t* embed_row = embed_weight_ + token_id * config_.hidden_size;
        for (int i = 0; i < config_.hidden_size; ++i) {
            hidden_state_[i] = fp16_to_fp32(embed_row[i]);
        }
    } else {
        std::fill(hidden_state_.begin(), hidden_state_.end(), 0.01f);
    }

    // 2. Transformer Decoder Layers
    for (int l = 0; l < config_.num_layers; ++l) {
        const auto& lp = layers_[l];

        // Attention Pre-RMSNorm
        computeRMSNorm(norm_state_.data(), hidden_state_.data(), lp.attn_norm_w, config_.hidden_size, config_.rms_norm_eps);

        // Q, K, V Projections
        gemv_fp16(q_state_.data(), lp.q_proj_w, norm_state_.data(), config_.hidden_size, config_.hidden_size);
        gemv_fp16(k_state_.data(), lp.k_proj_w, norm_state_.data(), config_.num_kv_heads * config_.head_dim, config_.hidden_size);
        gemv_fp16(v_state_.data(), lp.v_proj_w, norm_state_.data(), config_.num_kv_heads * config_.head_dim, config_.hidden_size);

        // 1. Router logits (16) = router_w * norm_state_
        if (lp.router_w) {
            gemv_fp16(router_logits_.data(), lp.router_w, norm_state_.data(), 16, config_.hidden_size);
        } else {
            std::fill(router_logits_.begin(), router_logits_.end(), 0.0f);
        }

        // 2. Blend Q experts per group (4 groups, 4 experts each)
        std::fill(q_group_buf_.begin(), q_group_buf_.end(), 0.0f);
        float rw[16] = {0.0f};
        for (int g = 0; g < config_.num_kv_heads; ++g) {
            float max_r = router_logits_[g * 4];
            for (int e = 1; e < 4; ++e) {
                if (router_logits_[g * 4 + e] > max_r) max_r = router_logits_[g * 4 + e];
            }
            float sum_r = 0.0f;
            float r_exp[4];
            for (int e = 0; e < 4; ++e) {
                r_exp[e] = std::exp(router_logits_[g * 4 + e] - max_r);
                sum_r += r_exp[e];
            }
            for (int e = 0; e < 4; ++e) {
                float weight = r_exp[e] / sum_r;
                rw[g * 4 + e] = weight;
                const float* q_expert = q_state_.data() + (g * 4 + e) * config_.head_dim;
                float* q_dest = q_group_buf_.data() + g * config_.head_dim;
                for (int d = 0; d < config_.head_dim; ++d) {
                    q_dest[d] += weight * q_expert[d];
                }
            }
        }

        // 3. Aux Query: shared query head (128) + dynamic hypernetwork head (128) -> 2 heads
        std::fill(q_aux_buf_.begin(), q_aux_buf_.end(), 0.0f);
        if (lp.q_shared_w) {
            gemv_fp16(q_aux_buf_.data(), lp.q_shared_w, norm_state_.data(), config_.head_dim, config_.hidden_size);
        }
        if (lp.expert_emb && lp.sel_mlp_0_w && lp.sel_mlp_2_w && lp.hyper_w_d && lp.hyper_w_u) {
            float context_vec[64] = {0.0f};
            for (int g = 0; g < 4; ++g) {
                for (int e = 0; e < 4; ++e) {
                    float w = rw[g * 4 + e] * 0.25f;
                    const uint16_t* emb = lp.expert_emb + (g * 4 + e) * 64;
                    for (int d = 0; d < 64; ++d) {
                        context_vec[d] += w * fp16_to_fp32(emb[d]);
                    }
                }
            }
            float sel_h[128];
            gemv_fp16(sel_h, lp.sel_mlp_0_w, context_vec, 128, 64);
            for (int i = 0; i < 128; ++i) {
                float v = sel_h[i] + (lp.sel_mlp_0_b ? fp16_to_fp32(lp.sel_mlp_0_b[i]) : 0.0f);
                sel_h[i] = v / (1.0f + std::exp(-v)); // SiLU
            }
            float sel[64];
            gemv_fp16(sel, lp.sel_mlp_2_w, sel_h, 64, 128);
            if (lp.sel_mlp_2_b) {
                for (int i = 0; i < 64; ++i) sel[i] += fp16_to_fp32(lp.sel_mlp_2_b[i]);
            }

            const int wd_rows = config_.hidden_size * 16;
            gemv_fp16(w_d_buf_.data(), lp.hyper_w_d, sel, wd_rows, 64);
            if (lp.hyper_w_d_b) {
                for (int i = 0; i < wd_rows; ++i) w_d_buf_[i] += fp16_to_fp32(lp.hyper_w_d_b[i]);
            }
            const int wu_rows = 16 * config_.head_dim;
            gemv_fp16(w_u_buf_.data(), lp.hyper_w_u, sel, wu_rows, 64);
            if (lp.hyper_w_u_b) {
                for (int i = 0; i < wu_rows; ++i) w_u_buf_[i] += fp16_to_fp32(lp.hyper_w_u_b[i]);
            }

            float h_mid[16] = {0.0f};
            for (int c = 0; c < config_.hidden_size; ++c) {
                float xc = norm_state_[c];
                const float* wd_row = w_d_buf_.data() + c * 16;
                for (int r = 0; r < 16; ++r) {
                    h_mid[r] += xc * wd_row[r];
                }
            }
            float* q_dyn = q_aux_buf_.data() + config_.head_dim;
            for (int r = 0; r < 16; ++r) {
                float hr = h_mid[r];
                const float* wu_row = w_u_buf_.data() + r * config_.head_dim;
                for (int d = 0; d < config_.head_dim; ++d) {
                    q_dyn[d] += hr * wu_row[d];
                }
            }
        } else {
            std::memcpy(q_aux_buf_.data() + config_.head_dim, q_aux_buf_.data(), config_.head_dim * sizeof(float));
        }

        // 4. RoPE Positional Encoding
        computeRoPE(q_group_buf_.data(), k_state_.data(), pos, config_.num_kv_heads, config_.num_kv_heads, config_.head_dim);
        computeRoPE(q_aux_buf_.data(), nullptr, pos, 2, 0, config_.head_dim);

        // 5. Store K, V into KV Cache at pos
        if (kv_cache_) {
            for (int h = 0; h < config_.num_kv_heads; ++h) {
                float* k_dest = kv_cache_->get_k(l, h, pos);
                float* v_dest = kv_cache_->get_v(l, h, pos);
                std::memcpy(k_dest, k_state_.data() + h * config_.head_dim, config_.head_dim * sizeof(float));
                std::memcpy(v_dest, v_state_.data() + h * config_.head_dim, config_.head_dim * sizeof(float));
            }
        }

        // 6. Scaled Dot-Product Attention over cache [0 .. pos]
        float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
        int total_len = std::min(pos + 1, config_.max_seq_len);
        std::fill(cat_heads_buf_.begin(), cat_heads_buf_.end(), 0.0f);

        // Group heads attention: 4 heads
        for (int g = 0; g < config_.num_kv_heads; ++g) {
            const float* q_h = q_group_buf_.data() + g * config_.head_dim;
            float* scores = attn_scores_buf_.data();
            float max_s = -1e9f;

            for (int t = 0; t < total_len; ++t) {
                const float* k_t = kv_cache_ ? kv_cache_->get_k(l, g, t) : (k_state_.data() + g * config_.head_dim);
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t d0 = vdupq_n_f32(0.0f);
                float32x4_t d1 = vdupq_n_f32(0.0f);
                for (int d = 0; d < config_.head_dim; d += 8) {
                    float32x4_t q_lo = vld1q_f32(q_h + d);
                    float32x4_t q_hi = vld1q_f32(q_h + d + 4);
                    float32x4_t k_lo = vld1q_f32(k_t + d);
                    float32x4_t k_hi = vld1q_f32(k_t + d + 4);
                    d0 = vmlaq_f32(d0, q_lo, k_lo);
                    d1 = vmlaq_f32(d1, q_hi, k_hi);
                }
                float dot = vaddvq_f32(vaddq_f32(d0, d1));
#else
                float dot = 0.0f;
                for (int d = 0; d < config_.head_dim; ++d) {
                    dot += q_h[d] * k_t[d];
                }
#endif
                scores[t] = dot * scale;
                if (scores[t] > max_s) max_s = scores[t];
            }

            float sum_exp = 0.0f;
            for (int t = 0; t < total_len; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);

            float* out_h = cat_heads_buf_.data() + g * config_.head_dim;
            for (int t = 0; t < total_len; ++t) {
                float a = scores[t] * inv_sum;
                const float* v_t = kv_cache_ ? kv_cache_->get_v(l, g, t) : (v_state_.data() + g * config_.head_dim);
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t a_vec = vdupq_n_f32(a);
                for (int d = 0; d < config_.head_dim; d += 8) {
                    float32x4_t o_lo = vld1q_f32(out_h + d);
                    float32x4_t o_hi = vld1q_f32(out_h + d + 4);
                    float32x4_t v_lo = vld1q_f32(v_t + d);
                    float32x4_t v_hi = vld1q_f32(v_t + d + 4);
                    vst1q_f32(out_h + d, vmlaq_f32(o_lo, a_vec, v_lo));
                    vst1q_f32(out_h + d + 4, vmlaq_f32(o_hi, a_vec, v_hi));
                }
#else
                for (int d = 0; d < config_.head_dim; ++d) {
                    out_h[d] += a * v_t[d];
                }
#endif
            }
        }

        // Aux heads attention: 2 heads (attend to mean-pooled KV)
        for (int a = 0; a < 2; ++a) {
            const float* q_h = q_aux_buf_.data() + a * config_.head_dim;
            float* scores = attn_scores_buf_.data();
            float max_s = -1e9f;

            for (int t = 0; t < total_len; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < config_.head_dim; ++d) {
                    float k_mean = 0.0f;
                    if (kv_cache_) {
                        for (int g = 0; g < config_.num_kv_heads; ++g) {
                            k_mean += kv_cache_->get_k(l, g, t)[d];
                        }
                        k_mean *= 0.25f;
                    }
                    dot += q_h[d] * k_mean;
                }
                scores[t] = dot * scale;
                if (scores[t] > max_s) max_s = scores[t];
            }

            float sum_exp = 0.0f;
            for (int t = 0; t < total_len; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                sum_exp += scores[t];
            }
            float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);

            float* out_h = cat_heads_buf_.data() + 512 + a * config_.head_dim;
            for (int t = 0; t < total_len; ++t) {
                float alpha = scores[t] * inv_sum;
                for (int d = 0; d < config_.head_dim; ++d) {
                    float v_mean = 0.0f;
                    if (kv_cache_) {
                        for (int g = 0; g < config_.num_kv_heads; ++g) {
                            v_mean += kv_cache_->get_v(l, g, t)[d];
                        }
                        v_mean *= 0.25f;
                    }
                    out_h[d] += alpha * v_mean;
                }
            }
        }

        // 7. Output projection (768 -> 2048)
        gemv_fp16(attn_out_.data(), lp.o_proj_w, cat_heads_buf_.data(), config_.hidden_size, 768);

        // 8. Residual Add
        for (int i = 0; i < config_.hidden_size; ++i) {
            hidden_state_[i] += attn_out_[i];
        }

        // FFN Pre-RMSNorm
        computeRMSNorm(norm_state_.data(), hidden_state_.data(), lp.ffn_norm_w, config_.hidden_size, config_.rms_norm_eps);

        // SwiGLU: gate & up projections
        gemv_fp16(gate_state_.data(), lp.gate_proj_w, norm_state_.data(), config_.intermediate_size, config_.hidden_size);
        gemv_fp16(up_state_.data(), lp.up_proj_w, norm_state_.data(), config_.intermediate_size, config_.hidden_size);

        for (int i = 0; i < config_.intermediate_size; ++i) {
            ffn_act_[i] = silu(gate_state_[i]) * up_state_[i];
        }

        // Down projection (5632 -> 2048)
        gemv_fp16(ffn_out_.data(), lp.down_proj_w, ffn_act_.data(), config_.hidden_size, config_.intermediate_size);

        // Residual Add
        for (int i = 0; i < config_.hidden_size; ++i) {
            hidden_state_[i] += ffn_out_[i];
        }
    }

    if (kv_cache_) {
        kv_cache_->current_len = pos + 1;
    }

    // 3 & 4. Skip expensive 128k LM head on non-final prefill tokens
    if (!compute_logits) {
        return logits_.data();
    }

    computeRMSNorm(norm_state_.data(), hidden_state_.data(), final_norm_weight_, config_.hidden_size, config_.rms_norm_eps);

    if (embed_weight_) {
        gemv_fp16(logits_.data(), embed_weight_, norm_state_.data(), config_.vocab_size, config_.hidden_size);
    }

    return logits_.data();
}

int HyperGQEEngine::sample(const float* logits, float temperature, float top_p, int top_k, float rep_penalty) {
    std::vector<float> work_logits(logits, logits + config_.vocab_size);

    // Calibrated sliding-window repetition penalty (last 32 tokens, content words only)
    if (rep_penalty != 1.0f && !generated_tokens_.empty()) {
        int window_size = 32;
        int start_idx = std::max(0, static_cast<int>(generated_tokens_.size()) - window_size);
        std::unordered_set<int> seen;
        for (size_t i = start_idx; i < generated_tokens_.size(); ++i) {
            int prev_tok = generated_tokens_[i];
            // Never penalize newlines, spaces, periods, or list numbers (0-9)
            if (prev_tok == 198 || prev_tok == 220 || prev_tok == 13 || (prev_tok >= 15 && prev_tok <= 25)) {
                continue;
            }
            if (prev_tok >= 0 && prev_tok < config_.vocab_size && seen.insert(prev_tok).second) {
                if (work_logits[prev_tok] > 0.0f) {
                    work_logits[prev_tok] /= rep_penalty;
                } else {
                    work_logits[prev_tok] *= rep_penalty;
                }
            }
        }
    }

    if (temperature <= 0.0f) {
        int best_idx = 0;
        float max_l = work_logits[0];
        for (int i = 1; i < config_.vocab_size; ++i) {
            if (work_logits[i] > max_l) {
                max_l = work_logits[i];
                best_idx = i;
            }
        }
        generated_tokens_.push_back(best_idx);
        return best_idx;
    }

    // Softmax with temperature
    std::vector<std::pair<float, int>> indexed_logits;
    indexed_logits.reserve(config_.vocab_size);
    for (int i = 0; i < config_.vocab_size; ++i) {
        indexed_logits.emplace_back(work_logits[i] / temperature, i);
    }

    std::partial_sort(indexed_logits.begin(), indexed_logits.begin() + std::min(top_k, config_.vocab_size),
                      indexed_logits.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    float max_l = indexed_logits[0].first;
    float sum_exp = 0.0f;
    std::vector<float> probs(top_k);
    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp(indexed_logits[i].first - max_l);
        sum_exp += probs[i];
    }
    for (int i = 0; i < top_k; ++i) {
        probs[i] /= sum_exp;
    }

    float cum_p = 0.0f;
    int cutoff = top_k;
    for (int i = 0; i < top_k; ++i) {
        cum_p += probs[i];
        if (cum_p >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, cum_p);
    float r = dist(gen);

    int chosen = indexed_logits[0].second;
    float current = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        current += probs[i];
        if (r <= current) {
            chosen = indexed_logits[i].second;
            break;
        }
    }

    generated_tokens_.push_back(chosen);
    return chosen;
}

} // namespace backend
} // namespace embeddra
