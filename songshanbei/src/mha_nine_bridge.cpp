#include "infinicore/ops/flash_attention.hpp"
#include "infinicore/ops/mha.hpp"

#ifdef H3_DIRECT_AOT
#include "infinicore/context/context.hpp"
#include "flash_attention.h"
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infinicore::op {
namespace {
struct PlannedMeta {
    graph::GraphTensor out;
    graph::GraphTensor q;
    graph::GraphTensor k;
    graph::GraphTensor v;
    std::optional<Tensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi_slopes, float scale, bool is_causal) {
    return new PlannedMeta{
        graph::GraphTensor(out), graph::GraphTensor(q), graph::GraphTensor(k),
        graph::GraphTensor(v), alibi_slopes, scale, is_causal};
}

void launch_nine_attention(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                           const Tensor &lengths, float scale, bool is_causal) {
#ifdef H3_DIRECT_AOT
    const auto argument = [](const Tensor &tensor) {
        return NineToothedTensor{
            const_cast<void *>(static_cast<const void *>(tensor->data())),
            const_cast<uint64_t *>(tensor->shape().data()),
            const_cast<int64_t *>(tensor->strides().data())};
    };
    uint64_t empty_shape[4]{16, 16, 16, 16};
    int64_t empty_strides[4]{1, 1, 1, 1};
    const NineToothedTensor mask{nullptr, empty_shape, empty_strides};
    const NineToothedTensor unused{nullptr, nullptr, nullptr};
    const NineToothedTensor scale_arg{&scale, nullptr, nullptr};
    context::setDevice(q->device());
    const auto result = launch_flash_attention(
        context::getStream(), argument(q), argument(k), argument(v), argument(lengths),
        mask, unused, scale_arg, argument(out), unused, unused,
        0, 128, is_causal, 0, 1, INFINI_DTYPE_BF16, 64, 64);
    if (result != 0) {
        std::fprintf(stderr, "H3_NINE_AOT_ERROR device=%zu status=%d\n",
                     q->device().getIndex(), int(result));
        throw std::runtime_error("NineToothed flash attention launch failed");
    }
#else
    flash_attention_(out, q, k, v, lengths, scale, is_causal);
#endif
}

#include "mha_nine_run.inc"

void cleanup(void **planned_meta) {
    delete static_cast<PlannedMeta *>(*planned_meta);
    *planned_meta = nullptr;
}
} // namespace

extern "C" void h3_register_nine_mha() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
}
} // namespace infinicore::op
