// Replace the Hygon MHA runner only after both Nine variants pass.
void run(void *planned_meta) {
    auto &plan = *static_cast<PlannedMeta *>(planned_meta);
    const auto &query = plan.q->shape();
    const auto &key = plan.k->shape();
    const auto &value = plan.v->shape();
    if (query.size() != 4 || key.size() != 4 || value != key ||
        plan.out->shape() != query || query[0] == 0 || query[0] != key[0] ||
        query[1] == 0 || key[1] == 0 || key[2] == 0 || query[2] == 0 ||
        query[2] % key[2] != 0 || query[3] != key[3] ||
        (query[3] != 72 && query[3] != 128) || plan.alibi_slopes ||
        (plan.is_causal && query[1] != key[1]) ||
        key[1] > static_cast<Size>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Unsupported Hygon NineToothed MHA shape");
    }
    const auto device = plan.q->device();
    if (plan.q->dtype() != DataType::BF16 || plan.k->dtype() != DataType::BF16 ||
        plan.v->dtype() != DataType::BF16 || plan.out->dtype() != DataType::BF16 ||
        plan.k->device() != device || plan.v->device() != device ||
        plan.out->device() != device) {
        throw std::runtime_error("Hygon NineToothed MHA requires colocated BF16 tensors");
    }

    const auto to_bhsd = [](const Tensor &tensor) {
        return tensor->permute({0, 2, 1, 3});
    };
    const auto pad_head = [](const Tensor &tensor) {
        Shape shape = tensor->shape();
        const auto width = shape.back();
        if (width == 128) {
            return tensor;
        }
        shape.back() = 128;
        auto padded = Tensor::zeros(shape, tensor->dtype(), tensor->device());
        padded->narrow({{3, 0, width}})->copy_from(tensor);
        return padded;
    };

    auto q = pad_head(to_bhsd(plan.q));
    auto k = pad_head(to_bhsd(plan.k));
    auto v = pad_head(to_bhsd(plan.v));
    auto out = to_bhsd(plan.out);
    auto kernel_out = query[3] == 128
        ? out
        : Tensor::empty(q->shape(), q->dtype(), device);

    const Size lengths_size = ((query[0] + 15) / 16) * 16;
    std::vector<int32_t> host_lengths(lengths_size, static_cast<int32_t>(key[1]));
    auto lengths = Tensor::from_blob(
        host_lengths.data(), {lengths_size}, DataType::I32, Device::cpu())->to(device);
    launch_nine_attention(kernel_out, q, k, v, lengths, plan.scale, plan.is_causal);
    if (query[3] != 128) {
        out->copy_from(kernel_out->narrow({{3, 0, query[3]}}));
    }

    static std::atomic<unsigned> reported_devices{0};
    const auto index = device.getIndex();
    if (index < std::numeric_limits<unsigned>::digits) {
        const unsigned mask = 1U << index;
        if ((reported_devices.fetch_or(mask, std::memory_order_relaxed) & mask) == 0) {
            std::fprintf(stderr, "H3_NINE_MHA_DISPATCH device=%zu\n", index);
        }
    }
}
