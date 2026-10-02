#include "ring.dp.cpp"

static constexpr int NUM_DEVICES_MAX = 6;
static constexpr int D = ATTN_D;
static constexpr int WARMUP = 3;
static constexpr int ITERS = 10;

int main() {
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;
    for (auto &platform : platforms)
        for (auto &device : platform.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(device);

    int n_gpus = std::min(static_cast<int>(gpus.size()), NUM_DEVICES_MAX);
    std::cout << "=== Ring Attention v3 (best-of-best: fence-free v16 wg256/sq16 + memcpy rotation) ===" << std::endl;
    std::cout << "GPUs: " << n_gpus << " | D=" << D << " H=" << ATTN_H
              << " H_KV=" << ATTN_H_KV << " WG=" << RING_WG
              << " seq_q=" << RING_SEQ_Q << std::endl;
    if (n_gpus < 2) {
        std::cout << "Need >= 2 GPUs" << std::endl;
        return 1;
    }

    std::vector<sycl::device> selected(gpus.begin(), gpus.begin() + n_gpus);
    sycl::context context(selected);
    std::vector<sycl::queue> queues;
    for (int device = 0; device < n_gpus; device++)
        queues.emplace_back(context, selected[device],
            sycl::property_list{sycl::property::queue::in_order{}});

    int heads = ATTN_H;
    int kv_heads = ATTN_H_KV;
    struct TestConfig { int batch; int sequence_per_device; };
    std::vector<TestConfig> configs = {
        {1, 256}, {1, 1024}, {1, 2048}, {1, 4096}, {1, 8192},
        {1, 16384}, {1, 32768}, {4, 2048}, {4, 4096}, {4, 8192},
    };

    for (auto &config : configs) {
        int batch = config.batch;
        int sequence = config.sequence_per_device;
        int total_sequence = sequence * n_gpus;
        printf("\n========================================\n");
        printf("B=%d N_total=%d N_per=%d\n", batch, total_sequence, sequence);
        printf("========================================\n");

        size_t query_elements = (size_t)batch * sequence * heads * D;
        size_t kv_elements = (size_t)batch * sequence * kv_heads * D;
        size_t lse_elements = (size_t)batch * heads * sequence;
        std::vector<sbf16 *> queries(n_gpus);
        std::vector<sbf16 *> keys_0(n_gpus), keys_1(n_gpus);
        std::vector<sbf16 *> values_0(n_gpus), values_1(n_gpus);
        std::vector<float *> outputs(n_gpus), block_outputs(n_gpus);
        std::vector<float *> logsumexp(n_gpus), block_logsumexp(n_gpus);

        for (int device = 0; device < n_gpus; device++) {
            queries[device] = sycl::malloc_device<sbf16>(query_elements, queues[device]);
            keys_0[device] = sycl::malloc_device<sbf16>(kv_elements, queues[device]);
            keys_1[device] = sycl::malloc_device<sbf16>(kv_elements, queues[device]);
            values_0[device] = sycl::malloc_device<sbf16>(kv_elements, queues[device]);
            values_1[device] = sycl::malloc_device<sbf16>(kv_elements, queues[device]);
            outputs[device] = sycl::malloc_device<float>(query_elements, queues[device]);
            block_outputs[device] = sycl::malloc_device<float>(query_elements, queues[device]);
            logsumexp[device] = sycl::malloc_device<float>(lse_elements, queues[device]);
            block_logsumexp[device] = sycl::malloc_device<float>(lse_elements, queues[device]);
        }

        auto reinitialize = [&]() {
            for (int device = 0; device < n_gpus; device++) {
                int device_id = device;
                sbf16 *query = queries[device];
                sbf16 *key = keys_0[device];
                sbf16 *value = values_0[device];
                queues[device].parallel_for(
                    sycl::range<1>(query_elements), [=](sycl::id<1> index) {
                        unsigned hash =
                            (unsigned)(device_id * 100003 + index[0] * 31 + 7);
                        hash ^= hash >> 16; hash *= 0x45d9f3b; hash ^= hash >> 16;
                        query[index] = static_cast<sbf16>(
                            (float)(hash % 1000) / 5000.0f - 0.1f);
                    });
                queues[device].parallel_for(
                    sycl::range<1>(kv_elements), [=](sycl::id<1> index) {
                        unsigned hash =
                            (unsigned)(device_id * 200003 + index[0] * 37 + 11);
                        hash ^= hash >> 16; hash *= 0x45d9f3b; hash ^= hash >> 16;
                        key[index] = static_cast<sbf16>(
                            (float)(hash % 1000) / 5000.0f - 0.1f);
                    });
                queues[device].parallel_for(
                    sycl::range<1>(kv_elements), [=](sycl::id<1> index) {
                        unsigned hash =
                            (unsigned)(device_id * 300003 + index[0] * 41 + 13);
                        hash ^= hash >> 16; hash *= 0x45d9f3b; hash ^= hash >> 16;
                        value[index] = static_cast<sbf16>(
                            (float)(hash % 1000) / 5000.0f - 0.1f);
                    });
            }
            for (auto &queue : queues)
                queue.wait();
        };
        auto run = [&]() {
            run_ring_attention(
                queues, queries, keys_0, keys_1, values_0, values_1,
                outputs, block_outputs, logsumexp, block_logsumexp,
                batch, sequence, heads, kv_heads, kv_elements);
        };

        reinitialize();
        for (int warmup = 0; warmup < WARMUP; warmup++) {
            reinitialize();
            run();
        }
        std::vector<double> times;
        for (int iteration = 0; iteration < ITERS; iteration++) {
            reinitialize();
            auto start = std::chrono::high_resolution_clock::now();
            run();
            auto stop = std::chrono::high_resolution_clock::now();
            times.push_back(
                std::chrono::duration<double, std::micro>(stop - start).count());
        }
        std::sort(times.begin(), times.end());
        double median_microseconds = times[ITERS / 2];
        double flops_per_call =
            4.0 * batch * heads * (double)sequence * sequence * D;
        double tflops = flops_per_call * n_gpus /
            (median_microseconds * 1e-6) / 1e12;
        printf("  Ring: %d us, %.1f TFLOP/s/GPU\n",
               (int)median_microseconds, tflops);

        double kv_bytes = (double)kv_elements * sizeof(sbf16);
        double p2p_mb = kv_bytes * 2.0 * (n_gpus - 1) / (1024.0 * 1024.0);
        printf("\n  KV per chunk: %.1f MB  Total P2P rotation: %.1f MB/dev\n",
               kv_bytes * 2 / (1024.0 * 1024.0), p2p_mb);

        for (int device = 0; device < n_gpus; device++) {
            sycl::free(queries[device], context);
            sycl::free(keys_0[device], context);
            sycl::free(keys_1[device], context);
            sycl::free(values_0[device], context);
            sycl::free(values_1[device], context);
            sycl::free(outputs[device], context);
            sycl::free(block_outputs[device], context);
            sycl::free(logsumexp[device], context);
            sycl::free(block_logsumexp[device], context);
        }
    }
    return 0;
}