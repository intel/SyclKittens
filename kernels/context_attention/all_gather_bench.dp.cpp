#include "all_gather.dp.cpp"

static constexpr int NUM_DEVICES_MAX = 6;
static constexpr int D = ATTN_D;
static constexpr int WARMUP = 3;
static constexpr int ITERS  = 10;

template <int UNUSED> class ag_naive_ref_kernel;

static void naive_attn_ref(sycl::queue &queue,
                           sbf16 *d_Q, sbf16 *d_Kfull, sbf16 *d_Vfull,
                           float *d_Oref, int B, int N_q, int N_kv,
                           int H, int H_KV) {
  constexpr int Dd = ATTN_D;
  const int group = H / H_KV;
  const float scale = 1.0f / sycl::sqrt((float)Dd);
  queue.parallel_for<ag_naive_ref_kernel<0>>(
      sycl::range<1>((size_t)B * H * N_q), [=](sycl::id<1> idx) {
        int i = (int)idx[0];
        int b = i / (H * N_q);
        int r = i % (H * N_q);
        int h = r / N_q;
        int q = r % N_q;
        int kvh = h / group;
        const sbf16 *Qp = d_Q + (((long)b * N_q + q) * H + h) * Dd;
        float acc[Dd];
        #pragma unroll
        for (int d = 0; d < Dd; d++) acc[d] = 0.0f;
        float m = -1e30f, l = 0.0f;
        for (int k = 0; k < N_kv; k++) {
          const sbf16 *Kp = d_Kfull + (((long)b * N_kv + k) * H_KV + kvh) * Dd;
          float s = 0.0f;
          #pragma unroll
          for (int d = 0; d < Dd; d++) s += (float)Qp[d] * (float)Kp[d];
          s *= scale;
          float m_new = sycl::fmax(m, s);
          float corr = sycl::exp(m - m_new);
          float p = sycl::exp(s - m_new);
          l = l * corr + p;
          const sbf16 *Vp = d_Vfull + (((long)b * N_kv + k) * H_KV + kvh) * Dd;
          #pragma unroll
          for (int d = 0; d < Dd; d++) acc[d] = acc[d] * corr + p * (float)Vp[d];
          m = m_new;
        }
        float invl = 1.0f / l;
        float *Op = d_Oref + (((long)b * N_q + q) * H + h) * Dd;
        #pragma unroll
        for (int d = 0; d < Dd; d++) Op[d] = acc[d] * invl;
      });
}

int main() {
    auto platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> gpus;
    for (auto &p : platforms)
        for (auto &d : p.get_devices(sycl::info::device_type::gpu))
            gpus.push_back(d);

    int n_gpus = std::min(static_cast<int>(gpus.size()), NUM_DEVICES_MAX);
    std::cout << "=== All-Gather Attention v1 (concurrent gather + fence-free wg256/sq16 flash) ===" << std::endl;
    std::cout << "GPUs: " << n_gpus << " | D=" << D << " H=" << ATTN_H
              << " H_KV=" << ATTN_H_KV << " WG=" << RING_WG
              << " seq_q=" << RING_SEQ_Q << std::endl;
    if (n_gpus < 2) { std::cout << "Need >= 2 GPUs" << std::endl; return 1; }

    std::vector<sycl::device> sel(gpus.begin(), gpus.begin() + n_gpus);
    sycl::context ctx(sel);
    std::vector<sycl::queue> queues;
    for (int i = 0; i < n_gpus; i++)
        queues.emplace_back(ctx, sel[i],
            sycl::property_list{sycl::property::queue::in_order{}});

    int H = ATTN_H, H_KV = ATTN_H_KV;

    struct TestConfig { int B; int N_per_dev; bool verify; };
    std::vector<TestConfig> configs = {
        {1,   256, true},
        {1,  1024, false},
        {1,  2048, false},
        {1,  4096, false},
        {1,  8192, false},
        {1, 16384, false},
        {1, 32768, false},
        {4,  2048, false},
        {4,  4096, false},
        {4,  8192, false},
    };

    for (auto &tc : configs) {
        int B = tc.B;
        int N_per = tc.N_per_dev;
        int N_total = N_per * n_gpus;

        printf("\n========================================\n");
        printf("B=%d N_total=%d N_per=%d\n", B, N_total, N_per);
        printf("========================================\n");

        size_t qo_size   = (size_t)B * N_per   * H    * D;
        size_t kv_shard  = (size_t)B * N_per   * H_KV * D;
        size_t kv_full   = (size_t)B * N_total * H_KV * D;
        size_t l_size    = (size_t)B * H * N_per;

        std::vector<sbf16*> d_Q(n_gpus), d_K(n_gpus), d_V(n_gpus);
        std::vector<sbf16*> d_Kf(n_gpus), d_Vf(n_gpus);
        std::vector<float*> d_O(n_gpus), d_L(n_gpus);
        std::vector<sbf16**> d_Kptrs(n_gpus), d_Vptrs(n_gpus);
        std::vector<float*> d_Oref(n_gpus);

        for (int d = 0; d < n_gpus; d++) {
            d_Q[d]  = sycl::malloc_device<sbf16>(qo_size, queues[d]);
            d_K[d]  = sycl::malloc_device<sbf16>(kv_shard, queues[d]);
            d_V[d]  = sycl::malloc_device<sbf16>(kv_shard, queues[d]);
            d_Kf[d] = sycl::malloc_device<sbf16>(kv_full, queues[d]);
            d_Vf[d] = sycl::malloc_device<sbf16>(kv_full, queues[d]);
            d_O[d]  = sycl::malloc_device<float>(qo_size, queues[d]);
            d_L[d]  = sycl::malloc_device<float>(l_size, queues[d]);
        }
        for (auto &q : queues) q.wait();

        sbf16* h_Kptrs[NUM_DEVICES_MAX];
        sbf16* h_Vptrs[NUM_DEVICES_MAX];
        for (int d = 0; d < n_gpus; d++) { h_Kptrs[d] = d_Kf[d]; h_Vptrs[d] = d_Vf[d]; }
        for (int d = 0; d < n_gpus; d++) {
            d_Kptrs[d] = sycl::malloc_device<sbf16*>(n_gpus, queues[d]);
            d_Vptrs[d] = sycl::malloc_device<sbf16*>(n_gpus, queues[d]);
            queues[d].memcpy(d_Kptrs[d], h_Kptrs, n_gpus * sizeof(sbf16*));
            queues[d].memcpy(d_Vptrs[d], h_Vptrs, n_gpus * sizeof(sbf16*));
        }
        for (auto &q : queues) q.wait();

        auto reinit = [&]() {
            for (int d = 0; d < n_gpus; d++) {
                int dev = d;
                sbf16 *pQ = d_Q[d], *pK = d_K[d], *pV = d_V[d];
                queues[d].parallel_for(sycl::range<1>(qo_size), [=](sycl::id<1> i) {
                    unsigned h = (unsigned)(dev * 100003 + i[0] * 31 + 7);
                    h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
                    pQ[i] = static_cast<sbf16>((float)(h % 1000) / 5000.0f - 0.1f);
                });
                queues[d].parallel_for(sycl::range<1>(kv_shard), [=](sycl::id<1> i) {
                    unsigned h = (unsigned)(dev * 200003 + i[0] * 37 + 11);
                    h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
                    pK[i] = static_cast<sbf16>((float)(h % 1000) / 5000.0f - 0.1f);
                });
                queues[d].parallel_for(sycl::range<1>(kv_shard), [=](sycl::id<1> i) {
                    unsigned h = (unsigned)(dev * 300003 + i[0] * 41 + 13);
                    h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
                    pV[i] = static_cast<sbf16>((float)(h % 1000) / 5000.0f - 0.1f);
                });
            }
            for (auto &q : queues) q.wait();
        };
        reinit();

        reinit();
        for (int w = 0; w < WARMUP; w++) {
            reinit();
            run_all_gather_attention(queues, d_Q, d_Kf, d_Kptrs, d_K, d_Vf, d_Vptrs,
                                     d_V, d_O, d_L, n_gpus, B, N_per, N_total, H, H_KV);
        }
        std::vector<double> times;
        for (int iter = 0; iter < ITERS; iter++) {
            reinit();
            auto t0 = std::chrono::high_resolution_clock::now();
            run_all_gather_attention(queues, d_Q, d_Kf, d_Kptrs, d_K, d_Vf, d_Vptrs,
                                     d_V, d_O, d_L, n_gpus, B, N_per, N_total, H, H_KV);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(times.begin(), times.end());
        double med_us = times[ITERS / 2];
        double flops_per_call = 4.0 * B * H * (double)N_per * N_per * D;
        double total_flops_per_gpu = flops_per_call * n_gpus;
        double tflops = total_flops_per_gpu / (med_us * 1e-6) / 1e12;
        printf("  All-gather attention: %d us, %.1f TFLOP/s/GPU\n",
               (int)med_us, tflops);
        printf("AGPARSE B=%d N_per=%d US=%d TFLOPS=%.2f\n", B, N_per, (int)med_us, tflops);

        if (tc.verify) {
            reinit();
            run_all_gather_kv_push(queues, d_Kf, d_Kptrs, d_K, d_Vf, d_Vptrs, d_V,
                                   n_gpus, B, N_per, N_total, H_KV);
            run_all_gather_attention_flash(queues, d_Q, d_Kf, d_Vf, d_O, d_L,
                                           n_gpus, B, N_per, N_total, H, H_KV);
            d_Oref[0] = sycl::malloc_device<float>(qo_size, queues[0]);
            naive_attn_ref(queues[0], d_Q[0], d_Kf[0], d_Vf[0], d_Oref[0],
                           B, N_per, N_total, H, H_KV);
            queues[0].wait();
            std::vector<float> hO(qo_size), hR(qo_size);
            queues[0].memcpy(hO.data(), d_O[0], qo_size * sizeof(float)).wait();
            queues[0].memcpy(hR.data(), d_Oref[0], qo_size * sizeof(float)).wait();
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < qo_size; i++) {
                double diff = (double)hO[i] - (double)hR[i];
                num += diff * diff;
                den += (double)hR[i] * (double)hR[i];
            }
            double rel = (den > 0) ? std::sqrt(num / den) : 0.0;
            printf("AGCORR B=%d N_per=%d REL=%.4f %s\n", B, N_per, rel,
                   rel < 0.02 ? "PASS" : "FAIL");
            sycl::free(d_Oref[0], ctx);
        }

        for (int d = 0; d < n_gpus; d++) {
            sycl::free(d_Q[d], ctx);
            sycl::free(d_K[d], ctx);
            sycl::free(d_V[d], ctx);
            sycl::free(d_Kf[d], ctx);
            sycl::free(d_Vf[d], ctx);
            sycl::free(d_O[d], ctx);
            sycl::free(d_L[d], ctx);
            sycl::free(d_Kptrs[d], ctx);
            sycl::free(d_Vptrs[d], ctx);
        }
    }
    return 0;
}