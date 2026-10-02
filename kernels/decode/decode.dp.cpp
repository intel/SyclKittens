/**
 * @file decode.dp.cpp
 * @brief SyclKittens decode orchestrator for single-process TP inference.
 *
 * Motivation: a Python-driven decode step is dominated by Python dispatch
 * -- the single Python thread submits ~2400 tiny ops/token sequentially while the
 * GPU sits nearly idle. This module moves the per-tile / per-op loop into C++ so
 * Python makes ONE call per decode half (per layer) instead of ~50, eliminating
 * the GIL-bound submission that makes single-process TP scale linearly worse than
 * multi-process xCCL.
 *
 * Phase 1 = the MLP half. Each tile d computes:
 *   h2   = rmsnorm(x[d], w[d])              [hidden]
 *   gate = h2 @ Bgate[d]                    [inter_shard]     (B-form [hidden,inter])
 *   up   = h2 @ Bup[d]                      [inter_shard]
 *   hmlp = silu(gate) * up                  [inter_shard]
 *   out  = hmlp @ Bdown[d]                  [hidden]          (B-form [inter,hidden])
 * All tiles are submitted on their own torch stream (peer-safe, async); one wait
 * at the end. Kernels are simple (decode is memory-bound; correctness > peak).
 */

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>
#include <vector>
#include <oneapi/mkl/blas.hpp>
#include "kernels.dp.hpp"

using sbf16 = sycl::ext::oneapi::bfloat16;

static inline sycl::queue &tq(int d) {
    return c10::xpu::getCurrentXPUStream(d).queue();
}

struct PtrPack { sbf16 *p[12]; };

// max K-splits for the flash-decode kernel (occupancy: launch nq*nsplit groups)
static constexpr int FSMAX = 64;

// 2D tensor x context parallel layout: N = g_tp * g_cp tiles. Set via set_2d().
static int    g_tp = 0;
static int    g_cp = 0;

// ---- persistent per-tile scratch (allocated once) ----
struct DecodeCtx {
    bool initialized = false;
    int n_gpus = 0, hidden = 0, inter_max = 0;
    int qkv_max = 0, nq_max = 0, hd = 0, max_ctx = 0;
    std::vector<sbf16*> h2, gate, up, hmlp;
    std::vector<sbf16*> qkv, qbuf, kbuf, attnb;   // attention scratch (bf16)
    std::vector<float*> gemm_f32;              // GEMM fp32 output scratch [max out]
    std::vector<float*> fO, fm, fl;            // flash-decode split partials
    std::vector<float*> cM, cD, cO;            // context-parallel per-tile local (max,sum,O)
    std::vector<sbf16*> opart, aro, x1b, dpart, ardp;   // fused-layer scratch [hidden]
    std::vector<long*>  barb;                            // device-barrier arrival counters

    void init(int ng, int H, int Imax, int qkvmax, int nqmax, int hdim, int mctx) {
        if (initialized) cleanup();
        n_gpus = ng; hidden = H; inter_max = Imax;
        qkv_max = qkvmax; nq_max = nqmax; hd = hdim; max_ctx = mctx;
        h2.resize(ng); gate.resize(ng); up.resize(ng); hmlp.resize(ng);
        qkv.resize(ng); qbuf.resize(ng); kbuf.resize(ng); attnb.resize(ng);
        opart.resize(ng); aro.resize(ng); x1b.resize(ng); dpart.resize(ng); ardp.resize(ng);
        gemm_f32.resize(ng);
        fO.resize(ng); fm.resize(ng); fl.resize(ng);
        cM.resize(ng); cD.resize(ng); cO.resize(ng);
        barb.resize(ng);
        int gout = Imax; if (qkvmax > gout) gout = qkvmax; if (H > gout) gout = H;
        if (gout < 1) gout = 1;
        for (int d = 0; d < ng; d++) {
            auto &q = tq(d);
            h2[d]   = sycl::malloc_device<sbf16>(H, q);
            gate[d] = sycl::malloc_device<sbf16>(Imax > 0 ? Imax : 1, q);
            up[d]   = sycl::malloc_device<sbf16>(Imax > 0 ? Imax : 1, q);
            hmlp[d] = sycl::malloc_device<sbf16>(Imax > 0 ? Imax : 1, q);
            qkv[d]   = sycl::malloc_device<sbf16>(qkvmax > 0 ? qkvmax : 1, q);
            qbuf[d]  = sycl::malloc_device<sbf16>(nqmax * hdim > 0 ? nqmax * hdim : 1, q);
            kbuf[d]  = sycl::malloc_device<sbf16>(nqmax * hdim > 0 ? nqmax * hdim : 1, q);
            attnb[d] = sycl::malloc_device<sbf16>(nqmax * hdim > 0 ? nqmax * hdim : 1, q);
            opart[d] = sycl::malloc_device<sbf16>(H, q);
            aro[d]   = sycl::malloc_device<sbf16>(H, q);
            x1b[d]   = sycl::malloc_device<sbf16>(H, q);
            dpart[d] = sycl::malloc_device<sbf16>(H, q);
            ardp[d]  = sycl::malloc_device<sbf16>(H, q);
            gemm_f32[d] = sycl::malloc_device<float>(gout, q);
            int nqm = nqmax > 0 ? nqmax : 1;
            int hdm = hdim > 0 ? hdim : 1;
            fO[d] = sycl::malloc_device<float>((long)nqm * FSMAX * hdm, q);
            fm[d] = sycl::malloc_device<float>((long)nqm * FSMAX, q);
            fl[d] = sycl::malloc_device<float>((long)nqm * FSMAX, q);
            cM[d] = sycl::malloc_device<float>(nqm, q);
            cD[d] = sycl::malloc_device<float>(nqm, q);
            cO[d] = sycl::malloc_device<float>((long)nqm * hdm, q);
            barb[d] = sycl::malloc_device<long>(1, q);
            q.memset(barb[d], 0, sizeof(long));
        }
        for (int d = 0; d < ng; d++) tq(d).wait();   // barrier counters zeroed globally before use
        initialized = true;
    }
    void cleanup() {
        for (int d = 0; d < n_gpus; d++) {
            auto &q = tq(d);
            auto F = [&](std::vector<sbf16*> &v) { if (v.size() > (size_t)d) sycl::free(v[d], q); };
            F(h2); F(gate); F(up); F(hmlp); F(qkv); F(qbuf); F(kbuf); F(attnb);
            F(opart); F(aro); F(x1b); F(dpart); F(ardp);
            if (gemm_f32.size() > (size_t)d) sycl::free(gemm_f32[d], q);
            if (fO.size() > (size_t)d) sycl::free(fO[d], q);
            if (fm.size() > (size_t)d) sycl::free(fm[d], q);
            if (fl.size() > (size_t)d) sycl::free(fl[d], q);
            if (cM.size() > (size_t)d) sycl::free(cM[d], q);
            if (cD.size() > (size_t)d) sycl::free(cD[d], q);
            if (cO.size() > (size_t)d) sycl::free(cO[d], q);
            if (barb.size() > (size_t)d) sycl::free(barb[d], q);
        }
        h2.clear(); gate.clear(); up.clear(); hmlp.clear();
        qkv.clear(); qbuf.clear(); kbuf.clear(); attnb.clear();
        opart.clear(); aro.clear(); x1b.clear(); dpart.clear(); ardp.clear();
        gemm_f32.clear();
        fO.clear(); fm.clear(); fl.clear();
        cM.clear(); cD.clear(); cO.clear();
        barb.clear();
        initialized = false;
    }
};
static DecodeCtx g_ctx;

// Cross-tile barrier: on-device atomic rendezvous. Each tile bumps every peer's
// monotonic counter and waits for the round's expected arrival count.
static long g_bround = 0;
static inline void device_barrier() {
    const int n = g_ctx.n_gpus;
    g_bround++; long ex = g_bround * (long)n;
    for (int d = 0; d < n; d++) {
        long* bpv[12]; for (int j = 0; j < 12; j++) bpv[j] = (j < n) ? g_ctx.barb[j] : nullptr;
        int dd = d, nn = n; long exx = ex;
        tq(d).single_task([=]() {
            for (int j = 0; j < nn; j++) {
                sycl::atomic_ref<long, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space> r(bpv[j][0]);
                r.fetch_add(1);
            }
            sycl::atomic_ref<long, sycl::memory_order::acq_rel, sycl::memory_scope::system,
                             sycl::access::address_space::global_space> me(bpv[dd][0]);
            while (me.load(sycl::memory_order::acquire) < exx) {}
        });
    }
}
static inline void barrier_all() {
    device_barrier();
}

// Wide-load split-K GEMV with a oneMKL fallback when K is not divisible by 2048.
static void launch_gemv(sycl::queue &q, sbf16 *in, sbf16 *W, sbf16 *out, float *scr,
                     int K, int N) {
    constexpr int WPR = 8, CHUNK = 256;
    if (K % (WPR * CHUNK) == 0) {
        decode_ops::launch_wide_gemv<WPR, CHUNK>(
            q, (decode_ops::kbf16*)in, (decode_ops::kbf16*)W,
            (decode_ops::kbf16*)out, K, N);
        return;
    }

    // Row-major C[N,1]=A[N,K]@B[K,1] via the col-major transpose trick.
    oneapi::mkl::blas::gemm(q,
        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        1, N, K, 1.0f, in, 1, W, K, 0.0f, scr, 1);
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { out[i] = (sbf16)scr[i]; });
}

// scatter roped k (kbuf [nkv,hd]) and v (vsrc [nkv,hd]) into the KV cache at
// position `pos` (KV layout [nkv, max_ctx, hd]). Pure data movement (glue).
static void k_kv_write(sycl::queue &q, const sbf16 *kroped, const sbf16 *vsrc,
                       sbf16 *kcache, sbf16 *vcache, int nkv, int hd, int pos,
                       int max_ctx) {
    q.parallel_for(sycl::range<1>((long)nkv * hd), [=](sycl::id<1> jj) {
        int j = (int)jj[0]; int kh = j / hd; int i = j % hd;
        long dst = (long)kh * max_ctx * hd + (long)pos * hd + i;
        kcache[dst] = kroped[(long)kh * hd + i];
        vcache[dst] = vsrc[(long)kh * hd + i];
    });
}

// flash-decode attention (split-K for occupancy): pass 1 launches nq*nsplit
// work-groups, each reducing its key sub-range to a local (max, sum, weighted-V)
// partial; pass 2 combines the nsplit partials per head via online softmax.
// This keeps the GPU busy at low batch (M=1) and, crucially, parallelizes the
// KV scan so cost grows sub-linearly with context length. GQA: group = nq/nkv.
static void k_flash(sycl::queue &q, const sbf16 *qbuf, const sbf16 *kcache,
                    const sbf16 *vcache, sbf16 *attnb,
                    int nq, int nkv, int hd, int L, int max_ctx,
                    float *fO, float *fm, float *fl) {
    constexpr int WG = 128;
    int group = nq / nkv;
    float scale = 1.f / sycl::sqrt((float)hd);
    int nsplit = (L + 511) / 512;                 // ~512 keys/split
    if (nsplit < 1) nsplit = 1;
    if (nsplit > FSMAX) nsplit = FSMAX;
    int per = (L + nsplit - 1) / nsplit;          // keys per split (<= 512)
    // ---- pass 1: per (head, split) partial ----
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(hd), cgh);
        sycl::local_accessor<float, 1> ssc(sycl::range<1>(per), cgh);
        cgh.parallel_for(sycl::nd_range<1>((long)nq * nsplit * WG, WG), [=](sycl::nd_item<1> it) {
            int g = (int)it.get_group(0);
            int s = g % nsplit, qh = g / nsplit;
            int lid = (int)it.get_local_id(0);
            int kvh = qh / group;
            const sbf16 *qv = qbuf + (long)qh * hd;
            const sbf16 *kh = kcache + (long)kvh * max_ctx * hd;
            const sbf16 *vh = vcache + (long)kvh * max_ctx * hd;
            int k0 = s * per, k1 = k0 + per; if (k1 > L) k1 = L;
            int cnt = k1 - k0;
            for (int d = lid; d < hd; d += WG) sq[d] = (float)qv[d];
            sycl::group_barrier(it.get_group());
            if (cnt <= 0) {                        // empty split -> neutral partial
                if (lid == 0) { fm[(long)qh * nsplit + s] = -1e30f; fl[(long)qh * nsplit + s] = 0.f; }
                for (int d = lid; d < hd; d += WG) fO[((long)qh * nsplit + s) * hd + d] = 0.f;
                return;
            }
            float lmax = -1e30f;
            for (int i = lid; i < cnt; i += WG) {
                const sbf16 *krow = kh + (long)(k0 + i) * hd;
                float a = 0.f;
                for (int d = 0; d < hd; d++) a += sq[d] * (float)krow[d];
                a *= scale; ssc[i] = a; lmax = sycl::max(lmax, a);
            }
            float m = sycl::reduce_over_group(it.get_group(), lmax, sycl::maximum<float>());
            float lsum = 0.f;
            for (int i = lid; i < cnt; i += WG) { float e = sycl::exp(ssc[i] - m); ssc[i] = e; lsum += e; }
            float l = sycl::reduce_over_group(it.get_group(), lsum, sycl::plus<float>());
            for (int d = lid; d < hd; d += WG) {
                float acc = 0.f;
                for (int i = 0; i < cnt; i++) acc += ssc[i] * (float)vh[(long)(k0 + i) * hd + d];
                fO[((long)qh * nsplit + s) * hd + d] = acc;   // unnormalized (max-shifted)
            }
            if (lid == 0) { fm[(long)qh * nsplit + s] = m; fl[(long)qh * nsplit + s] = l; }
        });
    });
    // ---- pass 2: combine nsplit partials per head (online softmax) ----
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> se(sycl::range<1>(nsplit), cgh);   // exp(m_s - M)
        cgh.parallel_for(sycl::nd_range<1>((long)nq * WG, WG), [=](sycl::nd_item<1> it) {
            int qh = (int)it.get_group(0);
            int lid = (int)it.get_local_id(0);
            const float *mrow = fm + (long)qh * nsplit;
            const float *lrow = fl + (long)qh * nsplit;
            const float *Orow = fO + (long)qh * nsplit * hd;
            float M = -1e30f;
            for (int s = 0; s < nsplit; s++) M = sycl::max(M, mrow[s]);
            for (int s = lid; s < nsplit; s += WG) se[s] = sycl::exp(mrow[s] - M);
            sycl::group_barrier(it.get_group());
            float denom = 0.f;
            for (int s = 0; s < nsplit; s++) denom += se[s] * lrow[s];
            float inv = 1.f / denom;
            for (int d = lid; d < hd; d += WG) {
                float acc = 0.f;
                for (int s = 0; s < nsplit; s++) acc += se[s] * Orow[(long)s * hd + d];
                attnb[(long)qh * hd + d] = (sbf16)(acc * inv);
            }
        });
    });
}

struct FPtrPack { float *p[12]; };

// ---- shared flash helpers (used by the 2D TPxCP orchestrator) ----
// One tile's split-K local partial over L keys (KV stride `mc`): writes per-query-
// head (max, sum, unnormalized-O) into cM/cD/cO. Same math as cp_flash pass1+combine.
static inline void flash_local(sycl::queue &q, sbf16 *qbuf, sbf16 *kc, sbf16 *vc,
    float *fO, float *fm, float *fl, float *cM, float *cD, float *cO,
    int nq, int nkv, int hd, int L, int mc) {
    constexpr int WG = 128;
    int group = nq / nkv;
    float scale = 1.f / sycl::sqrt((float)hd);
    if (L <= 0) {   // empty shard -> neutral partial
        q.parallel_for(sycl::range<1>(nq), [=](sycl::id<1> i) { cM[i[0]] = -1e30f; cD[i[0]] = 0.f; });
        q.parallel_for(sycl::range<1>((long)nq * hd), [=](sycl::id<1> i) { cO[i[0]] = 0.f; });
        return;
    }
    int nsplit = (L + 511) / 512; if (nsplit < 1) nsplit = 1; if (nsplit > FSMAX) nsplit = FSMAX;
    int per = (L + nsplit - 1) / nsplit;
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sq(sycl::range<1>(hd), cgh);
        sycl::local_accessor<float, 1> ssc(sycl::range<1>(per), cgh);
        cgh.parallel_for(sycl::nd_range<1>((long)nq * nsplit * WG, WG), [=](sycl::nd_item<1> it) {
            int g = (int)it.get_group(0); int s = g % nsplit, qh = g / nsplit;
            int lid = (int)it.get_local_id(0); int kvh = qh / group;
            const sbf16 *qh_v = qbuf + (long)qh * hd;
            const sbf16 *kh = kc + (long)kvh * mc * hd;
            const sbf16 *vh = vc + (long)kvh * mc * hd;
            int k0 = s * per, k1 = k0 + per; if (k1 > L) k1 = L; int cnt = k1 - k0;
            for (int i = lid; i < hd; i += WG) sq[i] = (float)qh_v[i];
            sycl::group_barrier(it.get_group());
            if (cnt <= 0) {
                if (lid == 0) { fm[(long)qh * nsplit + s] = -1e30f; fl[(long)qh * nsplit + s] = 0.f; }
                for (int i = lid; i < hd; i += WG) fO[((long)qh * nsplit + s) * hd + i] = 0.f;
                return;
            }
            float lmax = -1e30f;
            for (int i = lid; i < cnt; i += WG) {
                const sbf16 *krow = kh + (long)(k0 + i) * hd; float a = 0.f;
                for (int e = 0; e < hd; e++) a += sq[e] * (float)krow[e];
                a *= scale; ssc[i] = a; lmax = sycl::max(lmax, a);
            }
            float m = sycl::reduce_over_group(it.get_group(), lmax, sycl::maximum<float>());
            float lsum = 0.f;
            for (int i = lid; i < cnt; i += WG) { float e = sycl::exp(ssc[i] - m); ssc[i] = e; lsum += e; }
            float l = sycl::reduce_over_group(it.get_group(), lsum, sycl::plus<float>());
            for (int e = lid; e < hd; e += WG) {
                float acc = 0.f; for (int i = 0; i < cnt; i++) acc += ssc[i] * (float)vh[(long)(k0 + i) * hd + e];
                fO[((long)qh * nsplit + s) * hd + e] = acc;
            }
            if (lid == 0) { fm[(long)qh * nsplit + s] = m; fl[(long)qh * nsplit + s] = l; }
        });
    });
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> se(sycl::range<1>(nsplit), cgh);
        cgh.parallel_for(sycl::nd_range<1>((long)nq * WG, WG), [=](sycl::nd_item<1> it) {
            int qh = (int)it.get_group(0); int lid = (int)it.get_local_id(0);
            const float *mrow = fm + (long)qh * nsplit;
            const float *lrow = fl + (long)qh * nsplit;
            const float *Orow = fO + (long)qh * nsplit * hd;
            float M = -1e30f; for (int s = 0; s < nsplit; s++) M = sycl::max(M, mrow[s]);
            for (int s = lid; s < nsplit; s += WG) se[s] = sycl::exp(mrow[s] - M);
            sycl::group_barrier(it.get_group());
            float denom = 0.f; for (int s = 0; s < nsplit; s++) denom += se[s] * lrow[s];
            for (int e = lid; e < hd; e += WG) {
                float acc = 0.f; for (int s = 0; s < nsplit; s++) acc += se[s] * Orow[(long)s * hd + e];
                cO[(long)qh * hd + e] = acc;
            }
            if (lid == 0) { cM[qh] = M; cD[qh] = denom; }
        });
    });
}

// Merge `ng` per-tile partials (peer pointers in packs) -> normalized bf16 attn out.
static inline void flash_merge(sycl::queue &q, sbf16 *out, FPtrPack lM, FPtrPack lD,
                               FPtrPack lO, int ng, int nq, int hd) {
    constexpr int WG = 128;
    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> se(sycl::range<1>(ng), cgh);
        cgh.parallel_for(sycl::nd_range<1>((long)nq * WG, WG), [=](sycl::nd_item<1> it) {
            int qh = (int)it.get_group(0); int lid = (int)it.get_local_id(0);
            float M = -1e30f; for (int t = 0; t < ng; t++) M = sycl::max(M, lM.p[t][qh]);
            for (int t = lid; t < ng; t += WG) se[t] = sycl::exp(lM.p[t][qh] - M);
            sycl::group_barrier(it.get_group());
            float denom = 0.f; for (int t = 0; t < ng; t++) denom += se[t] * lD.p[t][qh];
            float inv = denom > 0.f ? 1.f / denom : 0.f;
            for (int e = lid; e < hd; e += WG) {
                float acc = 0.f; for (int t = 0; t < ng; t++) acc += se[t] * lO.p[t][(long)qh * hd + e];
                out[(long)qh * hd + e] = (sbf16)(acc * inv);
            }
        });
    });
}

// ============================================================
// Registered-weights fast path: register_layer() stores every layer's weight /
// KV pointers ONCE; decode_step() then runs ALL layers in a single C++ call, so
// Python marshals only x/cos/sin (3N tensors) per token instead of ~11 lists x N
// x 32 layers. Kills the per-layer pybind marshalling overhead.
// ============================================================
struct LayerW {
    sbf16 *norm_in[12], *Wqkv[12], *Wo[12];
    sbf16 *norm_post[12], *Wgate[12], *Wup[12], *Wdown[12];
    sbf16 *kc[12], *vc[12];
    long inter = 0;
};
static std::vector<LayerW> g_layers;

static void register_layer(int li, int nl,
    std::vector<torch::Tensor> norm_in, std::vector<torch::Tensor> Wqkv,
    std::vector<torch::Tensor> Wo, std::vector<torch::Tensor> norm_post,
    std::vector<torch::Tensor> Wgate, std::vector<torch::Tensor> Wup,
    std::vector<torch::Tensor> Wdown, std::vector<torch::Tensor> kcache,
    std::vector<torch::Tensor> vcache)
{
    if ((int)g_layers.size() != nl) g_layers.assign(nl, LayerW{});
    const int N = g_ctx.n_gpus;
    LayerW &lw = g_layers[li];
    for (int d = 0; d < N; d++) {
        lw.norm_in[d]   = reinterpret_cast<sbf16*>(norm_in[d].data_ptr());
        lw.Wqkv[d]      = reinterpret_cast<sbf16*>(Wqkv[d].data_ptr());
        lw.Wo[d]        = reinterpret_cast<sbf16*>(Wo[d].data_ptr());
        lw.norm_post[d] = reinterpret_cast<sbf16*>(norm_post[d].data_ptr());
        lw.Wgate[d]     = reinterpret_cast<sbf16*>(Wgate[d].data_ptr());
        lw.Wup[d]       = reinterpret_cast<sbf16*>(Wup[d].data_ptr());
        lw.Wdown[d]     = reinterpret_cast<sbf16*>(Wdown[d].data_ptr());
        lw.kc[d]        = reinterpret_cast<sbf16*>(kcache[d].data_ptr());
        lw.vc[d]        = reinterpret_cast<sbf16*>(vcache[d].data_ptr());
    }
    lw.inter = Wgate[0].numel() / g_ctx.hidden;
}

static inline void do_layer(const LayerW &lw, sbf16 **xptr, sbf16 **cptr,
                            sbf16 **sptr, int pos, int nq, int nkv, float eps) {
    const int N = g_ctx.n_gpus;
    const int H = g_ctx.hidden, hd = g_ctx.hd, mc = g_ctx.max_ctx;
    const int qkv_out = (nq + 2 * nkv) * hd;
    const int L = pos + 1;
    const long inter = lw.inter;
    PtrPack pop, pdp;
    for (int d = 0; d < N; d++) { pop.p[d] = g_ctx.opart[d]; pdp.p[d] = g_ctx.dpart[d]; }

    for (int d = 0; d < N; d++) {
        auto &q = tq(d);
        decode_ops::launch_rmsnorm_v6<4096, 8, false, false>(q, (decode_ops::kbf16*)xptr[d],
            nullptr, (decode_ops::kbf16*)lw.norm_in[d], (decode_ops::kbf16*)g_ctx.h2[d],
            nullptr, nullptr, 1, 1, eps);
        launch_gemv(q, g_ctx.h2[d], lw.Wqkv[d], g_ctx.qkv[d], g_ctx.gemm_f32[d], H, qkv_out);
        decode_ops::launch_rotary_qk<128, false>(q, (const decode_ops::kbf16*)g_ctx.qkv[d],
            (decode_ops::kbf16*)g_ctx.qbuf[d], (const decode_ops::kbf16*)(g_ctx.qkv[d] + nq * hd),
            (decode_ops::kbf16*)g_ctx.kbuf[d], (const decode_ops::kbf16*)cptr[d],
            (const decode_ops::kbf16*)sptr[d], 1, 1, nq, nkv);
        k_kv_write(q, g_ctx.kbuf[d], g_ctx.qkv[d] + (nq + nkv) * hd, lw.kc[d], lw.vc[d], nkv, hd, pos, mc);
        k_flash(q, g_ctx.qbuf[d], lw.kc[d], lw.vc[d], g_ctx.attnb[d], nq, nkv, hd, L, mc,
                g_ctx.fO[d], g_ctx.fm[d], g_ctx.fl[d]);
        launch_gemv(q, g_ctx.attnb[d], lw.Wo[d], g_ctx.opart[d], g_ctx.gemm_f32[d], nq * hd, H);
    }
    barrier_all();

    for (int d = 0; d < N; d++) {
        auto &q = tq(d);
        sbf16 *ar = g_ctx.aro[d]; PtrPack src = pop; int ng = N;
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; float s = 0.f;
            for (int t = 0; t < ng; t++) s += (float)src.p[t][i];
            ar[i] = (sbf16)s; });
        sbf16 *xp = xptr[d]; sbf16 *x1 = g_ctx.x1b[d];
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; x1[i] = (sbf16)((float)xp[i] + (float)ar[i]); });
    }
    for (int d = 0; d < N; d++) {                       // ---- phase C: mlp compute
        auto &q = tq(d);
        decode_ops::launch_rmsnorm_v6<4096, 8, false, false>(q, (decode_ops::kbf16*)g_ctx.x1b[d],
            nullptr, (decode_ops::kbf16*)lw.norm_post[d], (decode_ops::kbf16*)g_ctx.h2[d],
            nullptr, nullptr, 1, 1, eps);
        launch_gemv(q, g_ctx.h2[d], lw.Wgate[d], g_ctx.gate[d], g_ctx.gemm_f32[d], H, (int)inter);
        launch_gemv(q, g_ctx.h2[d], lw.Wup[d], g_ctx.up[d], g_ctx.gemm_f32[d], H, (int)inter);
        decode_ops::launch_silu_mul(q, (const uint16_t*)g_ctx.gate[d], (const uint16_t*)g_ctx.up[d],
                                (uint16_t*)g_ctx.hmlp[d], (size_t)inter);
        launch_gemv(q, g_ctx.hmlp[d], lw.Wdown[d], g_ctx.dpart[d], g_ctx.gemm_f32[d], (int)inter, H);
    }
    barrier_all();

    for (int d = 0; d < N; d++) {
        auto &q = tq(d);
        sbf16 *ar = g_ctx.ardp[d]; PtrPack src = pdp; int ng = N;
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; float s = 0.f;
            for (int t = 0; t < ng; t++) s += (float)src.p[t][i];
            ar[i] = (sbf16)s; });
        sbf16 *xp = xptr[d]; sbf16 *x1 = g_ctx.x1b[d];
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; xp[i] = (sbf16)((float)x1[i] + (float)ar[i]); });
    }
}

// Run ALL registered layers in one call. With register_io(), Python passes ZERO
// tensors per token (no pybind marshaling): it copy_'s the new embed/cos/sin into
// the registered bf16 buffers (format already matched -> no dtype conversion).
static sbf16 *g_xbuf[12], *g_cosbuf[12], *g_sinbuf[12];
static bool g_io_registered = false;

static void register_io(std::vector<torch::Tensor> xbuf,
                        std::vector<torch::Tensor> cosbuf,
                        std::vector<torch::Tensor> sinbuf) {
    const int N = g_ctx.n_gpus;
    for (int d = 0; d < N; d++) {
        g_xbuf[d]   = reinterpret_cast<sbf16*>(xbuf[d].data_ptr());
        g_cosbuf[d] = reinterpret_cast<sbf16*>(cosbuf[d].data_ptr());
        g_sinbuf[d] = reinterpret_cast<sbf16*>(sinbuf[d].data_ptr());
    }
    g_io_registered = true;
}

static void decode_step(int pos, int nq, int nkv, double eps) {
    TORCH_CHECK(g_ctx.initialized && !g_layers.empty() && g_io_registered,
                "sk_decode: init / register_layer / register_io first");
    for (size_t li = 0; li < g_layers.size(); li++)
        do_layer(g_layers[li], g_xbuf, g_cosbuf, g_sinbuf, pos, nq, nkv, (float)eps);
}

// ============================================================
// 2D TENSOR x CONTEXT PARALLEL decode layer. N = g_tp * g_cp tiles arranged as a
// [TP][CP] grid: tile d has tp = d/g_cp, cp = d % g_cp. Weights are TP-sharded and
// REPLICATED across the CP dimension (register_layer stores per-tile pointers, so
// the CP copies just point at the same-shape replica on their device). The KV cache
// is sequence-sharded: global position p lives on the CP tile p % g_cp at local
// index p / g_cp, so each CP tile holds ~1/g_cp of the keys. Per layer:
//   A: rmsnorm, QKV (TP gemm), rope, kv-write (owner CP tile only), LOCAL flash
//      partial over this tile's key shard  -> cM/cD/cO           [barrier]
//   B: CP-merge the g_cp partials within each TP group -> full attn; O-proj (TP)
//                                                                 [barrier]
//   C: TP all-reduce O within each CP column (g_tp tiles) + residual
//   D: rmsnorm, MLP (TP gemms + silu)                            [barrier]
//   E: TP all-reduce down within each CP column + residual
// Two collective groupings: CP-merge over a TP group (contiguous g_cp tiles),
// TP-AR over a CP column (strided by g_cp). gemm/AR are identical to plain TP=g_tp
// (CP only replicates them); the sole change is attention, sharded g_cp x further.
// ============================================================
static inline void do_layer_2d(const LayerW &lw, sbf16 **xptr, sbf16 **cptr,
                               sbf16 **sptr, int pos, int nq, int nkv, float eps) {
    const int TP = g_tp, CP = g_cp, N = TP * CP;
    const int H = g_ctx.hidden, hd = g_ctx.hd;
    const int Lshcap = (g_ctx.max_ctx + CP - 1) / CP;   // per-CP-tile KV capacity
    const int qkv_out = (nq + 2 * nkv) * hd;
    const long inter = lw.inter;

    // ---- A: norm, qkv, rope, kv-write(owner), local flash partial ----
    for (int d = 0; d < N; d++) {
        int cp = d % CP; auto &q = tq(d);
        decode_ops::launch_rmsnorm_v6<4096, 8, false, false>(q, (decode_ops::kbf16*)xptr[d],
            nullptr, (decode_ops::kbf16*)lw.norm_in[d], (decode_ops::kbf16*)g_ctx.h2[d],
            nullptr, nullptr, 1, 1, eps);
        launch_gemv(q, g_ctx.h2[d], lw.Wqkv[d], g_ctx.qkv[d], g_ctx.gemm_f32[d], H, qkv_out);
        decode_ops::launch_rotary_qk<128, false>(q, (const decode_ops::kbf16*)g_ctx.qkv[d],
            (decode_ops::kbf16*)g_ctx.qbuf[d], (const decode_ops::kbf16*)(g_ctx.qkv[d] + nq * hd),
            (decode_ops::kbf16*)g_ctx.kbuf[d], (const decode_ops::kbf16*)cptr[d],
            (const decode_ops::kbf16*)sptr[d], 1, 1, nq, nkv);
        if (pos % CP == cp) {          // only the owner CP tile stores the new k/v
            int lpos = pos / CP;
            k_kv_write(q, g_ctx.kbuf[d], g_ctx.qkv[d] + (nq + nkv) * hd,
                       lw.kc[d], lw.vc[d], nkv, hd, lpos, Lshcap);
        }
        int nlocal = (pos < cp) ? 0 : (pos - cp) / CP + 1;   // owned keys <= pos on this CP tile
        flash_local(q, g_ctx.qbuf[d], lw.kc[d], lw.vc[d], g_ctx.fO[d], g_ctx.fm[d],
                    g_ctx.fl[d], g_ctx.cM[d], g_ctx.cD[d], g_ctx.cO[d], nq, nkv, hd, nlocal, Lshcap);
    }
    barrier_all();                                   // CP partials ready

    // ---- B: CP-merge within TP group -> attnb; O-proj (TP) ----
    for (int d = 0; d < N; d++) {
        int tp = d / CP; auto &q = tq(d);
        FPtrPack pM, pD, pO;
        for (int c = 0; c < CP; c++) { int e = tp * CP + c; pM.p[c] = g_ctx.cM[e]; pD.p[c] = g_ctx.cD[e]; pO.p[c] = g_ctx.cO[e]; }
        flash_merge(q, g_ctx.attnb[d], pM, pD, pO, CP, nq, hd);
        launch_gemv(q, g_ctx.attnb[d], lw.Wo[d], g_ctx.opart[d], g_ctx.gemm_f32[d], nq * hd, H);
    }
    barrier_all();                                   // opart ready for TP-AR

    // ---- C: TP all-reduce O within CP column + residual (x1b = x + AR) ----
    for (int d = 0; d < N; d++) {
        int cp = d % CP; auto &q = tq(d);
        PtrPack src; int ng = TP;
        for (int t = 0; t < TP; t++) src.p[t] = g_ctx.opart[cp + t * CP];
        sbf16 *ar = g_ctx.aro[d];
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; float s = 0.f; for (int t = 0; t < ng; t++) s += (float)src.p[t][i]; ar[i] = (sbf16)s; });
        sbf16 *xp = xptr[d]; sbf16 *x1 = g_ctx.x1b[d];
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; x1[i] = (sbf16)((float)xp[i] + (float)ar[i]); });
    }

    // ---- D: MLP (TP) ----
    for (int d = 0; d < N; d++) {
        auto &q = tq(d);
        decode_ops::launch_rmsnorm_v6<4096, 8, false, false>(q, (decode_ops::kbf16*)g_ctx.x1b[d],
            nullptr, (decode_ops::kbf16*)lw.norm_post[d], (decode_ops::kbf16*)g_ctx.h2[d],
            nullptr, nullptr, 1, 1, eps);
        launch_gemv(q, g_ctx.h2[d], lw.Wgate[d], g_ctx.gate[d], g_ctx.gemm_f32[d], H, (int)inter);
        launch_gemv(q, g_ctx.h2[d], lw.Wup[d], g_ctx.up[d], g_ctx.gemm_f32[d], H, (int)inter);
        decode_ops::launch_silu_mul(q, (const uint16_t*)g_ctx.gate[d], (const uint16_t*)g_ctx.up[d],
                                (uint16_t*)g_ctx.hmlp[d], (size_t)inter);
        launch_gemv(q, g_ctx.hmlp[d], lw.Wdown[d], g_ctx.dpart[d], g_ctx.gemm_f32[d], (int)inter, H);
    }
    barrier_all();                                   // dpart ready for TP-AR

    // ---- E: TP all-reduce down within CP column + residual (x = x1b + AR) ----
    for (int d = 0; d < N; d++) {
        int cp = d % CP; auto &q = tq(d);
        PtrPack src; int ng = TP;
        for (int t = 0; t < TP; t++) src.p[t] = g_ctx.dpart[cp + t * CP];
        sbf16 *ar = g_ctx.ardp[d];
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; float s = 0.f; for (int t = 0; t < ng; t++) s += (float)src.p[t][i]; ar[i] = (sbf16)s; });
        sbf16 *xp = xptr[d]; sbf16 *x1 = g_ctx.x1b[d];
        q.parallel_for(sycl::range<1>(H), [=](sycl::id<1> ii) {
            long i = (long)ii[0]; xp[i] = (sbf16)((float)x1[i] + (float)ar[i]); });
    }
}

static void decode_2d_step(int pos, int nq, int nkv, double eps) {
    TORCH_CHECK(g_ctx.initialized && !g_layers.empty() && g_io_registered && g_tp > 0,
                "sk_decode: init / register_layer / register_io / set_2d first");
    for (size_t li = 0; li < g_layers.size(); li++)
        do_layer_2d(g_layers[li], g_xbuf, g_cosbuf, g_sinbuf, pos, nq, nkv, (float)eps);
}
static void py_set_2d(int tp, int cp) { g_tp = tp; g_cp = cp; }

// ---- python interface ----
static void py_init(int n_gpus, int hidden, int inter_max, int qkv_max,
                    int nq_max, int hd, int max_ctx) {
    g_ctx.init(n_gpus, hidden, inter_max, qkv_max, nq_max, hd, max_ctx);
    g_bround = 0;   // counters were re-zeroed by init; keep host round in lockstep
}
static void py_cleanup() { g_ctx.cleanup(); g_layers.clear(); g_io_registered = false; }
static bool py_is_initialized() { return g_ctx.initialized; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SyclKittens registered all-layer decode orchestrator";
    m.def("init", &py_init, pybind11::arg("n_gpus"), pybind11::arg("hidden"),
          pybind11::arg("inter_max"), pybind11::arg("qkv_max") = 0,
          pybind11::arg("nq_max") = 0, pybind11::arg("hd") = 128,
          pybind11::arg("max_ctx") = 0);
    m.def("cleanup", &py_cleanup);
    m.def("is_initialized", &py_is_initialized);
    m.def("register_layer", &register_layer,
          "store one layer's weight+KV pointers (call once per layer at setup)",
          pybind11::arg("li"), pybind11::arg("nl"), pybind11::arg("norm_in"),
          pybind11::arg("Wqkv"), pybind11::arg("Wo"), pybind11::arg("norm_post"),
          pybind11::arg("Wgate"), pybind11::arg("Wup"), pybind11::arg("Wdown"),
          pybind11::arg("kcache"), pybind11::arg("vcache"));
    m.def("register_io", &register_io,
          "store the fixed bf16 x/cos/sin input buffers (once); Python copy_'s into them",
          pybind11::arg("xbuf"), pybind11::arg("cosbuf"), pybind11::arg("sinbuf"));
    m.def("decode_step", &decode_step,
          "run ALL registered layers in one call, ZERO tensor marshaling per token",
          pybind11::arg("pos"), pybind11::arg("nq"), pybind11::arg("nkv"),
          pybind11::arg("eps"));
    m.def("set_2d", &py_set_2d,
          "set 2D tensor x context parallel layout: N = tp*cp tiles",
          pybind11::arg("tp"), pybind11::arg("cp"));
    m.def("decode_2d_step", &decode_2d_step,
          "run ALL layers of the 2D TPxCP decode (set_2d first)",
          pybind11::arg("pos"), pybind11::arg("nq"), pybind11::arg("nkv"),
          pybind11::arg("eps"));
}
