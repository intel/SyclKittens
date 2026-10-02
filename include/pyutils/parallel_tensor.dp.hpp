/**
 * @file
 * @brief TKParallelTensor — distributed tensor for Intel PVC multi-GPU.
 *
 * Ported from ThunderKittens CUDA (parallel_tensor.cuh).
 *
 * Key differences from CUDA version:
 *   - Uses SYCL USM device allocations (sycl::malloc_device)
 *   - Uses shared SYCL context for all devices (P2P via shared context)
 *   - No NVLS multicast (Intel PVC doesn't have it)
 *   - Level Zero IPC handles for cross-process sharing
 *   - c10::xpu instead of c10::cuda for device guards
 *
 * Within a single process, all devices share one sycl::context,
 * enabling direct P2P memory access without IPC handles.
 * IPC handles (Level Zero zeMemGetIpcHandle) are used for
 * cross-process scenarios (e.g., torchrun multi-process).
 */

#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <stdexcept>

#include <ATen/ops/from_blob.h>
#include <torch/csrc/utils/pybind.h>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUGuard.h>
#include <c10/xpu/XPUStream.h>

#include "broker.dp.hpp"

namespace kittens {
namespace py {

/**
 * @brief Distributed tensor wrapper for multi-GPU sharing on Intel PVC.
 *
 * Two modes:
 *   1. INTRA-PROCESS (default): Uses shared SYCL context + USM for P2P.
 *      All devices in the same process can access each other's memory.
 *   2. INTER-PROCESS: Uses Level Zero IPC handles via KittensBroker.
 *      Required for torchrun / multi-process distributed training.
 *
 * For the initial port, we support intra-process mode only.
 */
struct TKParallelTensor {
    at::Tensor data_;                // PyTorch tensor for direct access
    std::vector<int64_t> shape_;
    at::ScalarType dtype_;

    std::vector<void *> raw_ptrs_;   // per-device USM pointers (P2P accessible)
    size_t allocated_size_;

    int local_rank_;                 // device index
    int local_world_size_;

    // Constructor 1: wrap a pre-allocated PyTorch XPU tensor
    inline TKParallelTensor(
        const at::Tensor &tensor,
        int local_rank,
        int local_world_size
    ) : data_(tensor),
        shape_(tensor.sizes().vec()),
        dtype_(tensor.scalar_type()),
        raw_ptrs_(local_world_size, nullptr),
        allocated_size_(tensor.nbytes()),
        local_rank_(local_rank),
        local_world_size_(local_world_size) {

        TORCH_CHECK(tensor.is_xpu(), "Tensor must be on XPU device");
        TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
        TORCH_CHECK(tensor.dim() <= 4, "Only tensors with dim <= 4 are supported");
        TORCH_CHECK(tensor.device().index() == local_rank_, "Tensor device index must match local_rank");
        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");

        raw_ptrs_[local_rank_] = reinterpret_cast<void *>(tensor.data_ptr());
    }

    // Constructor 2: allocate new USM device memory with P2P access
    inline TKParallelTensor(
        const std::vector<int64_t> &shape,
        const at::ScalarType dtype,
        int local_rank,
        int local_world_size
    ) : shape_(shape),
        dtype_(dtype),
        raw_ptrs_(local_world_size, nullptr),
        allocated_size_(0),
        local_rank_(local_rank),
        local_world_size_(local_world_size) {

        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");
        TORCH_CHECK(!shape_.empty(), "Shape must be non-empty");
        TORCH_CHECK(shape_.size() <= 4, "Shape must have at most 4 dimensions");

        create_xpu_tensor();
    }

    TKParallelTensor(const TKParallelTensor&) = delete;
    TKParallelTensor& operator=(const TKParallelTensor&) = delete;
    TKParallelTensor& operator=(TKParallelTensor&&) = delete;

    inline TKParallelTensor(TKParallelTensor&& other)
        : data_(std::move(other.data_)),
          shape_(std::move(other.shape_)),
          dtype_(std::move(other.dtype_)),
          raw_ptrs_(std::move(other.raw_ptrs_)),
          allocated_size_(other.allocated_size_),
          local_rank_(other.local_rank_),
          local_world_size_(other.local_world_size_) {
        other.data_ = at::Tensor();
        other.shape_.clear();
        other.dtype_ = at::ScalarType::Undefined;
        other.raw_ptrs_.clear();
        other.allocated_size_ = 0;
        other.local_rank_ = -1;
        other.local_world_size_ = -1;
    }

    inline ~TKParallelTensor() { destroy(); }

    inline at::Tensor data() const { return data_; }

    /**
     * @brief Exchange raw device pointers between all ranks.
     *
     * For intra-process (single-process multi-GPU): pointers are directly
     * exchanged via shared memory. All devices must share the same SYCL context.
     *
     * For inter-process: uses KittensBroker to exchange pointers.
     * L0 IPC would be needed for true cross-process sharing, but for
     * torchrun each process manages one GPU, so we use the pointer exchange
     * approach.
     */
    inline void exchange_ptrs_intra(std::vector<void*> &all_ptrs) {
        // In single-process mode, all_ptrs must be provided by the caller
        // (collected from all devices in the same process)
        for (int i = 0; i < local_world_size_; i++) {
            raw_ptrs_[i] = all_ptrs[i];
        }
    }

private:
    inline void create_xpu_tensor() {
        size_t size = c10::elementSize(dtype_);
        for (auto dim : shape_) {
            TORCH_CHECK(dim > 0, "Size dimensions must be positive");
            size *= static_cast<size_t>(dim);
        }
        allocated_size_ = size;

        at::TensorOptions options = at::TensorOptions()
            .dtype(dtype_)
            .device(at::kXPU, local_rank_);

        data_ = at::empty(shape_, options);
        raw_ptrs_[local_rank_] = reinterpret_cast<void *>(data_.data_ptr());
    }

    inline void destroy() {
        if (data_.defined())
            data_.reset();
        shape_.clear();
        dtype_ = at::ScalarType::Undefined;
        raw_ptrs_.clear();
        allocated_size_ = 0;
        local_rank_ = -1;
        local_world_size_ = -1;
    }
};

} // namespace py
} // namespace kittens

#define BIND_TK_PARALLEL_TENSOR(m) \
    pybind11::class_<kittens::py::TKParallelTensor>(m, "TKParallelTensor") \
        .def(pybind11::init<const at::Tensor&, int, int>(), \
             pybind11::arg("tensor"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size")) \
        .def(pybind11::init<const std::vector<int64_t>&, const at::ScalarType&, int, int>(), \
             pybind11::arg("shape"), \
             pybind11::arg("dtype"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size")) \
        .def("data", &kittens::py::TKParallelTensor::data) \
        .def_readonly("data_", &kittens::py::TKParallelTensor::data_) \
        .def_readonly("local_rank_", &kittens::py::TKParallelTensor::local_rank_) \
        .def_readonly("local_world_size_", &kittens::py::TKParallelTensor::local_world_size_)
