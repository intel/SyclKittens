COMMON_MK := $(lastword $(MAKEFILE_LIST))
SK_ROOT := $(abspath $(dir $(COMMON_MK))/..)

ifeq ($(origin CXX), default)
CXX := icpx
endif

PYTHON ?= python3
# AOT target. When not given, detect the local Intel GPU (see detect_gpu_target.sh);
# fall back to PVC when none is found, e.g. on a GPU-less build host.
ifeq ($(origin GPU_TARGET), undefined)
GPU_TARGET := $(shell sh $(SK_ROOT)/kernels/detect_gpu_target.sh "$(CXX)")
ifeq ($(GPU_TARGET),)
GPU_TARGET := intel_gpu_pvc
endif
endif
BUILD_DIR ?= build
SYCL_TARGET_FLAGS ?= -fsycl-targets=$(GPU_TARGET)

# Build mode for kernels that ship a harness (<kernel>_harness.impl):
#   standalone : executable whose main() comes from the harness; SYCL only, no torch (default)
#   pytorch    : Python extension built with -DTORCH_COMPILE
CONFIG ?= standalone
ifeq ($(filter $(CONFIG),pytorch standalone),)
$(error CONFIG must be pytorch or standalone, got '$(CONFIG)')
endif

# 256-GRF ("large register file") AOT mode. PVC builds pick 256 GRF automatically
# (-autoGRFSelection); Battlemage (Xe2) AOT defaults to 128 GRF and the PVC-sized
# register tiles then spill silently (B70: GEMM 4096^3 4.4 -> 148 TF, GQA forward
# 3.6x). LARGE_GRF=auto (default) enables it for bmg targets; 1/0 force on/off.
LARGE_GRF ?= auto
ifeq ($(LARGE_GRF),auto)
SK_LARGE_GRF := $(if $(findstring bmg,$(GPU_TARGET)),1,0)
else
SK_LARGE_GRF := $(LARGE_GRF)
endif
LARGE_GRF_FLAGS := $(if $(filter 1,$(SK_LARGE_GRF)),-Xs "-options -ze-opt-large-register-file")
# Lets kernels pick Battlemage-specific launch configs (e.g. gqa_forward choose_fwd).
TARGET_DEFS := $(if $(findstring bmg,$(GPU_TARGET)),-DKITTENS_BMG)

SPIRV_FLAGS := -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
COMMON_SYCL_FLAGS := -fsycl $(SYCL_TARGET_FLAGS) $(LARGE_GRF_FLAGS) $(TARGET_DEFS) -std=c++20 -O3 -fsycl-unnamed-lambda -Wno-deprecated-declarations $(SPIRV_FLAGS) -DKITTENS_INTEL -DKITTENS_XE -DKITTENS_FAST_BF16_CONVERT -I$(SK_ROOT)/include

PYTHON_INCLUDE = $(shell "$(PYTHON)" -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE = $(shell "$(PYTHON)" -c "import pybind11; print(pybind11.get_include())")
TORCH_INCLUDES = $(shell "$(PYTHON)" -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
TORCH_LIBS = $(shell "$(PYTHON)" -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L'+p for p in library_paths()))") -lc10 -ltorch -ltorch_cpu -ltorch_python
EXT_SUFFIX = $(shell "$(PYTHON)" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
EXTENSION_FLAGS = $(COMMON_SYCL_FLAGS) -shared -fPIC -DTORCH_COMPILE -DTORCH_API_INCLUDE_EXTENSION_H -I$(PYTHON_INCLUDE) -I$(PYBIND_INCLUDE) $(TORCH_INCLUDES)

.PHONY: check-sycl check-python

check-sycl:
	@command -v "$(CXX)" >/dev/null 2>&1 || { echo "ERROR: icpx not found. Source env.src first."; exit 1; }
	@echo "SyclKittens: GPU_TARGET=$(GPU_TARGET) LARGE_GRF=$(SK_LARGE_GRF)"

check-python: check-sycl
	@"$(PYTHON)" -c "import pathlib, pybind11, torch; from torch.utils.cpp_extension import include_paths; assert any((pathlib.Path(p) / 'c10/xpu/XPUStream.h').is_file() for p in include_paths()), 'XPU-enabled PyTorch headers are required'"

ifndef SK_NO_COMMON_CLEAN
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
endif
