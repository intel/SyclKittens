COMMON_MK := $(lastword $(MAKEFILE_LIST))
SK_ROOT := $(abspath $(dir $(COMMON_MK))/..)

ifeq ($(origin CXX), default)
CXX := icpx
endif

PYTHON ?= python3
GPU_TARGET ?= intel_gpu_pvc
BUILD_DIR ?= build
SYCL_TARGET_FLAGS ?= -fsycl-targets=$(GPU_TARGET)

SPIRV_FLAGS := -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
COMMON_SYCL_FLAGS := -fsycl $(SYCL_TARGET_FLAGS) -std=c++20 -O3 -fsycl-unnamed-lambda -Wno-deprecated-declarations $(SPIRV_FLAGS) -DKITTENS_INTEL -DKITTENS_XE -DKITTENS_FAST_BF16_CONVERT -I$(SK_ROOT)/include

PYTHON_INCLUDE = $(shell "$(PYTHON)" -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE = $(shell "$(PYTHON)" -c "import pybind11; print(pybind11.get_include())")
TORCH_INCLUDES = $(shell "$(PYTHON)" -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
TORCH_LIBS = $(shell "$(PYTHON)" -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L'+p for p in library_paths()))") -lc10 -ltorch -ltorch_cpu -ltorch_python
EXT_SUFFIX = $(shell "$(PYTHON)" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
EXTENSION_FLAGS = $(COMMON_SYCL_FLAGS) -shared -fPIC -DTORCH_COMPILE -DTORCH_API_INCLUDE_EXTENSION_H -I$(PYTHON_INCLUDE) -I$(PYBIND_INCLUDE) $(TORCH_INCLUDES)

.PHONY: check-sycl check-python

check-sycl:
	@command -v "$(CXX)" >/dev/null 2>&1 || { echo "ERROR: icpx not found. Source env.src first."; exit 1; }

check-python: check-sycl
	@"$(PYTHON)" -c "import pathlib, pybind11, torch; from torch.utils.cpp_extension import include_paths; assert any((pathlib.Path(p) / 'c10/xpu/XPUStream.h').is_file() for p in include_paths()), 'XPU-enabled PyTorch headers are required'"

ifndef SK_NO_COMMON_CLEAN
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
endif
