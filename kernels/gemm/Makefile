include ../common.mk

M ?= 4096
N ?= 4096
K ?= 4096

GEMM_DEFS := -DGEMM_M=$(M) -DGEMM_N=$(N) -DGEMM_K=$(K)

.PHONY: all bf16 qkv

all: bf16 qkv

ifeq ($(CONFIG),standalone)
bf16: check-sycl
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) $(GEMM_DEFS) bf16_gemm.dp.cpp \
	  -o $(BUILD_DIR)/bf16_gemm_$(M)x$(N)x$(K)
else
bf16: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) $(GEMM_DEFS) \
	  -DTORCH_EXTENSION_NAME=sk_bf16_gemm bf16_gemm.dp.cpp \
	  -o $(BUILD_DIR)/sk_bf16_gemm$(EXT_SUFFIX) $(TORCH_LIBS)
endif

ifeq ($(CONFIG),standalone)
qkv: check-sycl
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) qkv_gemm.dp.cpp -o $(BUILD_DIR)/qkv_gemm
else
qkv: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DTORCH_EXTENSION_NAME=sk_qkv_gemm qkv_gemm.dp.cpp \
	  -o $(BUILD_DIR)/sk_qkv_gemm$(EXT_SUFFIX) $(TORCH_LIBS)
endif
