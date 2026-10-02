include ../common.mk

M ?= 4096
N ?= 4096
K ?= 4096

.PHONY: all bf16 qkv

all: bf16 qkv

bf16: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DGEMM_M=$(M) -DGEMM_N=$(N) -DGEMM_K=$(K) \
	  -DTORCH_EXTENSION_NAME=sk_bf16_gemm bf16_gemm.dp.cpp \
	  -o $(BUILD_DIR)/sk_bf16_gemm$(EXT_SUFFIX) $(TORCH_LIBS)

qkv: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DTORCH_EXTENSION_NAME=sk_qkv_gemm qkv_gemm.dp.cpp \
	  -o $(BUILD_DIR)/sk_qkv_gemm$(EXT_SUFFIX) $(TORCH_LIBS)
