# Decode is bandwidth-bound: 256 GRF halves the threads per XVE and costs 7-11%
# per token on B70, so keep 128 GRF unless LARGE_GRF=1 is passed explicitly.
LARGE_GRF ?= 0
include ../common.mk

.PHONY: all decode

all: decode

ifeq ($(CONFIG),standalone)
decode: check-sycl
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) decode.dp.cpp -o $(BUILD_DIR)/decode -qmkl
else
decode: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DTORCH_EXTENSION_NAME=sk_decode decode.dp.cpp \
	  -o $(BUILD_DIR)/sk_decode$(EXT_SUFFIX) $(TORCH_LIBS) -qmkl
endif
