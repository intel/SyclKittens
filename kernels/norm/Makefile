include ../common.mk

RMSNORM_D ?= 4096
LN_D ?= 4096
LN_WARPS_PER_ROW ?= 8

.PHONY: all rmsnorm layernorm

all: rmsnorm layernorm

rmsnorm: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DRMSNORM_D=$(RMSNORM_D) \
	  -DTORCH_EXTENSION_NAME=sk_rmsnorm rmsnorm.dp.cpp \
	  -o $(BUILD_DIR)/sk_rmsnorm$(EXT_SUFFIX) $(TORCH_LIBS)

layernorm: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DLN_D=$(LN_D) -DLN_WARPS_PER_ROW=$(LN_WARPS_PER_ROW) \
	  -DTORCH_EXTENSION_NAME=sk_layernorm layernorm.dp.cpp \
	  -o $(BUILD_DIR)/sk_layernorm$(EXT_SUFFIX) $(TORCH_LIBS)
