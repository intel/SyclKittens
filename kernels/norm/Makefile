include ../common.mk

RMSNORM_D ?= 4096
LN_D ?= 4096
LN_WARPS_PER_ROW ?= 8

RMSNORM_DEFS := -DRMSNORM_D=$(RMSNORM_D)
LN_DEFS := -DLN_D=$(LN_D) -DLN_WARPS_PER_ROW=$(LN_WARPS_PER_ROW)

.PHONY: all rmsnorm layernorm

all: rmsnorm layernorm

ifeq ($(CONFIG),standalone)
rmsnorm: check-sycl
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) $(RMSNORM_DEFS) rmsnorm.dp.cpp -o $(BUILD_DIR)/rmsnorm

layernorm: check-sycl
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) $(LN_DEFS) layernorm.dp.cpp -o $(BUILD_DIR)/layernorm
else
rmsnorm: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) $(RMSNORM_DEFS) \
	  -DTORCH_EXTENSION_NAME=sk_rmsnorm rmsnorm.dp.cpp \
	  -o $(BUILD_DIR)/sk_rmsnorm$(EXT_SUFFIX) $(TORCH_LIBS)

layernorm: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) $(LN_DEFS) \
	  -DTORCH_EXTENSION_NAME=sk_layernorm layernorm.dp.cpp \
	  -o $(BUILD_DIR)/sk_layernorm$(EXT_SUFFIX) $(TORCH_LIBS)
endif
