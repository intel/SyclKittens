include ../common.mk

ROTARY_D ?= 128
ROTARY_NUM_WORKERS ?= 4

ROTARY_DEFS := -DROTARY_D=$(ROTARY_D) -DROTARY_NUM_WORKERS=$(ROTARY_NUM_WORKERS)

.PHONY: all rotary rotary_qk rope_kvwrite

all: rotary rotary_qk rope_kvwrite

ifeq ($(CONFIG),standalone)
rotary rotary_qk rope_kvwrite: check-sycl
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) $(ROTARY_DEFS) $@.dp.cpp -o $(BUILD_DIR)/$@
else
rotary: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) $(ROTARY_DEFS) \
	  -DTORCH_EXTENSION_NAME=sk_rotary rotary.dp.cpp \
	  -o $(BUILD_DIR)/sk_rotary$(EXT_SUFFIX) $(TORCH_LIBS)

rotary_qk: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) $(ROTARY_DEFS) \
	  -DTORCH_EXTENSION_NAME=sk_rotary_qk rotary_qk.dp.cpp \
	  -o $(BUILD_DIR)/sk_rotary_qk$(EXT_SUFFIX) $(TORCH_LIBS)

rope_kvwrite: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) $(ROTARY_DEFS) \
	  -DTORCH_EXTENSION_NAME=sk_rope_kvwrite rope_kvwrite.dp.cpp \
	  -o $(BUILD_DIR)/sk_rope_kvwrite$(EXT_SUFFIX) $(TORCH_LIBS)
endif
