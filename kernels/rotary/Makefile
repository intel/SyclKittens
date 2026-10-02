include ../common.mk

ROTARY_D ?= 128
ROTARY_NUM_WORKERS ?= 4

.PHONY: all rotary rotary_qk rope_kvwrite

all: rotary rotary_qk rope_kvwrite

rotary: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DROTARY_D=$(ROTARY_D) -DROTARY_NUM_WORKERS=$(ROTARY_NUM_WORKERS) \
	  -DTORCH_EXTENSION_NAME=sk_rotary rotary.dp.cpp \
	  -o $(BUILD_DIR)/sk_rotary$(EXT_SUFFIX) $(TORCH_LIBS)

rotary_qk: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DROTARY_D=$(ROTARY_D) -DROTARY_NUM_WORKERS=$(ROTARY_NUM_WORKERS) \
	  -DTORCH_EXTENSION_NAME=sk_rotary_qk rotary_qk.dp.cpp \
	  -o $(BUILD_DIR)/sk_rotary_qk$(EXT_SUFFIX) $(TORCH_LIBS)

rope_kvwrite: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DROTARY_D=$(ROTARY_D) -DROTARY_NUM_WORKERS=$(ROTARY_NUM_WORKERS) \
	  -DTORCH_EXTENSION_NAME=sk_rope_kvwrite rope_kvwrite.dp.cpp \
	  -o $(BUILD_DIR)/sk_rope_kvwrite$(EXT_SUFFIX) $(TORCH_LIBS)
