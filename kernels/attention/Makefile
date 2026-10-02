include ../common.mk

FWD_H ?= 32
FWD_H_KV ?= 8
FWD_D ?= 128
BWD_H ?= 16
BWD_H_KV ?= 16
BWD_D ?= 128

.PHONY: all forward backward

all: forward backward

forward: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) \
	  -DATTN_H_VALUE=$(FWD_H) -DATTN_H_KV_VALUE=$(FWD_H_KV) -DATTN_D_VALUE=$(FWD_D) \
	  -DTORCH_EXTENSION_NAME=sk_gqa_forward gqa_forward.dp.cpp \
	  -o $(BUILD_DIR)/sk_gqa_forward$(EXT_SUFFIX) $(TORCH_LIBS)

backward: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) \
	  -DATTN_H_VALUE=$(BWD_H) -DATTN_H_KV_VALUE=$(BWD_H_KV) -DATTN_D_VALUE=$(BWD_D) \
	  -ftarget-register-alloc-mode=pvc:large \
	  -DTORCH_EXTENSION_NAME=sk_gqa_backward gqa_backward.dp.cpp \
	  -o $(BUILD_DIR)/sk_gqa_backward$(EXT_SUFFIX) \
	  $(TORCH_LIBS)
