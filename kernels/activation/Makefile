include ../common.mk

.PHONY: all silu_mul

all: silu_mul

silu_mul: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DTORCH_EXTENSION_NAME=sk_silu_mul silu_mul.dp.cpp \
	  -o $(BUILD_DIR)/sk_silu_mul$(EXT_SUFFIX) $(TORCH_LIBS)
