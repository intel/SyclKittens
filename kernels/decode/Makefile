include ../common.mk

.PHONY: all decode

all: decode

decode: check-python
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(EXTENSION_FLAGS) -DTORCH_EXTENSION_NAME=sk_decode decode.dp.cpp \
	  -o $(BUILD_DIR)/sk_decode$(EXT_SUFFIX) $(TORCH_LIBS) -qmkl