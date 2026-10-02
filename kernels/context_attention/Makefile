include ../common.mk

.PHONY: all ring ring_overlap all_gather

all: ring ring_overlap all_gather

ring: check-sycl ring.dp.cpp ring_bench.dp.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) ring_bench.dp.cpp -o $(BUILD_DIR)/ring_attention

ring_overlap: check-sycl ring_overlap.dp.cpp ring_overlap_bench.dp.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) ring_overlap_bench.dp.cpp -o $(BUILD_DIR)/ring_attention_overlap

all_gather: check-sycl all_gather.dp.cpp all_gather_bench.dp.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) all_gather_bench.dp.cpp -o $(BUILD_DIR)/all_gather_attention
