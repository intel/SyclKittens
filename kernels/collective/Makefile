include ../common.mk

.PHONY: all standalone all_reduce all_gather reduce_scatter all_to_all ag_gemm gemm_rs

all: standalone ag_gemm gemm_rs
standalone: all_reduce all_gather reduce_scatter all_to_all

all_reduce: check-sycl all_reduce_kernel.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) all_reduce.dp.cpp -o $(BUILD_DIR)/all_reduce

all_gather: check-sycl all_gather_kernel.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) all_gather.dp.cpp -o $(BUILD_DIR)/all_gather

reduce_scatter: check-sycl reduce_scatter_kernel.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) reduce_scatter.dp.cpp -o $(BUILD_DIR)/reduce_scatter

all_to_all: check-sycl all_to_all_kernel.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) all_to_all.dp.cpp -o $(BUILD_DIR)/all_to_all

ag_gemm: check-sycl ag_gemm_kernel.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) -DKITTENS_MULTI_GPU ag_gemm.dp.cpp \
	  -o $(BUILD_DIR)/ag_gemm

gemm_rs: check-sycl gemm_rs_kernel.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(COMMON_SYCL_FLAGS) gemm_rs.dp.cpp -o $(BUILD_DIR)/gemm_rs
