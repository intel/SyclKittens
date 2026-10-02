# NOTE: WE HIGHLY RECOMMEND RUNNING WITH `make -j32` or however many threads your machine has.
# If you want to run lots of tests without parallel compilation, be prepared
# to spend a *long* time compiling. (like, HOURS single-threaded.)

COMP_LEVEL=profile
NAIVE_ONLY?=0

SK_NO_COMMON_CLEAN := 1
SYCL_TARGET_FLAGS :=
include ../kernels/common.mk

SYCL=$(CXX)
SYCLFLAGS=$(COMMON_SYCL_FLAGS) -Itesting_commons

# At what level to run tests?
# Level 1 takes 15 seconds to compile 1,000 tests.
# (Default) Level 2 takes a minute to compile 3,000 tests.
# Level 3 takes 5 minutes to compile 10,000 tests.
# Level 4 takes 15 while to compile 25,000 tests
SYCLFLAGS+= -DTEST_INTENSITY=2  #-DPORT_DEBUG
# Which tests to run?
# -DTEST_ALL means run all tests.
# You can also specify subsections, e.g. -DTEST_WARP_MEMORY
# Or individual tests, like -DTEST_WARP_MEMORY_VEC_DSMEM. Useful for debugging!
# The TEST_WARP_REGISTER_TILE_MMA will take at least 5 minutes to finish.
# The TEST_WARP_MEMORY_TILE_GLOBAL_TO_shared will take at least 5 minutes to finish.

WHICHFLAGS= passing
# Different macros: passing, all, or testing
# Add already passing tests to passing keyword below
# 'passing' = all tests known to work on Intel SYCL
# 'all' = full test suite (includes TMA/PGL/WGMMA which need Hopper)
# 'testing' = specific tests being debugged

# Kittens Intel will work on the intel device.
#-DKITTENS_INTEL

ifeq ($(WHICHFLAGS), passing)
SYCLFLAGS+= -DTEST_ALL_WARP_SHARED_VEC -DTEST_ALL_WARP_SHARED_TILE -DTEST_ALL_WARP_REGISTER_TILE -DTEST_ALL_WARP_REGISTER_VEC -DTEST_ALL_WARP_MEMORY_TILE -DTEST_ALL_WARP_MEMORY_VEC
else ifeq ($(WHICHFLAGS), all)
SYCLFLAGS+= -DTEST_ALL
else ifeq ($(WHICHFLAGS), testing)
SYCLFLAGS+=-DKITTENS_XE #-fsycl-targets=intel_gpu_pvc

SYCLFLAGS+=   -DTEST_WARP_REGISTER_TILE_REDUCTIONS -DTEST_WARP_REGISTER_TILE_MAPS  -DTEST_WARP_REGISTER_TILE_MMA -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER #-DTEST_ALL_WARP_MEMORY_VEC -DTEST_WARP_REGISTER_VEC_CONVERSIONS
#  SYCLFLAGS+= -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER #-DTEST_WARP_REGISTER_TILE_REDUCTIONS # -DTEST_WARP_REGISTER_TILE_REDUCTIONS   #-DTEST_WARP_REGISTER_TILE_MAPS  #-DTEST_WARP_REGISTER_TILE_MMA        # -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER  #-DTEST_WARP_REGISTER_TILE_MMA
endif


ifeq ($(COMP_LEVEL),fast)
# SYCLFLAGS+= -O0 --threads=0 -diag-suppress 20054
SYCLFLAGS+= -O2
else ifeq ($(COMP_LEVEL),debug)
SYCLFLAGS+= -g
else ifeq ($(COMP_LEVEL),profile)
# SYCLFLAGS+= -O3 --threads=0 -diag-suppress 20054 --dlto
SYCLFLAGS+= -O3
endif

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),INTEL)
SYCLFLAGS+=  -DKITTENS_INTEL
endif

# Target binary name
TARGET=unit_tests
BUILD_DIR=build

# Test source files
TESTS_SRC=$(shell find . -name '*.dp.cpp')

# Object files
OBJS=$(patsubst %.dp.cpp,$(BUILD_DIR)/%.o,$(TESTS_SRC))

# Default target
all: check-sycl build $(TARGET) collective-api testing-utils

# Create the build directory
build:
	mkdir -p $(BUILD_DIR)

# Rule to compile each CU file
$(BUILD_DIR)/%.o: %.dp.cpp
	mkdir -p $(@D)
	$(SYCL) $(SYCLFLAGS) -c $< -o $@

# Link object files to create the final executable
$(TARGET): $(OBJS)
	$(SYCL) $(SYCLFLAGS) $^ -o $(TARGET)

collective-api: $(BUILD_DIR)/collective_api_unit

$(BUILD_DIR)/collective_api_unit: collective_api_unit.cpp ../include/ops/collective/collective.dp.hpp ../include/ops/collective/topology.dp.hpp
	@mkdir -p $(@D)
	$(SYCL) $(COMMON_SYCL_FLAGS) $< -o $@

testing-utils: $(BUILD_DIR)/testing_utils_unit

$(BUILD_DIR)/testing_utils_unit: testing_utils_unit.cpp unit_tests.dp.cpp testing_commons/testing_utils.dp.hpp testing_commons/testing_commons.dp.hpp ../include/types/sycl_type.hpp ../include/ops/warp/register/tile/mma.dp.hpp ../include/ops/warp/register/tile/reductions.dp.hpp ../include/ops/warp/memory/tile/global_to_register.dp.hpp
	@mkdir -p $(@D)
	$(SYCL) $(COMMON_SYCL_FLAGS) -Itesting_commons $< -o $@

$(BUILD_DIR)/testing_utils_unit_release: $(BUILD_DIR)/testing_utils_unit
	$(SYCL) $(COMMON_SYCL_FLAGS) -DNDEBUG -Itesting_commons testing_utils_unit.cpp -o $@

check-load-contracts:
	@mkdir -p $(BUILD_DIR)
	@set -eu; \
	for mode in ROW COLUMN TRANSPOSE; do \
		case "$$mode" in \
			ROW) diagnostic="Unsupported row-layout load shape" ;; \
			COLUMN) diagnostic="Unsupported column-layout load shape" ;; \
			TRANSPOSE) diagnostic="Unsupported transpose load shape" ;; \
		esac; \
		log="$(BUILD_DIR)/unsupported_$${mode}.log"; \
		if $(SYCL) $(COMMON_SYCL_FLAGS) -Itesting_commons -fsyntax-only -DTEST_UNSUPPORTED_$${mode}_LOAD testing_utils_unit.cpp >"$$log" 2>&1; then \
			printf 'FAIL: unsupported %s load compiled\n' "$$mode"; exit 1; \
		fi; \
		if ! grep -F "$$diagnostic" "$$log" >/dev/null; then cat "$$log"; exit 1; fi; \
		printf 'PASS: unsupported %s load rejected\n' "$$mode"; \
	done

check-test-type-contracts:
	@mkdir -p $(BUILD_DIR)
	@set -eu; \
	for operation in VALIDATE INITIALIZE; do \
		for configuration in release debug; do \
			case "$$configuration" in release) assertion_flags="-DNDEBUG" ;; debug) assertion_flags="" ;; esac; \
			$(SYCL) $(COMMON_SYCL_FLAGS) -Itesting_commons -fsyntax-only $$assertion_flags -DTEST_$${operation}_TYPE testing_utils_unit.cpp; \
			log="$(BUILD_DIR)/unsupported_$${operation}_type_$${configuration}.log"; \
			if $(SYCL) $(COMMON_SYCL_FLAGS) -Itesting_commons -fsyntax-only $$assertion_flags -DTEST_$${operation}_TYPE -DTEST_UNSUPPORTED_TEST_TYPE testing_utils_unit.cpp >"$$log" 2>&1; then \
				printf 'FAIL: unsupported %s type compiled (%s)\n' "$$operation" "$$configuration"; exit 1; \
			fi; \
			if ! grep -F "Unsupported test element type" "$$log" >/dev/null; then cat "$$log"; exit 1; fi; \
			printf 'PASS: %s types accepted/rejected (%s)\n' "$$operation" "$$configuration"; \
		done; \
	done

check-host: collective-api testing-utils $(BUILD_DIR)/testing_utils_unit_release check-load-contracts check-test-type-contracts
	./$(BUILD_DIR)/collective_api_unit
	./$(BUILD_DIR)/testing_utils_unit
	./$(BUILD_DIR)/testing_utils_unit_release

# Run target
run: all
	./$(TARGET)
	./$(BUILD_DIR)/collective_api_unit
	./$(BUILD_DIR)/testing_utils_unit

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all build run clean collective-api testing-utils check-load-contracts check-test-type-contracts check-host
