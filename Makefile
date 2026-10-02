.PHONY: help check tests attention gemm norm rotary activation decode collective context-attention clean

help:
	@printf '%s\n' \
	  'SyclKittens build targets:' \
	  '  check              Validate toolchain and source layout' \
	  '  tests              Build the reusable API test suite' \
	  '  attention          Build GQA forward and backward extensions' \
	  '  gemm               Build the default BF16 GEMM extension' \
	  '  norm               Build RMSNorm and LayerNorm extensions' \
	  '  rotary             Build rotary extensions' \
	  '  activation         Build fused activation extensions' \
	  '  decode             Build the multi-GPU decode orchestrator' \
	  '  collective         Build standalone production collectives' \
	  '  context-attention  Build context-parallel attention binaries' \
	  '  clean              Remove generated build directories'

check:
	@$(MAKE) -s -C kernels/collective check-sycl
	@test -f include/kittens.dp.hpp
	@test -f kernels/attention/gqa_forward.dp.cpp
	@test -f kernels/rotary/rope_kvwrite.dp.cpp
	@test -f kernels/decode/decode.dp.cpp
	@test -f kernels/decode/kernels.dp.hpp
	@test -f kernels/collective/all_reduce.dp.cpp
	@test -f tests/unit_tests.dp.cpp
	@echo "SyclKittens source layout: OK"

tests:
	@$(MAKE) -C tests all

attention:
	@$(MAKE) -C kernels/attention all

gemm:
	@$(MAKE) -C kernels/gemm bf16

norm:
	@$(MAKE) -C kernels/norm all

rotary:
	@$(MAKE) -C kernels/rotary all

activation:
	@$(MAKE) -C kernels/activation all

decode:
	@$(MAKE) -C kernels/decode all

collective:
	@$(MAKE) -C kernels/collective standalone

context-attention:
	@$(MAKE) -C kernels/context_attention all

clean:
	@$(MAKE) -C tests clean
	@$(MAKE) -C kernels/attention clean
	@$(MAKE) -C kernels/gemm clean
	@$(MAKE) -C kernels/norm clean
	@$(MAKE) -C kernels/rotary clean
	@$(MAKE) -C kernels/activation clean
	@$(MAKE) -C kernels/decode clean
	@$(MAKE) -C kernels/collective clean
	@$(MAKE) -C kernels/context_attention clean
