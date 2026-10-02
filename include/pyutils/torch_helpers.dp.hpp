#pragma once

#include <torch/extension.h>

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be an XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_XPU(x); CHECK_CONTIGUOUS(x)