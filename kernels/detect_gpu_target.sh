#!/bin/sh
# Print the AOT target (-fsycl-targets value) of the first local Intel GPU, or
# nothing if none is found. Used by common.mk when GPU_TARGET is not set.
#   usage: detect_gpu_target.sh [compiler]
# Tries the compiler's own sycl-ls first ("Architecture: intel_gpu_*"), then the
# PCI device id in sysfs, which works without a GPU runtime.

cxx=$(command -v "${1:-icpx}" 2>/dev/null)
for sl in "${cxx:+$(dirname "$cxx")/sycl-ls}" "$(command -v sycl-ls 2>/dev/null)"; do
    [ -n "$sl" ] && [ -x "$sl" ] || continue
    arch=$("$sl" --verbose 2>/dev/null |
           sed -n 's/^ *Architecture: *\(intel_gpu_[a-z0-9_]*\).*/\1/p' | head -n 1)
    [ -n "$arch" ] && { echo "$arch"; exit 0; }
done

for d in /sys/class/drm/card*/device; do
    [ "$(cat "$d/vendor" 2>/dev/null)" = 0x8086 ] || continue
    case "$(cat "$d/device" 2>/dev/null)" in
        0x0bd*)        echo intel_gpu_pvc;     exit 0 ;;  # Data Center GPU Max (PVC)
        0xe20*|0xe21*) echo intel_gpu_bmg_g21; exit 0 ;;  # Arc B580/B570 (BMG-G21)
        0xe22*)        echo intel_gpu_bmg_g31; exit 0 ;;  # Arc Pro B70/B60 (BMG-G31)
    esac
done
