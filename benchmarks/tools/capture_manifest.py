#!/usr/bin/env python3
"""Capture a self-contained environment manifest for an SK benchmark run."""

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = [
    "KITTENS_TARGET",
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
    "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY",
    "UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY",
    "ZE_AFFINITY_MASK",
    "ZE_FLAT_DEVICE_HIERARCHY",
]


def command_output(command):
    try:
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except OSError:
        return None
    output = result.stdout.strip()
    return output if output else None


def source_identity():
    top = command_output(["git", "rev-parse", "--show-toplevel"])
    if top and Path(top).resolve() == ROOT:
        return {
            "kind": "git",
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "dirty_paths": (command_output(["git", "status", "--porcelain"]) or "").splitlines(),
        }

    return {"kind": "unknown"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = {
        "schema_version": 1,
        "suite": args.suite,
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "source": source_identity(),
        "toolchain": {
            "icpx": command_output(["icpx", "--version"]),
            "python": platform.python_version(),
        },
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
