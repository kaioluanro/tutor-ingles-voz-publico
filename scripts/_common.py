"""Shared helpers for the check_* scripts (Phase 1 manual verification)."""

from __future__ import annotations

import ctypes
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ctranslate2 (faster-whisper) needs cuBLAS 12 + cuDNN 9 at runtime. On a system
# without a CUDA 12 toolkit those libs only exist inside the pip `nvidia-*`
# packages, which are not on the dynamic linker path. Preloading them here
# avoids forcing the user to export LD_LIBRARY_PATH.
_CUDNN_ORDER = [
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn.so.9",
]
_CUBLAS_ORDER = ["libcublasLt.so.12", "libcublas.so.12"]
_preloaded = False


def preload_cuda_libs() -> list[str]:
    """Load pip-provided cuDNN/cuBLAS into the process; returns what loaded."""
    global _preloaded
    if _preloaded:
        return []
    _preloaded = True

    site = pathlib.Path(sys.prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
    loaded: list[str] = []
    for name in _CUDNN_ORDER:
        path = site / "nvidia" / "cudnn" / "lib" / name
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(name)
            except OSError as exc:
                print(f"[AVISO] nao carregou {name}: {exc}", file=sys.stderr)
    for name in _CUBLAS_ORDER:
        path = site / "nvidia" / "cublas" / "lib" / name
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                loaded.append(name)
            except OSError as exc:
                print(f"[AVISO] nao carregou {name}: {exc}", file=sys.stderr)
    return loaded


def load_yaml(name: str) -> dict:
    path = ROOT / name
    if not path.exists():
        print(f"[ERRO] arquivo nao encontrado: {path}", file=sys.stderr)
        raise SystemExit(1)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config() -> dict:
    return load_yaml("config.yaml")


def fmt_ms(ms: float) -> str:
    return f"{ms:8.1f} ms"
