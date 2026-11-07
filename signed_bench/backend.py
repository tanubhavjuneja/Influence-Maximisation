"""
backend.py
==========
Array backend for the signed benchmark.

  1. torch with CUDA  -> GPU (RTX 3060), used when available
  2. numpy            -> CPU fallback

Exposes the small numpy-compatible function surface that diffusion.py and
abcvns_paper.py use (zeros / arange / asarray / asnumpy / add_at_flat /
clip / maximum / minimum / astype / sum_axis / any_axis / dtypes), so the
algorithm code is identical on both backends.  Threshold draws and RNG
seeds stay on the CPU (numpy), so results are reproducible; GPU float32
atomic-adds may differ from CPU in the last bits (statistically identical).

NOTE: no cupy wheels exist for this Python version, so the GPU path is
torch's index_add_ scatter (equivalent to numpy's add.at).
"""
import os

import numpy as np

GPU = False
BACKEND = "numpy"

_FORCE = os.environ.get("SB_BACKEND", "").lower()
if _FORCE == "cpu":
    torch = None
else:
    try:
        import torch
        if torch.cuda.is_available():
            GPU = True
            BACKEND = "torch"
            _DEV = "cuda"
    except Exception:
        torch = None


class _Dtypes:
    pass


if BACKEND == "torch":
    f32 = torch.float32
    f64 = torch.float64
    i64 = torch.int64
    bool_ = torch.bool

    def zeros(shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device=_DEV)

    def arange(n, dtype=torch.int64):
        return torch.arange(n, dtype=dtype, device=_DEV)

    def asarray(x, dtype=None):
        if isinstance(x, torch.Tensor):
            t = x.to(_DEV)
            return t if dtype is None else t.to(dtype)
        t = torch.from_numpy(np.asarray(x)).to(_DEV)
        return t if dtype is None else t.to(dtype)

    def asnumpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    def astype(x, dtype):
        return x.to(dtype)

    def add_at_flat(arr_flat, idx_flat, vals_flat):
        """scatter-add: arr_flat[idx_flat] += vals_flat (add.at semantics)."""
        arr_flat.index_add_(0, idx_flat, vals_flat)

    def clip(x, lo, hi):
        return torch.clamp(x, lo, hi)

    def maximum(a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.full_like(b, float(a))
        if not isinstance(b, torch.Tensor):
            b = torch.full_like(a, float(b))
        return torch.maximum(a, b)

    def minimum(a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.full_like(b, float(a))
        if not isinstance(b, torch.Tensor):
            b = torch.full_like(a, float(b))
        return torch.minimum(a, b)

    def sum_axis(x, axis, dtype=None):
        return x.sum(dim=axis, dtype=dtype) if dtype is not None \
            else x.sum(dim=axis)

    def any_axis(x, axis):
        return x.any(dim=axis)

else:
    f32 = np.float32
    f64 = np.float64
    i64 = np.int64
    bool_ = np.bool_

    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    def arange(n, dtype=np.int64):
        return np.arange(n, dtype=dtype)

    def asarray(x, dtype=None):
        a = np.asarray(x)
        return a.astype(dtype) if dtype is not None else a

    def asnumpy(x):
        return np.asarray(x)

    def astype(x, dtype):
        return x.astype(dtype)

    def add_at_flat(arr_flat, idx_flat, vals_flat):
        np.add.at(arr_flat, idx_flat, vals_flat)

    def clip(x, lo, hi):
        return np.clip(x, lo, hi)

    def maximum(a, b):
        return np.maximum(a, b)

    def minimum(a, b):
        return np.minimum(a, b)

    def sum_axis(x, axis, dtype=None):
        return x.sum(axis=axis, dtype=dtype) if dtype is not None \
            else x.sum(axis=axis)

    def any_axis(x, axis):
        return x.any(axis=axis)


print(f"[backend] GPU (torch-cuda)={'ON' if GPU else 'OFF'} "
      f"-> {'torch CUDA' if GPU else 'numpy CPU'}", flush=True)