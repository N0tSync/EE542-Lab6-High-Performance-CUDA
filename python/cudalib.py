import ctypes
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DEFAULT = _HERE.parent / "build" / "libmatrix.so"

_f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
_u32 = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")


class CudaLib:
    def __init__(self, path=None):
        path = path or os.environ.get("LIBMATRIX", str(_DEFAULT))
        self._lib = ctypes.cdll.LoadLibrary(str(path))
        L = self._lib
        for name in ("gpu_matrix_multiply", "gpu_matrix_multiply_opt", "gpu_matrix_multiply_cublas"):
            fn = getattr(L, name)
            fn.argtypes = [_f32, _f32, _f32, ctypes.c_int]
            fn.restype = ctypes.c_int
        L.gpu_convolve.argtypes = [_u32, _u32, ctypes.c_int, ctypes.c_int, _f32, ctypes.c_int, ctypes.c_int]
        L.gpu_convolve.restype = ctypes.c_int
        L.gpu_convolve_batch.argtypes = [_u32, _u32, ctypes.c_int, ctypes.c_int, _f32, ctypes.c_int, ctypes.c_int]
        L.gpu_convolve_batch.restype = ctypes.c_int
        L.gpu_last_kernel_ms.restype = ctypes.c_float
        L.gpu_last_total_ms.restype = ctypes.c_float
        L.gpu_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.gpu_warmup.restype = ctypes.c_int

    # ------------------------------------------------------------------ info
    @property
    def device_name(self):
        buf = ctypes.create_string_buffer(256)
        self._check(self._lib.gpu_device_name(buf, 256), "gpu_device_name")
        return buf.value.decode()

    def warmup(self):
        self._check(self._lib.gpu_warmup(), "gpu_warmup")

    @property
    def last_kernel_ms(self):
        return float(self._lib.gpu_last_kernel_ms())

    @property
    def last_total_ms(self):
        return float(self._lib.gpu_last_total_ms())

    # ------------------------------------------------------------------ ops
    def matmul(self, A, B, impl="tiled"):
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        N = A.shape[0]
        if A.shape != (N, N) or B.shape != (N, N):
            raise ValueError("matmul expects two N x N matrices")
        C = np.empty((N, N), dtype=np.float32)
        fn = {"tiled": self._lib.gpu_matrix_multiply,
              "opt": self._lib.gpu_matrix_multiply_opt,
              "cublas": self._lib.gpu_matrix_multiply_cublas}[impl]
        # reshape(-1) on a C-contiguous array is a view, so the GPU writes straight into C
        self._check(fn(A.reshape(-1), B.reshape(-1), C.reshape(-1), N), impl)
        return C

    def convolve(self, img, kernel, tiled=True):
        img = np.ascontiguousarray(img, dtype=np.uint32)
        k = np.ascontiguousarray(kernel, dtype=np.float32)
        H, W = img.shape
        F = k.shape[0]
        out = np.empty_like(img)
        self._check(self._lib.gpu_convolve(img.reshape(-1), out.reshape(-1), W, H,
                                           k.reshape(-1), F, int(tiled)), "gpu_convolve")
        return out

    def convolve_batch(self, img, kernels):
        """Apply several same-size filters to one image with a single upload."""
        img = np.ascontiguousarray(img, dtype=np.uint32)
        ks = np.ascontiguousarray(np.stack(kernels), dtype=np.float32)
        H, W = img.shape
        n, F, _ = ks.shape
        out = np.empty((n, H, W), dtype=np.uint32)
        self._check(self._lib.gpu_convolve_batch(img.reshape(-1), out.reshape(-1), W, H,
                                                 ks.reshape(-1), F, n), "gpu_convolve_batch")
        return out

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed with code {rc}")


# ---------------------------------------------------------------------- reference
def convolve_numpy(img, kernel):
    """Same math as conv_cpu.c (true convolution, zero padding, |x| clamped to 0..255),
    vectorised with NumPy. Used to verify the GPU and as the non-accelerated Python baseline."""
    img = np.asarray(img, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    F = k.shape[0]
    L = F - 1 - F // 2
    kf = k[::-1, ::-1]
    H, W = img.shape
    pad = np.zeros((H + F - 1, W + F - 1), np.float32)
    pad[L:L + H, L:L + W] = img
    acc = np.zeros((H, W), np.float32)
    for a in range(F):
        for b in range(F):
            acc += kf[a, b] * pad[a:a + H, b:b + W]
    return np.minimum(np.floor(np.abs(acc) + 0.5), 255).astype(np.uint32)
