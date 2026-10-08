"""Arithmetic helpers: unit roundoffs, rounding emulation, bit flips."""
import numpy as np

U32 = 2.0 ** -24          # unit roundoff of binary32, round to nearest
U16 = 2.0 ** -11          # unit roundoff of binary16
UBF = 2.0 ** -8 + 2.0 ** -23   # emulated bfloat16 add (covers double rounding)
F16_HALF_SUB = 2.0 ** -25  # half the binary16 subnormal spacing
F32_TINY = 2.0 ** -149     # smallest binary32 subnormal


def gamma(u, d):
    """(1+u)^d - 1, the path constant of the accumulator-aware bound."""
    return np.expm1(d * np.log1p(u))


def flip_bit_f32(value, bit):
    """Return the binary32 value with bit `bit` (0 = least significant) inverted."""
    a = np.array([value], dtype=np.float32).view(np.uint32)
    a ^= np.uint32(1 << bit)
    return a.view(np.float32)[0]


def round_bf16(x):
    """Round a binary32 array to bfloat16 (round to nearest even); returns binary32."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> np.uint64(16)) & np.uint64(1)
    u = (u + np.uint64(0x7FFF) + lsb) & np.uint64(0xFFFF0000)
    return u.astype(np.uint32).view(np.float32)


def max_abs_rows(x):
    """Largest absolute entry of each row (per token)."""
    return np.max(np.abs(x), axis=-1)
