"""Small, readable pseudorandom number generators, written from scratch.

Question: what is actually inside a generator such as numpy's default PCG64,
and what goes wrong inside the old ones?

Each generator is a *state* plus two functions: a transition (state -> next
state) and an output (state -> number).  All of them are pure Python so that
every line can be read like a formula; they are slow, and that is fine for the
sizes used in chapter 6.
"""
from __future__ import annotations

import numpy as np

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
M128 = (1 << 128) - 1


# ----------------------------------------------------------------------------
# Linear congruential generators:  x_{n+1} = (a x_n + c) mod m
# ----------------------------------------------------------------------------
def lcg_ints(a: int, c: int, m: int, seed: int, n: int) -> np.ndarray:
    """Return x_1, ..., x_n of the LCG (the seed x_0 itself is not returned)."""
    out = np.empty(n, dtype=np.int64 if m <= 2**62 else object)
    x = seed % m
    for i in range(n):
        x = (a * x + c) % m
        out[i] = x
    return out


def lcg_uniform(a: int, c: int, m: int, seed: int, n: int) -> np.ndarray:
    """u_n = x_n / m, the usual way an LCG is turned into numbers in [0, 1)."""
    return lcg_ints(a, c, m, seed, n).astype(float) / m


def lcg_ints_vec(a: int, c: int, m: int, seed: int, n: int) -> np.ndarray:
    """Same sequence as lcg_ints, computed in blocks with numpy (m must be <= 2**32).

    x_{n+k} = (a^k x_n + c (a^{k-1}+...+1)) mod m, so a block of k consecutive
    values follows from one value by a vectorised multiply-add.
    """
    assert m <= 2**32
    k = 4096
    ak = np.empty(k, dtype=object)
    ck = np.empty(k, dtype=object)
    A, C = 1, 0
    for j in range(k):
        A, C = (a * A) % m, (a * C + c) % m
        ak[j], ck[j] = A, C
    ak = ak.astype(np.uint64)
    ck = ck.astype(np.uint64)
    out = np.empty(n, dtype=np.uint64)
    x = np.uint64(seed % m)
    mm = np.uint64(m)
    done = 0
    while done < n:
        # products < 2**64 because a^j mod m < 2**32 and x < 2**32
        blk = (ak * x % mm + ck) % mm
        take = min(k, n - done)
        out[done:done + take] = blk[:take]
        x = blk[take - 1]
        done += take
    return out.astype(np.int64)


def randu(seed: int, n: int) -> np.ndarray:
    """IBM's RANDU (1960s): x_{n+1} = 65539 x_n mod 2^31, returned as x_n / 2^31."""
    return lcg_ints_vec(65539, 0, 2**31, seed, n).astype(float) / 2**31


# ----------------------------------------------------------------------------
# Middle-square (von Neumann, 1949), four-digit version
# ----------------------------------------------------------------------------
def middle_square(seed: int, n: int) -> list[int]:
    """Square a 4-digit number, keep the middle 4 digits of the 8-digit square."""
    x, out = seed, []
    for _ in range(n):
        x = (x * x // 100) % 10000
        out.append(x)
    return out


# ----------------------------------------------------------------------------
# Xorshift64 (Marsaglia 2003): three shift-and-xor steps, linear over bits
# ----------------------------------------------------------------------------
def xorshift64(seed: int, n: int) -> list[int]:
    x, out = seed & M64, []
    assert x != 0, "the all-zero state maps to itself"
    for _ in range(n):
        x ^= (x << 13) & M64
        x ^= x >> 7
        x ^= (x << 17) & M64
        out.append(x)
    return out


# ----------------------------------------------------------------------------
# Mersenne Twister MT19937 (Matsumoto & Nishimura 1998)
# ----------------------------------------------------------------------------
class MT19937:
    """624 words of state; 'twist' is the transition, 'temper' the output."""

    def __init__(self, seed: int = 5489):
        self.mt = [0] * 624
        self.mt[0] = seed & M32
        for i in range(1, 624):  # standard init_genrand
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & M32
        self.i = 624

    def _twist(self):
        mt = self.mt
        for k in range(624):
            y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7FFFFFFF)
            mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self.i = 0

    def next32(self) -> int:
        if self.i >= 624:
            self._twist()
        y = self.mt[self.i]
        self.i += 1
        y ^= y >> 11                      # tempering: an invertible scramble
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y


# ----------------------------------------------------------------------------
# PCG64 (O'Neill 2014), the numpy default: 128-bit LCG state + XSL-RR output
# ----------------------------------------------------------------------------
PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645


def rotr64(x: int, r: int) -> int:
    return ((x >> r) | (x << ((64 - r) % 64))) & M64


class PCG64:
    """Bit-for-bit the same as numpy.random.PCG64 when given its (state, inc)."""

    def __init__(self, state: int, inc: int):
        self.state, self.inc = state & M128, inc & M128   # inc is odd

    def next64(self) -> int:
        self.state = (self.state * PCG_MULT + self.inc) & M128     # transition: an LCG mod 2^128
        s = self.state
        return rotr64(((s >> 64) ^ s) & M64, s >> 122)           # output: xor-fold, then rotate

    @classmethod
    def like_numpy(cls, bitgen: np.random.PCG64) -> "PCG64":
        st = bitgen.state["state"]
        return cls(int(st["state"]), int(st["inc"]))


def to_double(x64: int) -> float:
    """numpy's rule: keep the top 53 bits and scale by 2^-53, giving k/2^53 in [0, 1)."""
    return (x64 >> 11) * 2.0**-53
