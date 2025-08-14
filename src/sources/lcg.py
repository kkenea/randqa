from __future__ import annotations


class LCG:
    """Linear Congruential Generator for pseudorandom bitstreams.

    Recurrence: state = (a * state + c) mod m

    Byte/bit convention:
      - next_bit() returns the least-significant bit (LSB) of the new state.
      - next_bytes(n) packs 8 successive bits into each byte in LSB-first order,
        i.e., the first bit generated goes to bit position 0, the next to bit 1, etc.
        This matches NumPy's np.unpackbits(..., bitorder="little").
    """

    def __init__(
        self, seed: int = 1, a: int = 1664525, c: int = 1013904223, m: int = 2**32
    ):
        self.state = int(seed)  # current internal state
        self.a = int(a)  # multiplier
        self.c = int(c)  # increment
        self.m = int(m)  # modulus

        # Fast path for power-of-two modulus (common: 2**32)
        self._mask = (self.m - 1) if (self.m & (self.m - 1) == 0) else None

        # Normalize state into [0, m)
        if self._mask is not None:
            self.state &= self._mask
        else:
            self.state %= self.m

    def _advance(self) -> None:
        """Advance state once."""
        if self._mask is not None:
            # Equivalent to modulo when m is power of two
            self.state = (self.a * self.state + self.c) & self._mask
        else:
            self.state = (self.a * self.state + self.c) % self.m

    def next_bit(self) -> int:
        """Advance the state and return its least-significant bit (0 or 1)."""
        self._advance()
        return self.state & 1

    def next_bytes(self, n: int) -> bytes:
        """Generate n bytes by collecting 8 bits at a time (LSB-first within each byte)."""
        n = int(n)
        if n <= 0:
            return b""

        out = bytearray(n)
        # Pack 8 successive next_bit() results into one byte in LSB-first order.
        for i in range(n):
            b = 0
            # Unroll tiny loop for clarity; LSB-first: bit j goes to position j.
            self._advance()
            b |= (self.state & 1) << 0
            self._advance()
            b |= (self.state & 1) << 1
            self._advance()
            b |= (self.state & 1) << 2
            self._advance()
            b |= (self.state & 1) << 3
            self._advance()
            b |= (self.state & 1) << 4
            self._advance()
            b |= (self.state & 1) << 5
            self._advance()
            b |= (self.state & 1) << 6
            self._advance()
            b |= (self.state & 1) << 7
            out[i] = b
        return bytes(out)
