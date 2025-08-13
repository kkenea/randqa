class XorShift32:
    """
    Marsaglia Xorshift32 PRNG with a next_bit/next_bytes interface.
    Reference: Marsaglia, G. (2003). Xorshift RNGs.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int = 2463534242):
        if seed == 0:
            seed = 2463534242
        self.state = seed & 0xFFFFFFFF

    def _next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def next_bit(self) -> int:
        return self._next_u32() & 1

    def next_bytes(self, n: int) -> bytes:
        out = bytearray(n)
        word = 0
        for i in range(n):
            if i % 4 == 0:
                word = self._next_u32()
            out[i] = (word >> (8 * (i % 4))) & 0xFF
        return bytes(out)
