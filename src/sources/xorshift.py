from __future__ import annotations


class XorShift32:
    """
    Marsaglia Xorshift32 PRNG with a next_bit/next_bytes interface.
    Reference: Marsaglia, G. (2003). Xorshift RNGs.

    Stream & bit/byte conventions
    -----------------------------
    - next_bit() and next_bytes() draw from a single sequential stream.
    - Within each 32-bit word, bits are exposed LSB-first.
      (Matches NumPy: np.unpackbits(bytes, bitorder="little"))
    - next_bytes() preserves continuity with prior next_bit() calls:
      if we've already consumed k bits from the current word,
      the next bytes continue from bit k, not from a fresh word.
    """

    __slots__ = ("state", "_word", "_bit_i")

    def __init__(self, seed: int = 2463534242):
        # XorShift32 must not be seeded with zero.
        if seed == 0:
            seed = 2463534242
        self.state = seed & 0xFFFFFFFF
        self._word = 0  # current 32-bit reservoir
        self._bit_i = 32  # next bit index within _word (0..32); 32 => empty

    # ---- core xorshift32 step ----
    def _next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

    def _refill_word(self) -> None:
        self._word = self._next_u32()
        self._bit_i = 0  # start at LSB of the new word

    # ---- bit API ----
    def next_bit(self) -> int:
        """Return the next bit (0/1), LSB-first within the current 32-bit word."""
        if self._bit_i >= 32:
            self._refill_word()
        bit = (self._word >> self._bit_i) & 1
        self._bit_i += 1
        return bit

    # ---- byte API ----
    def next_bytes(self, n: int) -> bytes:
        """
        Return the next n bytes from the stream.

        Fast path when aligned to a byte boundary (bit_i % 8 == 0): pull bytes
        directly from the current 32-bit reservoir in little-endian order,
        refilling words as needed.

        If not byte-aligned, assemble each byte from 8 next_bit() calls to
        preserve exact stream continuity.
        """
        n = int(n)
        if n <= 0:
            return b""

        out = bytearray(n)

        # Fast path: byte-aligned to 8 bits
        if self._bit_i % 8 == 0:
            i = 0
            while i < n:
                if self._bit_i >= 32:
                    self._refill_word()

                # Byte offset within current word: 0..3
                byte_off = self._bit_i // 8  # (since _bit_i % 8 == 0)
                # How many bytes remain in this word?
                can_take = 4 - byte_off
                take = min(n - i, can_take)

                # Emit 'take' bytes in little-endian order from the current word
                w = self._word
                for j in range(take):
                    out[i + j] = (w >> (8 * (byte_off + j))) & 0xFF

                # Advance bit index by the bytes we took
                self._bit_i += 8 * take
                i += take
            return bytes(out)

        # Slow path: not aligned to a byte boundary, compose bytes from bits
        for i in range(n):
            b = 0
            # LSB-first packing
            b |= self.next_bit() << 0
            b |= self.next_bit() << 1
            b |= self.next_bit() << 2
            b |= self.next_bit() << 3
            b |= self.next_bit() << 4
            b |= self.next_bit() << 5
            b |= self.next_bit() << 6
            b |= self.next_bit() << 7
            out[i] = b
        return bytes(out)
