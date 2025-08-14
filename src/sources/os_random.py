from __future__ import annotations
import os


class OSRandom:
    """
    Wrapper over os.urandom with next_bit/next_bytes interface.

    Stream & bit/byte conventions
    -----------------------------
    - The generator exposes a single sequential stream of randomness.
    - next_bit() consumes one bit at a time from that stream.
    - next_bytes(n) consumes the next n bytes from the same stream.
    - Within each byte, bits are read/packed in LSB-first order:
        the first bit generated goes to bit position 0,
        then position 1, ..., up to position 7.
      This matches NumPy: np.unpackbits(bytes, bitorder="little").

    Efficiency
    ----------
    - A 4 KiB internal buffer backs both next_bit() and next_bytes().
    - When byte-aligned, next_bytes() slices directly from the buffer.
      Otherwise, it assembles bytes by pulling 8 next_bit() calls.
    """

    __slots__ = ("_buf", "_byte_i", "_bit_i")

    def __init__(self):
        self._buf = b""
        self._byte_i = 0  # index of current byte in buffer
        self._bit_i = 0  # next bit position (0..7) within current byte

    def _refill(self) -> None:
        self._buf = os.urandom(4096)
        self._byte_i = 0
        self._bit_i = 0

    def _ensure_byte_available(self) -> None:
        if self._byte_i >= len(self._buf):
            self._refill()

    def next_bit(self) -> int:
        """Return the next bit (0/1) from the stream, LSB-first within each byte."""
        self._ensure_byte_available()
        b = self._buf[self._byte_i]
        bit = (b >> self._bit_i) & 1
        self._bit_i += 1
        if self._bit_i == 8:
            self._bit_i = 0
            self._byte_i += 1
        return bit

    def next_bytes(self, n: int) -> bytes:
        """
        Return the next n bytes from the stream.

        - If currently byte-aligned (bit_i == 0), slice from the internal buffer,
          refilling as needed (fast path).
        - If not aligned, assemble bytes by pulling 8 next_bit() calls per byte
          to preserve exact stream continuity (slow path, but rare).
        """
        n = int(n)
        if n <= 0:
            return b""

        # Fast path: byte-aligned; just slice from the buffer and advance.
        if self._bit_i == 0:
            out = bytearray(n)
            filled = 0
            while filled < n:
                self._ensure_byte_available()
                # Consume as many buffered bytes as possible
                remain = len(self._buf) - self._byte_i
                take = min(n - filled, remain)
                out[filled : filled + take] = self._buf[
                    self._byte_i : self._byte_i + take
                ]
                self._byte_i += take
                filled += take
            return bytes(out)

        # Slow path: not byte-aligned; produce bytes from next_bit() to keep continuity.
        out = bytearray(n)
        for i in range(n):
            b = 0
            # LSB-first packing: bit j goes to position j.
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
