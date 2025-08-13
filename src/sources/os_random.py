import os


class OSRandom:
    """
    Wrapper over os.urandom with next_bit/next_bytes interface.
    Uses all 8 bits per byte (buffered) for efficiency.
    """

    __slots__ = ("_buf", "_byte_i", "_bit_i")

    def __init__(self):
        self._buf = b""
        self._byte_i = 0
        self._bit_i = 0

    def _refill(self):
        self._buf = os.urandom(4096)
        self._byte_i = 0
        self._bit_i = 0

    def next_bit(self) -> int:
        if self._byte_i >= len(self._buf):
            self._refill()
        b = self._buf[self._byte_i]
        bit = (b >> self._bit_i) & 1
        self._bit_i += 1
        if self._bit_i == 8:
            self._bit_i = 0
            self._byte_i += 1
        return bit

    def next_bytes(self, n: int) -> bytes:
        return os.urandom(n)
