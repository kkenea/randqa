import zlib


def compression_ratio(raw_bytes: bytes) -> float:
    """
    Compression ratio = compressed_size / raw_size using zlib(level=9).
    Truly random data should be ~incompressible (ratio ≈ 1.0).
    """
    if not raw_bytes:
        return 0.0
    comp = zlib.compress(raw_bytes, level=9)
    return len(comp) / len(raw_bytes)
