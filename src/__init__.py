"""
randqa - Randomness Quality Assessment Tool

A toolkit for analyzing randomness quality in cryptographic applications.
Implements NIST SP 800-22 statistical tests, SP 800-90B health tests,
and machine learning predictability analysis.
"""

__version__ = "0.1.0"
__author__ = "Kena Kenea"
__email__ = "kkenea@proton.me"

# Import main functions for easy access
from .main import run_tests, bits_from_source
from .sources.lcg import LCG
from .sources.xorshift import XorShift32
from .sources.os_random import OSRandom

__all__ = [
    "run_tests",
    "bits_from_source",
    "LCG",
    "XorShift32",
    "OSRandom",
]
