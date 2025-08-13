class LCG:
    """Linear Congruential Generator for pseudorandom bitstreams."""

    def __init__(self, seed=1, a=1664525, c=1013904223, m=2**32):
        self.state = seed  # current internal state
        self.a = a  # multiplier
        self.c = c  # increment
        self.m = m  # modulus

    def next_bit(self):
        """
        Advance the state and return its least-significant bit.
        This extracts one bit of pseudorandomness per call.
        """
        self.state = (self.a * self.state + self.c) % self.m
        return self.state & 1

    def next_bytes(self, n):
        """
        Generate n bytes by collecting 8 bits at a time.
        Returns a bytes object of length n.
        """
        bits = [self.next_bit() for _ in range(n * 8)]
        # pack bits into bytes
        return bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(n))
