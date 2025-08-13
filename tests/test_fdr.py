from util.fdr import benjamini_hochberg


def test_bh_fdr_monotone_and_correct():
    """Test that BH-FDR produces monotone q-values."""
    # Test with increasing p-values
    pvals = {"a": 0.01, "b": 0.05, "c": 0.1}
    qvals = benjamini_hochberg(pvals)
    assert qvals["a"] <= qvals["b"] <= qvals["c"]


def test_bh_fdr_monotonicity():
    """Test that q-values are non-decreasing when mapped back to sorted order."""
    pvals = {"a": 0.1, "b": 0.05, "c": 0.01}
    qvals = benjamini_hochberg(pvals)
    # q-values should be non-decreasing when mapped back to sorted order
    assert qvals["c"] <= qvals["b"] <= qvals["a"]
