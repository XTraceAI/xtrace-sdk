import numpy as np
import pytest
from xtrace_sdk.x_vec.inference.embedding import Embedding


def test_float_2_bin_positive_values() -> None:
    arr = np.array([0.5, -0.3, 1.0, -1.0, 0.0])
    result = Embedding.float_2_bin(arr)
    np.testing.assert_array_equal(result, np.array([1, 0, 1, 0, 0], dtype=np.int8))


def test_float_2_bin_from_list() -> None:
    result = Embedding.float_2_bin([0.1, -0.1, 0.0])
    np.testing.assert_array_equal(result, np.array([1, 0, 0], dtype=np.int8))


def test_float_2_bin_all_positive() -> None:
    arr = np.ones(10)
    result = Embedding.float_2_bin(arr)
    assert result.sum() == 10


def test_float_2_bin_all_negative() -> None:
    arr = -np.ones(10)
    result = Embedding.float_2_bin(arr)
    assert result.sum() == 0


def test_unsupported_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        Embedding("nonexistent_provider", "model", 512)


def test_hash_and_equality() -> None:
    e1 = Embedding("ollama", "model-a", 512)
    e2 = Embedding("ollama", "model-a", 512)
    e3 = Embedding("ollama", "model-b", 512)
    assert e1 == e2
    assert hash(e1) == hash(e2)
    assert e1 != e3
