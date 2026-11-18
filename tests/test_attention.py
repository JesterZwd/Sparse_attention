import numpy as np
import pytest

from gqa_probe.attention import attention_output, page_mass


def test_page_mass_handles_partial_page():
    logits = np.zeros((2, 5))
    np.testing.assert_allclose(page_mass(logits, 2), [[.4, .4], [.4, .4], [.2, .2]])


def test_full_pages_and_sparse_output_use_same_values():
    logits = np.array([[0., 0., 0.], [0., np.log(2), 0.]])
    values = np.array([[0., 3.], [4., 0.], [2., 1.]])
    dense = attention_output(logits, values)
    np.testing.assert_allclose(dense, [[2., 4 / 3], [2.5, 1.]])
    np.testing.assert_allclose(attention_output(logits, values, [0, 1], 2), dense)
    np.testing.assert_allclose(attention_output(logits, values, [1], 2), [[2, 1], [2, 1]])


def test_sparse_softmax_survives_dense_mass_underflow():
    logits = np.array([[10000., -10000., -10001.]])
    values = np.array([[100.], [2.], [2.]])
    assert page_mass(logits, 1)[1, 0] == 0
    np.testing.assert_allclose(attention_output(logits, values, [1, 2], 1), [[2.]])


@pytest.mark.parametrize("logits", [np.empty((1, 0)), np.array([[np.nan]]), np.array([[-np.inf]])])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError):
        page_mass(logits, 2)


def test_empty_selection_and_invalid_page_rejected():
    for pages in [[], [-1], [3]]:
        with pytest.raises(ValueError):
            attention_output(np.zeros((1, 3)), np.ones((3, 2)), pages, 2)
