import os
import sys

import numpy as np
import pytest

# テスト実行時に同じディレクトリの softmax.py を import できるようにする
here = os.path.dirname(__file__)
if here not in sys.path:
    sys.path.insert(0, here)

from softmax import softmax, softmax_overflow


def test_softmax_sum_1d():
    a = np.array([1010, 1000, 990])
    y = softmax(a)
    assert np.all(np.isfinite(y)), "softmax produced non-finite values"
    assert np.isclose(np.sum(y), 1.0, atol=1e-12)


def test_softmax_stability_vs_overflow():
    a = np.array([1010, 1000, 990])
    y_naive = softmax_overflow(a)
    # ナイーブ実装はオーバーフローで NaN/Inf を含むはず（少なくとも全てが有限ではない）
    assert not np.all(np.isfinite(y_naive))
    y = softmax(a)
    assert np.all(np.isfinite(y))


def test_softmax_against_manual():
    # 小さな入力で手計算と一致すること
    a = np.array([1.0, 2.0, 3.0])
    exp = np.exp(a - np.max(a))
    expected = exp / np.sum(exp)
    y = softmax(a)
    np.testing.assert_allclose(y, expected, rtol=1e-7, atol=0)
