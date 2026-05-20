"""Unit tests for training/evaluate.py (tasks 5.2, 5.4, 5.5)."""
from __future__ import annotations

import numpy as np
import pytest

from training.evaluate import compute_bf_score


# --- Task 5.2: BF Score edge cases ---

def test_bf_score_identical_masks():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:50, 10:50] = True
    score = compute_bf_score([mask], [mask])
    assert score >= 0.99, f"Expected ≥0.99 for identical masks, got {score}"


def test_bf_score_inverse_masks():
    mask = np.zeros((64, 64), dtype=bool)
    mask[5:30, 5:30] = True
    inverse = ~mask
    score = compute_bf_score([mask], [inverse])
    assert score < 0.5, f"Expected <0.5 for inverse masks, got {score}"


def test_bf_score_range():
    mask_a = np.zeros((64, 64), dtype=bool)
    mask_a[10:40, 10:40] = True
    mask_b = np.zeros((64, 64), dtype=bool)
    mask_b[20:50, 20:50] = True
    score = compute_bf_score([mask_a], [mask_b])
    assert 0.0 <= score <= 1.0, f"BF Score out of [0,1]: {score}"


def test_bf_score_empty_list():
    score = compute_bf_score([], [])
    assert score == 0.0


def test_bf_score_all_empty_masks():
    empty = np.zeros((64, 64), dtype=bool)
    # No boundary pixels in both → should handle gracefully
    score = compute_bf_score([empty], [empty])
    assert 0.0 <= score <= 1.0
