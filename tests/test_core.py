"""Tests for a_hat_optimizer."""

import numpy as np
import pytest


def test_ahat_creation():
    from a_hat_optimizer import AHat
    direction = np.random.randn(4096).astype(np.float32)
    a_hat = AHat(direction, threshold=0.5)
    assert a_hat.hidden_dim == 4096
    assert abs(np.linalg.norm(a_hat.direction) - 1.0) < 1e-5


def test_ahat_predict():
    from a_hat_optimizer import AHat
    direction = np.zeros(128, dtype=np.float32)
    direction[0] = 1.0
    a_hat = AHat(direction, threshold=0.0)

    h_positive = np.zeros(128, dtype=np.float32)
    h_positive[0] = 1.0
    should_call, conf = a_hat.predict(h_positive)
    assert should_call is True
    assert conf > 0

    h_negative = np.zeros(128, dtype=np.float32)
    h_negative[0] = -1.0
    should_call, conf = a_hat.predict(h_negative)
    assert should_call is False
    assert conf < 0


def test_ahat_batch():
    from a_hat_optimizer import AHat
    direction = np.random.randn(64).astype(np.float32)
    a_hat = AHat(direction, threshold=0.0)

    H = np.random.randn(20, 64).astype(np.float32)
    should_call, confidences = a_hat.predict_batch(H)
    assert should_call.shape == (20,)
    assert confidences.shape == (20,)


def test_ahat_save_load(tmp_path):
    from a_hat_optimizer import AHat
    direction = np.random.randn(256).astype(np.float32)
    original = AHat(direction, threshold=1.23, metadata={"model": "test"})
    original.save(tmp_path / "a_hat")

    loaded = AHat.from_file(tmp_path / "a_hat")
    assert loaded.hidden_dim == 256
    assert abs(loaded.threshold - 1.23) < 1e-6
    assert np.allclose(original.direction, loaded.direction, atol=1e-6)


def test_ahat_from_traces():
    from a_hat_optimizer import AHat
    np.random.seed(42)
    n = 100
    dim = 64
    H = np.random.randn(n, dim).astype(np.float32)
    labels = np.zeros(n)
    # Make tool-call states have higher values in dim 0
    H[:50, 0] += 3.0
    labels[:50] = 1

    a_hat = AHat.from_traces(H, labels, calibrate=True)
    assert a_hat.metadata["auc"] > 0.9
    # Should correctly predict the separation
    should_call, _ = a_hat.predict(H[0])  # tool state
    assert should_call is True
    should_call, _ = a_hat.predict(H[99])  # no-tool state
    assert should_call is False


def test_calibrator_strategies():
    from a_hat_optimizer.calibrator import AHatCalibrator
    np.random.seed(42)
    direction = np.zeros(32, dtype=np.float32)
    direction[0] = 1.0
    
    H = np.random.randn(200, 32).astype(np.float32)
    labels = np.zeros(200)
    H[:100, 0] += 2.0
    labels[:100] = 1

    cal = AHatCalibrator(direction)
    
    t_mid = cal.calibrate(H, labels, strategy="midpoint")
    t_f1 = cal.calibrate(H, labels, strategy="f1")
    t_you = cal.calibrate(H, labels, strategy="youden")
    t_pct = cal.calibrate(H, labels, strategy="percentile", percentile=95)
    
    # All should be somewhere reasonable
    assert -5 < t_mid < 5
    assert -5 < t_f1 < 5
    assert -5 < t_you < 5
    assert -5 < t_pct < 5


def test_calibrator_sweep():
    from a_hat_optimizer.calibrator import AHatCalibrator
    np.random.seed(42)
    direction = np.zeros(32, dtype=np.float32)
    direction[0] = 1.0
    
    H = np.random.randn(100, 32).astype(np.float32)
    labels = np.zeros(100)
    H[:50, 0] += 2.0
    labels[:50] = 1

    cal = AHatCalibrator(direction)
    sweep = cal.sweep(H, labels)
    
    assert "thresholds" in sweep
    assert "f1" in sweep
    assert "best_f1_threshold" in sweep
    assert len(sweep["thresholds"]) == 100


def test_npy_load(tmp_path):
    from a_hat_optimizer import AHat
    direction = np.random.randn(512).astype(np.float32)
    np.save(tmp_path / "direction.npy", direction)
    
    a_hat = AHat.from_file(tmp_path / "direction.npy")
    assert a_hat.hidden_dim == 512
    assert a_hat.threshold == 0.0
