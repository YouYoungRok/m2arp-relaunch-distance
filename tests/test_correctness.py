"""
tests/test_correctness.py
--------------------------
이 프로젝트의 핵심 주장 두 가지를 실제로 검증하는 테스트.

1. test_lstm_gradient_check
   NumPy로 직접 구현한 LSTM의 BPTT(역전파) 미분이 올바른지, 수치미분
   (finite-difference)과 비교하여 검증한다.

2. test_m2arp_reduces_to_lru
   Lee & Park (2023) 4.4절의 핵심 주장 — "M2ARP에서 모든 예측 재실행
   거리를 0으로 두면 기존 LRU와 완전히 동일하게 동작한다" — 가 본
   구현(Algorithm 1)에서 실제로 성립하는지 검증한다.

실행 방법
    cd src && python3 -m pytest ../tests -q
    (pytest이 없다면: cd tests && python3 test_correctness.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from lstm_numpy import NumpyLSTMRegressor
from trace_generator import generate_trace, TraceConfig
from policies import validate_lru_equivalence


def test_lstm_gradient_check():
    rng = np.random.default_rng(0)
    n_apps, ed, hd, T, B = 6, 3, 4, 5, 3
    model = NumpyLSTMRegressor(n_apps, ed, hd, seed=1)
    X = rng.integers(0, n_apps, size=(B, T))
    target = rng.normal(size=B)
    mask = np.array([1.0, 1.0, 0.0])  # censored 샘플 포함

    y_hat, cache = model.forward(X)
    loss, dy_hat = model.smse_loss(y_hat, target, mask)
    grads = model.backward(X, cache, dy_hat)

    eps = 1e-5
    max_rel_err = 0.0
    for name in ["Wz", "Wout"]:
        p = model.params()[name]
        flat = p.reshape(-1)
        g_flat = grads[name].reshape(-1)
        idxs = rng.choice(len(flat), size=min(10, len(flat)), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            y1, _ = model.forward(X); l1, _ = model.smse_loss(y1, target, mask)
            flat[idx] = orig - eps
            y2, _ = model.forward(X); l2, _ = model.smse_loss(y2, target, mask)
            flat[idx] = orig
            num_grad = (l1 - l2) / (2 * eps)
            ana_grad = g_flat[idx]
            rel_err = abs(num_grad - ana_grad) / max(1e-6, abs(num_grad) + abs(ana_grad))
            max_rel_err = max(max_rel_err, rel_err)

    print(f"[test_lstm_gradient_check] max relative error = {max_rel_err:.2e}")
    assert max_rel_err < 1e-3, f"BPTT gradient mismatch: {max_rel_err:.2e}"


def test_m2arp_reduces_to_lru():
    app_ids, _ = generate_trace(TraceConfig(n_days=20, seed=1))
    ok = validate_lru_equivalence(app_ids, capacity=6)
    print(f"[test_m2arp_reduces_to_lru] all-zero-prediction == LRU: {ok}")
    assert ok, "Algorithm 1 구현이 논문의 LRU-동치성 성질을 만족하지 않습니다."


if __name__ == "__main__":
    test_lstm_gradient_check()
    print("PASSED: test_lstm_gradient_check")
    test_m2arp_reduces_to_lru()
    print("PASSED: test_m2arp_reduces_to_lru")
    print("\nAll tests passed.")
