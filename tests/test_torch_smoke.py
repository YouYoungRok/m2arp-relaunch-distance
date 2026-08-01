import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def test_shapes_and_forward():
    from lstm_torch import TorchLSTMRegressor
    n_apps, ed, hd, T, B = 10, 8, 16, 6, 4
    model = TorchLSTMRegressor(n_apps, ed, hd, seed=0)
    rng = np.random.default_rng(0)
    X = rng.integers(0, n_apps, size=(B, T))
    y_hat = model.predict(X)
    assert y_hat.shape == (B,), f"예상 shape (B,)={B} 인데 실제 {y_hat.shape}"
    assert (y_hat >= 0).all(), "재실행 거리 예측값은 0 이상이어야 합니다"
    print("[test_shapes_and_forward] OK — output shape:", y_hat.shape)


def test_loss_decreases_on_tiny_overfit():
    """아주 작은 고정 배치를 여러 번 학습시키면 SMSE가 확실히 줄어야 한다
    (그래야 순전파/역전파/옵티마이저가 제대로 연결되어 있다는 최소한의 증거가 됨)."""
    from lstm_torch import TorchLSTMRegressor
    n_apps, ed, hd, T, B = 8, 6, 12, 5, 16
    model = TorchLSTMRegressor(n_apps, ed, hd, seed=1)
    rng = np.random.default_rng(1)
    X = rng.integers(0, n_apps, size=(B, T))
    target = rng.uniform(0, 10, size=B)
    mask = np.ones(B)

    losses = []
    for _ in range(50):
        loss = model.train_step(X, target, mask, lr=0.05)
        losses.append(loss)

    print(f"[test_loss_decreases_on_tiny_overfit] loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < losses[0] * 0.5, (
        f"50 스텝 학습 후에도 손실이 충분히 줄지 않았습니다 ({losses[0]:.3f} -> {losses[-1]:.3f}). "
        f"forward/backward 연결에 문제가 있을 수 있습니다."
    )


def test_save_load_roundtrip(tmp_path="/tmp/_m2arp_torch_test.pt"):
    from lstm_torch import TorchLSTMRegressor
    n_apps, ed, hd, T, B = 10, 8, 16, 6, 4
    model = TorchLSTMRegressor(n_apps, ed, hd, seed=2)
    rng = np.random.default_rng(2)
    X = rng.integers(0, n_apps, size=(B, T))
    pred_before = model.predict(X)

    model.save(tmp_path)
    loaded = TorchLSTMRegressor.load(tmp_path)
    pred_after = loaded.predict(X)

    assert np.allclose(pred_before, pred_after, atol=1e-5), "저장/로드 후 예측값이 달라졌습니다"
    os.remove(tmp_path)
    print("[test_save_load_roundtrip] OK — save/load 후 예측값 일치")


if __name__ == "__main__":
    test_shapes_and_forward()
    test_loss_decreases_on_tiny_overfit()
    test_save_load_roundtrip()
    print("\nAll torch smoke tests passed.")
