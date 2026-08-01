"""
train_predictor_torch.py
-------------------------
train_predictor.py와 동일한 파이프라인이지만, NumPy 대신 lstm_torch.py의
PyTorch 모델을 사용한다. (선택적 대안 — README의 "PyTorch 버전" 절 참고.)

실행 전: pip install torch
"""
from __future__ import annotations
import os
import time
import numpy as np

from trace_generator import generate_trace, TraceConfig
from relaunch import compute_relaunch_distances
from dataset import build_windows, train_val_split
from lstm_torch import TorchLSTMRegressor
from train_predictor import iterate_minibatches  # numpy 버전과 동일한 미니배치 유틸 재사용

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT_PATH = os.path.join(_SCRIPT_DIR, "..", "results", "torch", "predictor_torch.pt")


def _selective_mse_numpy(y_hat, target, mask):
    """baseline(평균값 예측기) 비교용 — 순수 numpy로 계산."""
    mask = mask.astype(np.float64)
    n_valid = max(mask.sum(), 1.0)
    diff = (y_hat - target) * mask
    return float(np.sum(diff ** 2) / n_valid)


def evaluate(model, X, y, m):
    y_hat = model.predict(X)
    loss = _selective_mse_numpy(y_hat, y, m)
    valid = m.astype(bool)
    mae = float(np.mean(np.abs(y_hat[valid] - y[valid]))) if valid.sum() > 0 else float("nan")
    return loss, mae


def main(window=10, embed_dim=16, hidden_dim=32, epochs=20, batch_size=64, lr=0.01,
          n_days=150, max_distance=50, seed=0, out_path=None):
    if out_path is None:
        out_path = _DEFAULT_OUT_PATH
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print("1) 합성 트레이스 생성 중...")
    cfg = TraceConfig(n_days=n_days, seed=seed)
    app_ids, hours = generate_trace(cfg)
    print(f"   trace length = {len(app_ids)}, n_apps = {cfg.n_apps}")

    print("2) 정답 재실행 거리 계산 중...")
    distances, valid_mask = compute_relaunch_distances(app_ids, max_distance=max_distance)
    print(f"   valid target ratio = {valid_mask.mean()*100:.1f}%")

    print("3) 슬라이딩 윈도우 데이터셋 구성 중...")
    X, y, m = build_windows(app_ids, distances, valid_mask, window=window)
    (Xtr, ytr, mtr), (Xval, yval, mval) = train_val_split(X, y, m, val_ratio=0.15, seed=seed)
    print(f"   train={len(Xtr)}, val={len(Xval)}")

    mean_pred_val = np.full(len(yval), ytr[mtr.astype(bool)].mean())
    baseline_loss = _selective_mse_numpy(mean_pred_val, yval, mval)
    print(f"   baseline(mean predictor) val SMSE = {baseline_loss:.3f}")

    model = TorchLSTMRegressor(cfg.n_apps, embed_dim, hidden_dim, seed=seed)
    rng = np.random.default_rng(seed)

    print("4) 학습 시작 (PyTorch, validation SMSE 기준 early stopping)...")
    t0 = time.time()
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    for ep in range(1, epochs + 1):
        ep_losses = []
        for xb, yb, mb in iterate_minibatches(Xtr, ytr, mtr, batch_size, rng):
            if mb.sum() == 0:
                continue
            ep_losses.append(model.train_step(xb, yb, mb, lr=lr))
        val_loss, val_mae = evaluate(model, Xval, yval, mval)
        marker = ""
        if val_loss < best_val:
            best_val, best_epoch = val_loss, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  *best*"
        print(f"   epoch {ep:2d}  train_SMSE={np.mean(ep_losses):.3f}  "
              f"val_SMSE={val_loss:.3f}  val_MAE={val_mae:.3f}{marker}")
    print(f"   학습 완료 ({time.time()-t0:.1f}s) — best epoch={best_epoch}, best val_SMSE={best_val:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save(out_path)
    print(f"5) 모델 저장 완료 -> {out_path}")
    return model, cfg


if __name__ == "__main__":
    main()
