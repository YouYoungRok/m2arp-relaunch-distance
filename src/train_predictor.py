"""
train_predictor.py
-------------------
합성 트레이스로부터 (윈도우, 타겟) 학습 데이터를 만들고 NumpyLSTMRegressor를
SMSE 손실로 학습한 뒤, 학습된 가중치를 저장한다.
"""
from __future__ import annotations
import os
import time
import numpy as np

from trace_generator import generate_trace, TraceConfig
from relaunch import compute_relaunch_distances
from dataset import build_windows, train_val_split
from lstm_numpy import NumpyLSTMRegressor

# 이 스크립트 파일 위치를 기준으로 절대경로를 계산한다.
# (터미널 cwd가 src/든 프로젝트 루트든, VS Code의 "Run" 버튼으로 실행하든
#  항상 동일하게 동작하도록 하기 위함 — 상대경로 "../results"는 cwd에 따라
#  깨질 수 있다.)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT_PATH = os.path.join(_SCRIPT_DIR, "..", "results", "predictor.npz")


def iterate_minibatches(X, y, m, batch_size, rng):
    n = len(X)
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b], m[b]


def evaluate(model, X, y, m):
    y_hat, _ = model.forward(X)
    loss, _ = model.smse_loss(y_hat, y, m)
    # 참고 지표: 유효 샘플에 대한 MAE
    valid = m.astype(bool)
    mae = np.mean(np.abs(y_hat[valid] - y[valid])) if valid.sum() > 0 else float("nan")
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

    print("2) 정답 재실행 거리(ground-truth relaunch distance) 계산 중...")
    distances, valid_mask = compute_relaunch_distances(app_ids, max_distance=max_distance)
    print(f"   valid target ratio = {valid_mask.mean()*100:.1f}%")

    print("3) 슬라이딩 윈도우 데이터셋 구성 중...")
    X, y, m = build_windows(app_ids, distances, valid_mask, window=window)
    (Xtr, ytr, mtr), (Xval, yval, mval) = train_val_split(X, y, m, val_ratio=0.15, seed=seed)
    print(f"   train={len(Xtr)}, val={len(Xval)}")

    # baseline: 학습셋 평균 재실행 거리로만 예측했을 때의 SMSE (모델 성능 비교 기준선)
    mean_pred_val = np.full(len(yval), ytr[mtr.astype(bool)].mean())
    baseline_loss, _ = NumpyLSTMRegressor.smse_loss(mean_pred_val, yval, mval)
    print(f"   baseline(mean predictor) val SMSE = {baseline_loss:.3f}")

    model = NumpyLSTMRegressor(cfg.n_apps, embed_dim, hidden_dim, seed=seed)
    rng = np.random.default_rng(seed)

    print("4) 학습 시작 (validation SMSE 기준 early stopping 적용)...")
    t0 = time.time()
    history = []
    best_val = float("inf")
    best_params = {k: v.copy() for k, v in model.params().items()}
    best_epoch = 0
    for ep in range(1, epochs + 1):
        ep_losses = []
        for xb, yb, mb in iterate_minibatches(Xtr, ytr, mtr, batch_size, rng):
            if mb.sum() == 0:
                continue
            loss = model.train_step(xb, yb, mb, lr=lr)
            ep_losses.append(loss)
        val_loss, val_mae = evaluate(model, Xval, yval, mval)
        history.append((ep, np.mean(ep_losses), val_loss, val_mae))
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_params = {k: v.copy() for k, v in model.params().items()}
            best_epoch = ep
            marker = "  *best*"
        print(f"   epoch {ep:2d}  train_SMSE={np.mean(ep_losses):.3f}  "
              f"val_SMSE={val_loss:.3f}  val_MAE={val_mae:.3f}{marker}")
    print(f"   학습 완료 ({time.time()-t0:.1f}s) — best epoch={best_epoch}, best val_SMSE={best_val:.3f}")

    # 최적(early-stopped) 가중치로 복원
    for k, v in best_params.items():
        model.params()[k][...] = v

    model.save(out_path)
    print(f"5) 모델 저장 완료 -> {out_path}")
    return model, cfg, history, baseline_loss


if __name__ == "__main__":
    main()
