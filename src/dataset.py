"""
dataset.py
----------
"과거 N개의 앱 사용 이력을 바탕으로 각 앱의 재실행 거리를 예측" (논문 2.2절)
부분을 구현하기 위한 슬라이딩 윈도우 데이터셋 생성기.

각 학습 샘플:
    입력  X_i = (app_ids[i-N+1], ..., app_ids[i])   # 길이 N의 최근 이력, 마지막 원소가 '방금 실행된 앱'
    타겟  y_i = relaunch_distance(i)                # 그 앱이 다시 실행되기까지의 재실행 거리
    가중치 mask_i = 1 (유효) / 0 (censored, 학습 제외)
"""
from __future__ import annotations
import numpy as np


def build_windows(app_ids: np.ndarray, distances: np.ndarray, valid_mask: np.ndarray,
                    window: int = 10):
    T = len(app_ids)
    xs, ys, ms = [], [], []
    for i in range(window - 1, T):
        xs.append(app_ids[i - window + 1: i + 1])
        ys.append(0.0 if np.isnan(distances[i]) else distances[i])
        ms.append(1.0 if valid_mask[i] else 0.0)
    return np.stack(xs), np.array(ys, dtype=np.float64), np.array(ms, dtype=np.float64)


def train_val_split(X, y, m, val_ratio=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], y[train_idx], m[train_idx]), (X[val_idx], y[val_idx], m[val_idx])
