"""
run_experiment_torch.py
------------------------
run_experiment.py와 동일한 실험(트레이스 생성 -> 예측기 학습 -> 캐시 크기별
LRU/Optimal/M2ARP 시뮬레이션)을, 예측기 부분만 PyTorch(lstm_torch.py)로
바꿔서 수행하는 버전이다. 결과는 검증된 NumPy 버전의 결과를
덮어쓰지 않도록 results/torch/ 하위에 따로 저장한다.
실행 전: pip install torch pandas matplotlib
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trace_generator import generate_trace, TraceConfig
from relaunch import compute_relaunch_distances
from dataset import build_windows, train_val_split
from lstm_torch import TorchLSTMRegressor
from policies import (simulate, ForwardOracle, make_launch_costs,
                        lru_distance_fn, optimal_distance_fn, m2arp_distance_fn,
                        validate_lru_equivalence)
from train_predictor import iterate_minibatches
from train_predictor_torch import evaluate, _selective_mse_numpy


WINDOW = 10
EMBED_DIM = 16
HIDDEN_DIM = 32
EPOCHS = 30
LR = 0.02
N_DAYS = 150
MAX_DIST = 50
TRAIN_FRAC = 0.7
LOW_DATA_THRESHOLD = 6
CACHE_SIZES = list(range(5, 16))
SEED = 0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "results", "torch")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 60)
    print("0) 검증: 예측이 전부 0이면 M2ARP == LRU (프레임워크와 무관한 성질)")
    print("=" * 60)
    tmp_ids, _ = generate_trace(TraceConfig(n_days=20))
    ok = validate_lru_equivalence(tmp_ids)
    print(f"   -> {'PASSED' if ok else 'FAILED'}")
    assert ok

    print("\n" + "=" * 60)
    print("1) 합성 트레이스 생성")
    print("=" * 60)
    cfg = TraceConfig(n_days=N_DAYS, seed=SEED)
    app_ids, hours = generate_trace(cfg)
    T = len(app_ids)
    split = int(T * TRAIN_FRAC)
    train_ids, eval_ids = app_ids[:split], app_ids[split:]
    print(f"전체 길이={T}  train={len(train_ids)}  eval={len(eval_ids)}  n_apps={cfg.n_apps}")

    print("\n" + "=" * 60)
    print("2) train 구간으로 재실행 거리 예측기(PyTorch LSTM) 학습")
    print("=" * 60)
    distances, valid_mask = compute_relaunch_distances(train_ids, max_distance=MAX_DIST)
    X, y, m = build_windows(train_ids, distances, valid_mask, window=WINDOW)
    (Xtr, ytr, mtr), (Xval, yval, mval) = train_val_split(X, y, m, val_ratio=0.15, seed=SEED)

    mean_pred_val = np.full(len(yval), ytr[mtr.astype(bool)].mean())
    baseline_val_smse = _selective_mse_numpy(mean_pred_val, yval, mval)

    model = TorchLSTMRegressor(cfg.n_apps, EMBED_DIM, HIDDEN_DIM, seed=SEED)
    rng = np.random.default_rng(SEED)
    best_val, best_state, best_epoch = float("inf"), None, 0
    history = []
    for ep in range(1, EPOCHS + 1):
        losses = []
        for xb, yb, mb in iterate_minibatches(Xtr, ytr, mtr, 64, rng):
            if mb.sum() == 0:
                continue
            losses.append(model.train_step(xb, yb, mb, lr=LR))
        val_loss, val_mae = evaluate(model, Xval, yval, mval)
        history.append({"epoch": ep, "train_smse": float(np.mean(losses)),
                          "val_smse": float(val_loss), "val_mae": float(val_mae)})
        if val_loss < best_val:
            best_val, best_epoch = val_loss, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep:2d}  train_SMSE={np.mean(losses):7.2f}  val_SMSE={val_loss:7.2f}  "
              f"val_MAE={val_mae:5.2f}" + ("  *best*" if ep == best_epoch else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"-> best epoch={best_epoch}, val_SMSE={best_val:.2f} "
          f"(mean-predictor baseline val_SMSE={baseline_val_smse:.2f})")
    model.save(os.path.join(out_dir, "predictor_torch.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    print("\n" + "=" * 60)
    print("3) 앱별 학습 데이터 충분성 판정 (LRU Fallback)")
    print("=" * 60)
    vals, counts = np.unique(train_ids, return_counts=True)
    freq = dict(zip(vals.tolist(), counts.tolist()))
    low_data_apps = {a for a in range(cfg.n_apps) if freq.get(a, 0) < LOW_DATA_THRESHOLD}
    print(f"학습 데이터 부족(< {LOW_DATA_THRESHOLD}회) 앱: {len(low_data_apps)}/{cfg.n_apps}개")

    print("\n" + "=" * 60)
    print(f"4) eval 구간에서 캐시 용량 m={CACHE_SIZES[0]}..{CACHE_SIZES[-1]} 시뮬레이션")
    print("=" * 60)
    cold_base, warm_base = make_launch_costs(cfg.n_apps, seed=SEED + 1)

    warmup_ctx = train_ids[-(WINDOW - 1):] if WINDOW > 1 else np.array([], dtype=np.int64)
    sim_ids = np.concatenate([warmup_ctx, eval_ids])
    warmup_len = len(warmup_ctx)
    oracle = ForwardOracle(sim_ids)

    dfns = {
        "lru": lru_distance_fn(),
        "optimal": optimal_distance_fn(oracle),
        "m2arp": m2arp_distance_fn(model, WINDOW, sim_ids, low_data_apps),
    }

    rows = []
    for cap in CACHE_SIZES:
        for policy, dfn in dfns.items():
            res = simulate(sim_ids, capacity=cap, distance_fn=dfn,
                             cold_base=cold_base, warm_base=warm_base,
                             seed=SEED + cap, warmup_len=warmup_len)
            res["policy"] = policy
            rows.append(res)
            print(f"  m={cap:3d}  {policy:8s}  "
                  f"hit_ratio={res['hit_ratio']*100:5.1f}%  "
                  f"avg_launch={res['avg_launch_time_ms']:7.1f}ms")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comparison_table.csv"), index=False)

    hit_table = df.pivot(index="capacity", columns="policy", values="hit_ratio")[["lru", "m2arp", "optimal"]]
    hit_table.columns = ["LRU", "M2ARP", "Optimal"]
    hit_table = (hit_table * 100).round(1)
    hit_table.index.name = "m"
    hit_table.to_csv(os.path.join(out_dir, "hit_ratio_at_m_table.csv"))
    print("\nHit ratio@m (%):")
    print(hit_table.to_string())

    time_table = df.pivot(index="capacity", columns="policy", values="avg_launch_time_ms")[["lru", "m2arp", "optimal"]]
    time_table.columns = ["LRU", "M2ARP", "Optimal"]
    time_table = time_table.round(1)
    time_table.index.name = "m"
    time_table.to_csv(os.path.join(out_dir, "launch_time_at_m_table.csv"))

    print("\n" + "=" * 60)
    print("5) 그래프 저장")
    print("=" * 60)
    colors = {"LRU": "#9aa0a6", "M2ARP": "#4c8bf5", "Optimal": "#34a853"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in ["LRU", "M2ARP", "Optimal"]:
        ax.plot(hit_table.index, hit_table[policy], marker="o", color=colors[policy], label=policy)
    ax.set_xlabel("Cache size m (# background apps)")
    ax.set_ylabel("App launch hit ratio (%)")
    ax.set_title("App Launch Hit Ratio @ m  (PyTorch backend)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "hit_ratio_at_m.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for policy in ["LRU", "M2ARP", "Optimal"]:
        ax.plot(time_table.index, time_table[policy], marker="o", color=colors[policy], label=policy)
    ax.set_xlabel("Cache size m (# background apps)")
    ax.set_ylabel("Average launch time (ms)")
    ax.set_title("Average App Launch Time @ m  (PyTorch backend)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "launch_time_at_m.png"), dpi=150)
    plt.close(fig)

    hist_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist_df.epoch, hist_df.train_smse, label="train SMSE")
    ax.plot(hist_df.epoch, hist_df.val_smse, label="val SMSE")
    ax.axhline(baseline_val_smse, color="gray", linestyle="--", label="mean-predictor baseline")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SMSE")
    ax.set_title("Relaunch-Distance Predictor Training Curve (PyTorch backend)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({
            "backend": "pytorch", "window": WINDOW, "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM, "epochs": EPOCHS, "lr": LR, "n_days": N_DAYS,
            "n_apps": cfg.n_apps, "max_distance": MAX_DIST, "train_frac": TRAIN_FRAC,
            "low_data_threshold": LOW_DATA_THRESHOLD, "cache_sizes": CACHE_SIZES,
            "seed": SEED, "trace_length": int(T), "best_epoch": best_epoch,
            "best_val_smse": float(best_val), "baseline_val_smse": float(baseline_val_smse),
            "lru_equivalence_check": ok,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n완료. {out_dir} 디렉터리에 CSV 및 그래프가 저장되었습니다.")


if __name__ == "__main__":
    main()
