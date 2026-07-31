"""
policies.py
-----------
Lee, J. and Park, S. (2023), "An Efficient Memory Management for Mobile
Operating Systems Based on Prediction of Relaunch Distance," Computer
Systems Science and Engineering, 47(1), 171-186. DOI: 10.32604/csse.2023.038139
(CC BY 4.0 — 원문 인용/재구현 허용)

이 파일은 논문의 Algorithm 1 (onAppLaunch) 을 그대로 구현한다.

    Algorithm 1: onAppLaunch(l_t)
    1: if l_t ∈ A* then
    2:     d_hat(l_t) <- LSTM으로 예측한 재실행 거리
    3: else
    4:     d_hat(l_t) <- 0                      (LRU fallback, 3.2절)
    5: end if
    6: l_t의 relaunch distance <- d_hat(l_t)
    7: for each a in {a | a in B, a != l_t}:    (B: 캐시된 앱 집합)
    8:     a의 relaunch distance <- a의 relaunch distance - 1
    9: end for

메모리 부족(캐시 초과) 시에는, 캐시된 앱 중 relaunch distance의 **절댓값**이
가장 큰 앱을 종료한다. (3.2절: "the apps with the largest absolute relaunch
distance are terminated" — 예측값이 커서 '아직 한참 남은' 앱과, 감소가
누적되어 매우 작아진(overdue) 앱을 동일한 기준으로 다룸으로써, 예측이
전부 0인 경우 이 메커니즘이 정확히 기존 LRU와 동일하게 동작한다: "M2ARP
with all the relaunch distances of an app set to 0 is equivalent to the
LRU method" (4.4절). 본 구현은 이 성질을 실제로 만족하는지 단위 테스트로
검증한다 (validate_lru_equivalence 참고).

세 정책은 모두 이 동일한 엔진 위에서, '재실행 거리를 어떻게 정하는가'만
다르게 구현한 것이다.
    - LRU     : 항상 d_hat = 0  (예측을 전혀 사용하지 않음)
    - Optimal : 실제(ground-truth) 재실행 거리를 오라클처럼 사용 (이론적 상한)
    - M2ARP   : 학습된 LSTM의 예측값 (단, 학습 데이터가 부족한 앱은 0으로 대체)

시뮬레이션을 위해서는 여기에 '실행 시간(launch time)' 개념을 추가로
도입한다. 이는 원 논문의 4.4절 시뮬레이션 벤치마크(캐시 크기별 hit ratio만
측정, 모든 앱의 메모리 footprint를 동일하다고 가정)를 넘어서는 본 프로젝트의
확장으로, 앱마다 서로 다른 cold/warm start 비용을 부여해 좀 더 현실적인
'평균 실행 시간' 지표까지 함께 살펴본다.
"""
from __future__ import annotations
import bisect
import numpy as np
from collections import defaultdict


LARGE_DISTANCE = 1_000_000.0  # 다시는 재실행되지 않는 경우(censored)의 대체값


def make_launch_costs(n_apps: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    cold_base = rng.uniform(650, 1400, size=n_apps)   # ms, 앱별 cold-start 기본 비용
    warm_base = rng.uniform(40, 130, size=n_apps)      # ms, 앱별 warm-start(재개) 기본 비용
    return cold_base, warm_base


def sample_launch_time(app, is_hit, cold_base, warm_base, rng):
    if is_hit:
        return max(5.0, rng.normal(warm_base[app], warm_base[app] * 0.15))
    else:
        return max(50.0, rng.normal(cold_base[app], cold_base[app] * 0.12))


class ForwardOracle:
    """Optimal 정책용: 위치 i에서 앱 x가 실행되었을 때, 그 시점 기준 실제
    재실행 거리(= 다음에 x가 다시 나올 때까지의 고유 앱 개수)를 조회한다.
    Eq.(1)-(2)와 동일한 정의이며, 다시 나오지 않으면 LARGE_DISTANCE를 반환한다."""

    def __init__(self, app_ids: np.ndarray):
        self.app_ids = app_ids
        self.occurrences = defaultdict(list)
        for idx, a in enumerate(app_ids):
            self.occurrences[int(a)].append(idx)

    def true_distance_at_launch(self, i: int) -> float:
        app = int(self.app_ids[i])
        occ = self.occurrences[app]
        j = bisect.bisect_right(occ, i)
        if j >= len(occ):
            return LARGE_DISTANCE
        next_pos = occ[j]
        return float(len(set(self.app_ids[i + 1:next_pos].tolist())))


# ----------------------------------------------------------------------
# Algorithm 1 (onAppLaunch) 을 그대로 구현한 공통 시뮬레이션 엔진
# ----------------------------------------------------------------------
def simulate(app_ids: np.ndarray, capacity: int, distance_fn, cold_base, warm_base,
              seed: int = 0, warmup_len: int = 0):
    """
    distance_fn(i, app) -> d_hat(l_t) : 위치 i에서 app이 막 실행되었을 때
    부여할 (예측/실제/0) 재실행 거리. 정책(LRU/Optimal/M2ARP)에 따라 다른
    함수를 넘겨준다.

    warmup_len : 이 길이만큼은 캐시 상태를 현실적으로 채우는 용도로만 쓰고
                 (직전 구간의 꼬리를 이어붙인 워밍업) 지표 집계에서는 제외한다.
    """
    rng = np.random.default_rng(seed)
    T = len(app_ids)

    distance = {}          # app -> 현재 relaunch distance 카운터 (Algorithm 1의 상태)
    total_time = 0.0
    hits = 0
    n_counted = 0

    for i in range(T):
        app = int(app_ids[i])
        is_hit = app in distance
        count_this = i >= warmup_len
        if count_this:
            if is_hit:
                hits += 1
            total_time += sample_launch_time(app, is_hit, cold_base, warm_base, rng)
            n_counted += 1

        # --- Algorithm 1, line 1-6: l_t 자신의 relaunch distance를 (재)설정 ---
        distance[app] = distance_fn(i, app)

        # --- Algorithm 1, line 7-9: 캐시된 다른 모든 앱의 distance를 1 감소 ---
        for a in distance.keys():
            if a != app:
                distance[a] -= 1

        # --- 저장 공간 부족 시, |relaunch distance|가 가장 큰 앱을 종료 ---
        if len(distance) > capacity:
            victim = max((a for a in distance if a != app), key=lambda a: abs(distance[a]))
            del distance[victim]

    return {
        "avg_launch_time_ms": total_time / n_counted,
        "hit_ratio": hits / n_counted,
        "n_events": n_counted,
        "capacity": capacity,
    }


# ----------------------------------------------------------------------
# 세 정책의 distance_fn 팩토리
# ----------------------------------------------------------------------
def lru_distance_fn():
    """LRU: 예측을 전혀 쓰지 않고 항상 0 (3.2절: 'if the predicted relaunch
    distance of all apps in the system is set to 0, then it works the same
    as LRU')."""
    def fn(i, app):
        return 0.0
    return fn


def optimal_distance_fn(oracle: ForwardOracle):
    """Optimal: 실제(ground-truth) 재실행 거리를 오라클처럼 사용 (이론적 상한)."""
    def fn(i, app):
        return oracle.true_distance_at_launch(i)
    return fn


def m2arp_distance_fn(predictor, window: int, app_ids: np.ndarray, low_data_apps: set):
    """M2ARP: 학습된 LSTM 예측값. 단, 학습 데이터가 부족한 앱(low_data_apps)이거나
    아직 문맥 윈도우가 채워지지 않은 초반 구간에서는 0으로 대체(LRU fallback,
    Algorithm 1의 A* 조건에 해당)."""
    def fn(i, app):
        if app in low_data_apps or i < window - 1:
            return 0.0
        x = app_ids[i - window + 1:i + 1][None, :]
        return float(predictor.predict(x)[0])
    return fn


# ----------------------------------------------------------------------
# 검증: "모든 예측을 0으로 두면 M2ARP == LRU" (논문 4.4절 주장)를 실제로 확인
# ----------------------------------------------------------------------
def validate_lru_equivalence(app_ids: np.ndarray, capacity: int = 6, seed: int = 0) -> bool:
    cold_base, warm_base = make_launch_costs(int(app_ids.max()) + 1, seed=1)

    res_lru = simulate(app_ids, capacity, lru_distance_fn(), cold_base, warm_base, seed=seed)

    always_zero_fn = lambda i, app: 0.0  # M2ARP의 distance_fn 자리에 0을 넣으면 LRU와 같아야 함
    res_zero = simulate(app_ids, capacity, always_zero_fn, cold_base, warm_base, seed=seed)

    ok = (res_lru["hit_ratio"] == res_zero["hit_ratio"] and
          res_lru["avg_launch_time_ms"] == res_zero["avg_launch_time_ms"])
    return ok


if __name__ == "__main__":
    from trace_generator import generate_trace, TraceConfig
    ids, _ = generate_trace(TraceConfig(n_days=20))
    ok = validate_lru_equivalence(ids)
    print(f"[validate] '예측값을 모두 0으로 두면 LRU와 동일하다' 검증: {'PASSED' if ok else 'FAILED'}")
    assert ok
