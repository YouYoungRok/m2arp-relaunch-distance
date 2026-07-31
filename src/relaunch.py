"""
relaunch.py
-----------
논문의 정의를 그대로 구현한다.

    현재 시점 t에서 앱 A_i가 실행되었을 때, 향후 앱 A_i가 다시 실행되는
    미래 시점 t' 까지의 사이에 실행되는 서로 다른 고유 앱(unique apps)의
    개수를 재실행 거리 D(A_i)로 정의한다.

    D(A_i) = |{ A_k | t < k < t', A_k != A_i }|   (서로 다른 app_id의 개수, 중복 제거)

트레이스의 끝까지 같은 앱이 다시 실행되지 않는 경우(censored) 는 '유효한
재실행 거리가 없음(None)'으로 표시하고, 학습 시 SMSE 손실에서 제외한다.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict


def compute_relaunch_distances(app_ids: np.ndarray, max_distance: int | None = None):
    """
    Parameters
    ----------
    app_ids : (T,) int array, 시점 순서대로 나열된 앱 실행 시퀀스
    max_distance : 재실행 거리 상한 (학습 안정성을 위한 클리핑). None이면 클리핑 없음.

    Returns
    -------
    distances : (T,) float array. 유효하지 않은 위치는 np.nan
    valid_mask : (T,) bool array. True면 distances[i]가 유효한 학습 타겟
    """
    T = len(app_ids)
    distances = np.full(T, np.nan, dtype=np.float64)
    valid_mask = np.zeros(T, dtype=bool)

    # 각 app_id가 등장하는 인덱스 목록을 미리 구해두면 다음 등장 위치를 O(1)에 조회 가능
    next_occurrence = defaultdict(list)
    for idx, a in enumerate(app_ids):
        next_occurrence[int(a)].append(idx)

    # app별 포인터: 현재 인덱스 이후 '다음 등장 위치'를 순차적으로 찾기 위함
    ptr = {a: 0 for a in next_occurrence}

    for i in range(T):
        a = int(app_ids[i])
        occ = next_occurrence[a]
        # occ[ptr[a]] 는 항상 i 자신이거나 그 이전 위치를 가리키므로 i 다음 값을 찾는다
        p = ptr[a]
        while p < len(occ) and occ[p] <= i:
            p += 1
        ptr[a] = p
        if p < len(occ):
            j = occ[p]  # 다음 재실행 시점
            unique_apps = len(set(app_ids[i + 1:j].tolist()))
            d = float(unique_apps)
            if max_distance is not None:
                d = min(d, float(max_distance))
            distances[i] = d
            valid_mask[i] = True
        # else: 트레이스 끝까지 재실행되지 않음 -> invalid (censored)

    return distances, valid_mask


if __name__ == "__main__":
    from trace_generator import generate_trace, TraceConfig

    ids, hrs = generate_trace(TraceConfig(n_days=10))
    dist, mask = compute_relaunch_distances(ids, max_distance=50)
    print(f"valid targets: {mask.sum()}/{len(mask)} ({mask.mean()*100:.1f}%)")
    print("distance stats (valid only): mean=%.2f, median=%.2f, max=%.2f" % (
        np.nanmean(dist[mask]), np.nanmedian(dist[mask]), np.nanmax(dist[mask])
    ))
