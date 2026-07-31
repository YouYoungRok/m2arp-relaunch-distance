"""
trace_generator.py
-------------------
Lee & Park(2023)의 벤치마크는 Tsinghua 대학의 실제 사용자 앱 사용 로그
(단일 사용자, 93개 앱, 2,601건의 실행 기록)를 사용했다[1]. 해당 데이터셋은
공개적으로 접근할 수 없으므로, 본 재현 프로젝트에서는 현실적인 특성을 모사한
합성(synthetic) 앱 사용 트레이스를 생성해 동일한 방법론(3.2절 relaunch distance
정의, Algorithm 1, 4.4절 시뮬레이션 기반 hit ratio@m 벤치마크)을 적용한다.

실제 사람의 스마트폰 사용은 (1) 매일 반복되는 습관적 루틴(출근길 지도+음악,
아침 알람+날씨+메신저 등)과 (2) 그때그때 달라지는 자유로운 사용이 섞여
나타난다. 두 특성을 모두 반영해야 '재실행 거리 예측'이라는 과제가 의미를
가지므로(완전히 무작위라면 예측 자체가 불가능하다), 아래 두 요소를 혼합한다.

1. 인기도(Zipf) 기반 배경 사용 : 자유 세션에서의 무작위 전환에 적용
2. 시간대별 '루틴 템플릿'      : 각 시간대마다 2~3개의 고정된 앱 시퀀스를
   정의해두고, 매일 약간의 노이즈(치환)를 섞어 재생한다.
   -> LSTM이 단순 최근성이 아니라 '문맥+습관 패턴'을 학습할 여지가 생긴다.

[1] D. Yu, Y. Li, F. Xu, P. Zhang and V. Kostakos, "Smartphone app usage
    prediction using points of interest," Proc. ACM IMWUT, vol. 1, 2018.
    (Lee & Park 2023의 데이터셋 출처, 원 논문 참고문헌 [24])
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class TraceConfig:
    n_apps: int = 40
    n_days: int = 150
    sessions_per_day: tuple = (8, 20)      # 하루 세션 수 범위
    time_slots: int = 4                    # 하루를 4개 시간대로 분할
    templates_per_slot: int = 3            # 시간대별 루틴 템플릿 개수
    template_len: tuple = (3, 6)           # 템플릿 길이 범위
    template_use_p: float = 0.6            # 세션이 '루틴 템플릿'을 따를 확률 (나머지는 자유 사용)
    template_noise_p: float = 0.12         # 템플릿 재생 중 각 단계에서 노이즈(치환)가 낄 확률
    free_session_len: tuple = (3, 10)      # 자유 세션 길이 범위
    continuity_p: float = 0.10             # 자유 사용 중 직전 앱을 다시 쓸 확률
    novelty_p: float = 0.03                # 자유 사용 중 희귀/신규 앱을 실행할 확률
    zipf_s: float = 1.1
    seed: int = 42


def _zipf_popularity(n_apps: int, s: float, rng: np.random.Generator) -> np.ndarray:
    ranks = np.arange(1, n_apps + 1)
    weights = 1.0 / np.power(ranks, s)
    rng.shuffle(weights)
    return weights / weights.sum()


def generate_trace(cfg: TraceConfig = TraceConfig()):
    """
    Returns
    -------
    app_ids : np.ndarray[int]   각 시점에 실행된 앱 ID 시퀀스
    hours   : np.ndarray[int]   각 시점의 시간대(0..time_slots-1)
    """
    rng = np.random.default_rng(cfg.seed)
    base_pop = _zipf_popularity(cfg.n_apps, cfg.zipf_s, rng)

    slot_bias = rng.dirichlet(np.ones(cfg.n_apps) * 0.4, size=cfg.time_slots)
    slot_pop = 0.5 * base_pop[None, :] + 0.5 * slot_bias
    slot_pop = slot_pop / slot_pop.sum(axis=1, keepdims=True)

    # 시간대별 고정 루틴 템플릿 생성 (예: 아침엔 [12, 3, 7], 저녁엔 [22, 9, 22, 15] 등)
    templates = []
    for slot in range(cfg.time_slots):
        slot_templates = []
        for _ in range(cfg.templates_per_slot):
            tlen = rng.integers(cfg.template_len[0], cfg.template_len[1] + 1)
            seq = rng.choice(cfg.n_apps, size=tlen, p=slot_pop[slot], replace=True)
            slot_templates.append(seq)
        templates.append(slot_templates)

    app_ids = []
    hours = []
    last_app = None

    for day in range(cfg.n_days):
        n_sessions = rng.integers(cfg.sessions_per_day[0], cfg.sessions_per_day[1] + 1)
        session_slots = np.sort(rng.integers(0, cfg.time_slots, size=n_sessions))
        for slot in session_slots:
            pop = slot_pop[slot]
            if rng.random() < cfg.template_use_p:
                tmpl = templates[slot][rng.integers(0, cfg.templates_per_slot)]
                for app in tmpl:
                    if rng.random() < cfg.template_noise_p:
                        app = rng.choice(cfg.n_apps, p=pop)
                    app_ids.append(int(app))
                    hours.append(slot)
                    last_app = int(app)
            else:
                slen = rng.integers(cfg.free_session_len[0], cfg.free_session_len[1] + 1)
                for _ in range(slen):
                    r = rng.random()
                    if last_app is not None and r < cfg.continuity_p:
                        app = last_app
                    elif r < cfg.continuity_p + cfg.novelty_p:
                        app = int(rng.integers(0, cfg.n_apps))
                    else:
                        app = int(rng.choice(cfg.n_apps, p=pop))
                    app_ids.append(app)
                    hours.append(slot)
                    last_app = app

    return np.array(app_ids, dtype=np.int64), np.array(hours, dtype=np.int64)


if __name__ == "__main__":
    ids, hrs = generate_trace()
    print(f"trace length = {len(ids)}, unique apps used = {len(set(ids.tolist()))}")
    vals, counts = np.unique(ids, return_counts=True)
    top = np.argsort(-counts)[:5]
    print("top-5 apps by frequency:", list(zip(vals[top].tolist(), counts[top].tolist())))
