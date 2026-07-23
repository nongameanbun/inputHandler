"""
P1: HumanCursor(Bezier) 기반 합성 학습 데이터 생성.

HumanCursor.HumanizeMouseTrajectory 는 offset_boundary/knots_count 를 절대 픽셀값으로
받는다. 원본 라이브러리의 기본값(offset_boundary 20~100px 고정)은 짧은 이동(예: 20px)에
그대로 쓰면 곡선 진폭이 이동거리보다 커져 제자리에서 빙빙 도는 궤적이 나온다.
여기서는 offset_boundary/knots_count/duration을 모두 실거리(L)에 비례/로그비례로
스케일링해서, 짧은 거리는 짧고 곧게, 긴 거리는 길고 여유 있게 만든다.

duration은 Fitts's law 형태(ID = log2(L/W+1))로 거리에 비례 조건화한다.

출력: mouse_model/data/synth/synth.npz (common.save_packed 포맷)
"""
from __future__ import annotations

import os
import sys
import math
import random
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mouse_model import common  # noqa: E402

from humancursor.utilities.human_curve_generator import HumanizeMouseTrajectory
import pytweening

TWEEN_OPTIONS = [
    pytweening.easeOutExpo, pytweening.easeInOutQuint, pytweening.easeInOutSine,
    pytweening.easeInOutQuart, pytweening.easeInOutExpo, pytweening.easeInOutCubic,
    pytweening.easeInOutCirc, pytweening.linear, pytweening.easeOutSine,
    pytweening.easeOutQuart, pytweening.easeOutQuint, pytweening.easeOutCubic,
    pytweening.easeOutCirc,
]

FITTS_W = 15.0     # 목표 폭 가정치(px)
FITTS_A = 0.09      # 절편(s)
FITTS_B = 0.11      # 기울기(s / bit)


def fitts_duration(L: float, rng: random.Random) -> float:
    idx = math.log2(L / FITTS_W + 1.0)
    base = FITTS_A + FITTS_B * idx
    jitter = rng.uniform(0.8, 1.25)
    return max(0.06, base * jitter)


def scaled_curve_params(L: float, rng: random.Random) -> dict:
    """이동거리 L(px)에 비례/로그비례한 Bezier 곡선 파라미터."""
    # 진폭: 거리의 일정 비율, 아주 짧은 거리에서도 최소 진폭은 남기되 절대 상한을 둔다.
    frac = rng.uniform(0.04, 0.22)
    boundary = max(2.0, min(L * frac, 140.0))
    offset_boundary_x = boundary * rng.uniform(0.6, 1.3)
    offset_boundary_y = boundary * rng.uniform(0.6, 1.3)

    # knots: 짧은 거리는 1개(직선에 가까움), 길수록 최대 5개까지
    if L < 40:
        knots_count = rng.choices([0, 1], [0.5, 0.5])[0]
    elif L < 150:
        knots_count = rng.choices([1, 2], [0.6, 0.4])[0]
    elif L < 500:
        knots_count = rng.choices([1, 2, 3], [0.3, 0.4, 0.3])[0]
    else:
        knots_count = rng.choices([2, 3, 4, 5], [0.3, 0.35, 0.25, 0.1])[0]

    distortion_mean = rng.uniform(0.85, 1.1)
    distortion_st_dev = rng.uniform(0.7, 1.1)
    # 아주 짧은 거리는 손떨림 노이즈 빈도를 낮춰 과도한 지그재그 방지
    distortion_frequency = rng.uniform(0.15, 0.35) if L < 60 else rng.uniform(0.25, 0.55)

    tween = rng.choice(TWEEN_OPTIONS)
    target_points = max(int(L), 8)

    return dict(
        offset_boundary_x=offset_boundary_x, offset_boundary_y=offset_boundary_y,
        knots_count=knots_count, distortion_mean=distortion_mean,
        distortion_st_dev=distortion_st_dev, distortion_frequency=distortion_frequency,
        tween=tween, target_points=target_points,
    )


def sample_distance(rng: random.Random) -> float:
    """짧은 거리를 충분히 오버샘플링하는 로그-균등 분포."""
    lo, hi = math.log(common.MIN_DIST_PX), math.log(2400.0)
    return math.exp(rng.uniform(lo, hi))


def gen_one(rng: random.Random):
    L = sample_distance(rng)
    theta = rng.uniform(0, 2 * math.pi)
    start = (0.0, 0.0)
    target = (L * math.cos(theta), L * math.sin(theta))

    params = scaled_curve_params(L, rng)
    curve = HumanizeMouseTrajectory(start, target, **params)
    pts = np.asarray(curve.points, dtype=np.float64)
    if len(pts) < 2:
        return None

    duration = fitts_duration(L, rng)
    resampled = common.equal_time_resample(pts, duration, dt=common.DT)
    resampled[0] = start
    resampled[-1] = target
    if len(resampled) < 2:
        return None

    result = common.to_canonical(resampled)
    if result is None:
        return None
    total, offsets = result
    if len(offsets) < 1:
        return None
    return total, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    common.ensure_dirs()
    out_path = args.out or os.path.join(common.SYNTH_DIR, "synth.npz")

    rng = random.Random(args.seed)
    totals, offsets_list = [], []
    n_ok = 0
    for i in range(args.n):
        r = gen_one(rng)
        if r is None:
            continue
        total, offsets = r
        totals.append(total)
        offsets_list.append(offsets)
        n_ok += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{args.n} (ok={n_ok})")

    common.save_packed(out_path, totals, offsets_list)
    lens = [len(o) for o in offsets_list]
    print(f"저장: {out_path}  n={n_ok}  step_len min/med/max={min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")


if __name__ == "__main__":
    main()
