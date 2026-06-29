"""
공용 설정 + 기하 변환 + 시간 리샘플링 + 저장 포맷.

이 모듈은 외부 의존성이 numpy 뿐이라 torch/pygame 설치 전에도 import 가능하다.
(dataset / sample / evaluate / gather.ipynb 가 여기에 의존)

좌표계 규약
-----------
- screen(pixel) frame: 화면 절대 좌표. y는 아래로 증가(Windows GetCursorPos 기준).
- canonical frame: 한 번의 이동을 표준화한 좌표계.
    * 시작점을 원점으로 평행이동
    * start→end 방향이 +x축이 되도록 회전
    * start→end 거리(L)로 스케일한 뒤 CANON_SCALE(K)를 곱함  →  end = (K, 0)
  즉 모든 이동이 (0,0)→(K,0) 형태가 되어 방향/거리 불변. 사람 움직임의
  "형태 + 시간 동역학"만 학습하게 만든다. 절대 거리 정보는 log_dist feature로 따로 조건화.
"""
from __future__ import annotations

import os
import math
import numpy as np

# ───────────────────────── 설정 ─────────────────────────

DT = 0.008                 # 8ms — HID._CMD_INTERVAL_S 및 consumer 명령 cadence와 일치
CANON_SCALE = 100.0        # canonical 좌표계에서 end가 놓이는 x값 (K). offset이 O(1)이 되도록.
MIN_DIST_PX = 8.0          # 이보다 짧은 이동은 학습/생성 대상에서 제외(노이즈)
EOM_DIST_THRESH = 3.0      # canonical(K=100 기준) 타겟 근접 판정 임계(추론 종료용)

# ───────────────────────── 경로 ─────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_HUMAN_DIR = os.path.join(DATA_DIR, "raw_human")   # recorder.py 가 move 1건당 1파일
SYNTH_DIR = os.path.join(DATA_DIR, "synth")           # gather.ipynb 수집 + synth.npz 패킹
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")   # dataset.py 패킹 결과
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")


def ensure_dirs():
    for d in (DATA_DIR, RAW_HUMAN_DIR, SYNTH_DIR, PROCESSED_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)


# ───────────────────── 시간 기반 리샘플링 ─────────────────────

def time_resample(times: np.ndarray, points: np.ndarray, dt: float = DT) -> np.ndarray:
    """
    불균일 타임스탬프(times, 초)의 절대 좌표 시퀀스(points, (N,2))를
    0, dt, 2dt, ... 의 균일 시간격자로 선형 보간한다.

    속도 프로파일(가속/감속)을 보존하는 게 핵심 — 호장 기반 리샘플과 다르다.
    반환: (M,2) 균일 시간 샘플. 항상 첫/끝 점을 포함.
    """
    times = np.asarray(times, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if len(times) < 2:
        return points.copy()

    # 단조 증가 보장 (중복 타임스탬프 제거)
    t0 = times[0]
    times = times - t0
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times = times[keep]
    points = points[keep]
    if len(times) < 2:
        return points.copy()

    total_t = times[-1]
    n_out = max(int(round(total_t / dt)) + 1, 2)
    grid = np.linspace(0.0, total_t, n_out)

    xs = np.interp(grid, times, points[:, 0])
    ys = np.interp(grid, times, points[:, 1])
    return np.stack([xs, ys], axis=1)


def equal_time_resample(points: np.ndarray, duration: float, dt: float = DT) -> np.ndarray:
    """
    타임스탬프가 없는, '균일 시간 간격으로 가정 가능한' 점열(예: HumanCursor의 tween
    결과 — 각 점이 동일 시간 간격 샘플)을 duration에 걸쳐 dt 격자로 리샘플.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return points.copy()
    src_t = np.linspace(0.0, duration, n)
    return time_resample(src_t, points, dt)


# ───────────────────── canonical frame 변환 ─────────────────────

def to_canonical(points: np.ndarray):
    """
    pixel 절대 좌표 점열 (N,2) → (total_px, canonical_offsets)

    total_px : (2,)  end - start (pixel). 거리/방향 복원 및 log_dist 조건용.
    canonical_offsets : (N-1, 2)  canonical frame(K 스케일)에서의 스텝별 오프셋.
        canonical point c_0=(0,0) ... c_{N-1}=(K,0), offset_t = c_t - c_{t-1}.
    L(거리)==0 이면 None 반환(호출측에서 스킵).
    """
    points = np.asarray(points, dtype=np.float64)
    p0 = points[0]
    D = points[-1] - p0
    L = float(np.hypot(D[0], D[1]))
    if L < 1e-6:
        return None

    theta = math.atan2(D[1], D[0])
    c, s = math.cos(theta), math.sin(theta)

    rel = points - p0
    # 회전(-theta) 후 K/L 스케일 → end가 (K,0)
    xr = (c * rel[:, 0] + s * rel[:, 1]) * (CANON_SCALE / L)
    yr = (-s * rel[:, 0] + c * rel[:, 1]) * (CANON_SCALE / L)
    canon = np.stack([xr, yr], axis=1)

    offsets = np.diff(canon, axis=0).astype(np.float32)
    return D.astype(np.float32), offsets


def canonical_to_screen(canon_points: np.ndarray, start_px, total_px) -> np.ndarray:
    """
    canonical 점열 (M,2, K 스케일) → screen pixel 절대 좌표.
    to_canonical 의 역변환. start_px(이동 시작 절대좌표), total_px(=end-start)로 복원.
    """
    canon_points = np.asarray(canon_points, dtype=np.float64)
    start_px = np.asarray(start_px, dtype=np.float64)
    Dx, Dy = float(total_px[0]), float(total_px[1])
    L = float(np.hypot(Dx, Dy))
    if L < 1e-6:
        return np.repeat(start_px[None, :], len(canon_points), axis=0)

    theta = math.atan2(Dy, Dx)
    c, s = math.cos(theta), math.sin(theta)

    # K/L 역스케일 후 회전(+theta) 후 평행이동
    px = canon_points[:, 0] * (L / CANON_SCALE)
    py = canon_points[:, 1] * (L / CANON_SCALE)
    sx = c * px - s * py + start_px[0]
    sy = s * px + c * py + start_px[1]
    return np.stack([sx, sy], axis=1)


def offsets_to_canon_points(offsets: np.ndarray) -> np.ndarray:
    """canonical offsets (n,2) → canonical points (n+1,2), 시작 (0,0) 포함."""
    offsets = np.asarray(offsets, dtype=np.float64)
    pts = np.zeros((len(offsets) + 1, 2), dtype=np.float64)
    pts[1:] = np.cumsum(offsets, axis=0)
    return pts


# ───────────────────── 패킹 저장/로드 (.npz) ─────────────────────
#
# 가변 길이 시퀀스를 object 배열 없이 저장하기 위해 concat + lengths 방식 사용.
#   total   : (N, 2)   float32   이동별 총 변위(pixel)
#   lengths : (N,)     int32     이동별 offset 개수
#   offsets : (sum,2)  float32   모든 이동의 canonical offset을 이어붙임

def save_packed(path: str, totals: list[np.ndarray], offsets_list: list[np.ndarray]):
    totals = np.asarray(totals, dtype=np.float32).reshape(-1, 2)
    lengths = np.asarray([len(o) for o in offsets_list], dtype=np.int32)
    if len(offsets_list):
        offsets = np.concatenate(offsets_list, axis=0).astype(np.float32)
    else:
        offsets = np.zeros((0, 2), dtype=np.float32)
    np.savez_compressed(path, total=totals, lengths=lengths, offsets=offsets)


def load_packed(path: str):
    """반환: list[ (total(2,), offsets(n,2)) ]"""
    z = np.load(path)
    totals, lengths, offsets = z["total"], z["lengths"], z["offsets"]
    out, cur = [], 0
    for i, n in enumerate(lengths):
        n = int(n)
        out.append((totals[i], offsets[cur:cur + n]))
        cur += n
    return out
