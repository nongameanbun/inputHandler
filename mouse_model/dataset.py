"""
P2: packed npz(canonical offsets) → 학습용 (feature, target, eom_target, mask) 배치.

feature 구성은 sample.py의 generate_canonical 루프와 완전히 동일해야 한다(teacher forcing):
  [prev_off_x, prev_off_y, pos_x/K, pos_y/K, remaining_x/K, remaining_y/K,
   rem_dist/K, prev_speed, log_dist]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mouse_model import common  # noqa: E402

K = common.CANON_SCALE


def build_sequence(offsets: np.ndarray, log_dist: float):
    """offsets(n,2) canonical → feature(n,9), target(n,2), eom(n,)."""
    n = len(offsets)
    pos = np.zeros(2, dtype=np.float32)
    prev_off = np.zeros(2, dtype=np.float32)
    target_pt = np.array([K, 0.0], dtype=np.float32)

    feats = np.zeros((n, 9), dtype=np.float32)
    for t in range(n):
        remaining = target_pt - pos
        rem_dist = float(np.linalg.norm(remaining))
        prev_speed = float(np.linalg.norm(prev_off))
        feats[t] = [
            prev_off[0], prev_off[1],
            pos[0] / K, pos[1] / K,
            remaining[0] / K, remaining[1] / K,
            rem_dist / K,
            prev_speed,
            log_dist,
        ]
        prev_off = offsets[t]
        pos = pos + offsets[t]

    eom = np.zeros(n, dtype=np.float32)
    eom[-1] = 1.0
    return feats, offsets.astype(np.float32), eom


class TrajectoryDataset(Dataset):
    def __init__(self, npz_path: str, max_len: int = 400):
        packed = common.load_packed(npz_path)
        self.items = []
        for total, offsets in packed:
            if len(offsets) < 1 or len(offsets) > max_len:
                continue
            L = float(np.hypot(total[0], total[1]))
            log_dist = float(np.log(L + 1.0))
            feats, targets, eom = build_sequence(np.asarray(offsets, dtype=np.float32), log_dist)
            self.items.append((feats, targets, eom))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feats, targets, eom = self.items[idx]
        return (
            torch.from_numpy(feats),
            torch.from_numpy(targets),
            torch.from_numpy(eom),
        )


def collate_fn(batch):
    """가변 길이 시퀀스를 배치 내 최대 길이로 0-패딩 + mask 생성."""
    lens = [item[0].shape[0] for item in batch]
    max_len = max(lens)
    B = len(batch)
    feat_dim = batch[0][0].shape[1]

    feats = torch.zeros(B, max_len, feat_dim)
    targets = torch.zeros(B, max_len, 2)
    eom = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len)

    for i, (f, t, e) in enumerate(batch):
        n = f.shape[0]
        feats[i, :n] = f
        targets[i, :n] = t
        eom[i, :n] = e
        mask[i, :n] = 1.0

    return feats, targets, eom, mask
