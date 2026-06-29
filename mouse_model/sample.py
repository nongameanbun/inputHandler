"""
P4: 추론 — 학습된 MDN-LSTM 으로 (start, target) → 사람같은 화면 좌표 궤적 생성.

TrajectoryGenerator 가 HID 통합의 진입점:
    gen = TrajectoryGenerator.load("checkpoints/model.pt")
    pts = gen.generate(start_px=(x0,y0), target_px=(x1,y1))   # (M,2) pixel 절대좌표

생성 절차
  1) canonical frame(K=100, 시작 (0,0), 목표 (K,0))에서 오프셋을 autoregressive 샘플
  2) EOM 샘플 / 목표 근접 / max_steps 로 종료
  3) canonical_to_screen 로 화면 좌표 역변환, 마지막 점은 정확히 target 으로 고정
  (실제 전송/closed-loop 보정은 HID 의 기존 파이프라인이 담당)

CLI: python -m mouse_model.sample --plot   (몇 개 생성해 시각화)
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mouse_model import common  # noqa: E402
from mouse_model.model import MDNLSTM, sample_step  # noqa: E402

K = common.CANON_SCALE


class TrajectoryGenerator:
    def __init__(self, model: MDNLSTM, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, ckpt_path, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt["config"]
        model = MDNLSTM(cfg["input_dim"], cfg["hidden"],
                        cfg["num_layers"], cfg["num_mixtures"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, device)

    @torch.no_grad()
    def generate_canonical(self, L, bias=0.0, temperature=1.0,
                           max_steps=600, min_steps=3, seed=None):
        """canonical 오프셋 시퀀스를 샘플 → canonical points (n+1,2) 반환."""
        if seed is not None:
            torch.manual_seed(seed)
        dev = self.device
        log_dist = float(np.log(L + 1.0))

        pos = torch.zeros(2, device=dev)
        prev_off = torch.zeros(2, device=dev)
        hidden = None
        offsets = []

        for step in range(max_steps):
            remaining = torch.tensor([K, 0.0], device=dev) - pos
            rem_dist = torch.linalg.norm(remaining)
            prev_speed = torch.linalg.norm(prev_off)
            feat = torch.tensor([
                prev_off[0], prev_off[1],
                pos[0] / K, pos[1] / K,
                remaining[0] / K, remaining[1] / K,
                rem_dist / K,
                prev_speed,
                log_dist,
            ], device=dev, dtype=torch.float32).view(1, 1, -1)

            raw, hidden = self.model(feat, hidden)
            off, eom_prob = sample_step(raw[0, 0], self.model, bias, temperature)

            pos = pos + off
            prev_off = off
            offsets.append(off)

            rem_now = float(torch.linalg.norm(torch.tensor([K, 0.0], device=dev) - pos))
            if step + 1 >= min_steps and (
                rem_now < common.EOM_DIST_THRESH or np.random.rand() < eom_prob
            ):
                break

        offs = torch.stack(offsets).cpu().numpy() if offsets else np.zeros((1, 2))
        return common.offsets_to_canon_points(offs)     # (n+1, 2)

    def generate(self, start_px, target_px, **kw):
        """화면 절대좌표 (M,2) 궤적 (start 포함, 마지막=target 고정)."""
        start_px = np.asarray(start_px, dtype=np.float64)
        target_px = np.asarray(target_px, dtype=np.float64)
        D = target_px - start_px
        L = float(np.hypot(D[0], D[1]))
        if L < common.MIN_DIST_PX:
            return np.stack([start_px, target_px])

        canon_pts = self.generate_canonical(L, **kw)
        screen = common.canonical_to_screen(canon_pts, start_px, D)
        screen[0] = start_px
        screen[-1] = target_px          # 끝점 정확히 고정 (HID:540 과 동일 취지)
        return screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(common.CKPT_DIR, "model.pt"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--bias", type=float, default=0.0)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"체크포인트 없음: {args.ckpt} (먼저 train.py 실행)")
        return

    gen = TrajectoryGenerator.load(args.ckpt)
    start = (960, 540)
    targets = [(1500, 300), (300, 800), (1700, 900), (200, 200),
               (960, 100), (1200, 540), (700, 950), (1600, 540)]

    for i, tg in enumerate(targets[:args.n]):
        pts = gen.generate(start, tg, bias=args.bias)
        end_err = np.hypot(pts[-1, 0] - tg[0], pts[-1, 1] - tg[1])
        print(f"  move {i}: {len(pts)} pts ({len(pts)*common.DT*1000:.0f}ms), end_err={end_err:.2f}px")
        if args.plot:
            import matplotlib.pyplot as plt
            plt.plot(pts[:, 0], pts[:, 1], marker='o', ms=2, lw=1, label=f"m{i}")
            plt.scatter([tg[0]], [tg[1]], c='r', s=30, zorder=5)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.scatter([start[0]], [start[1]], c='k', s=40, marker='s', label='start')
        plt.gca().invert_yaxis(); plt.axis('equal'); plt.legend(fontsize=7)
        plt.title("Generated trajectories (model)")
        out = os.path.join(common.BASE_DIR, "sample_plot.png")
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print(f"플롯 저장 → {out}")


if __name__ == "__main__":
    main()
