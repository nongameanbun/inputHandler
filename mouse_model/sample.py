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
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mouse_model import common  # noqa: E402

K = common.CANON_SCALE


def _resolve_onnx(path: str) -> str:
    """설정이 .pt 를 가리켜도 같은 이름의 .onnx 를 쓴다 (export_onnx.py 산출물)."""
    return path if path.endswith(".onnx") else os.path.splitext(path)[0] + ".onnx"


class TrajectoryGenerator:
    """
    MDN-LSTM 을 8ms 스텝마다 한 번씩 돌려 궤적을 자기회귀 샘플링한다.

    LSTM 순전파만 onnxruntime 이 맡고, 혼합분포 샘플링은 numpy 로 한다
    (torch 를 올리면 이 서비스 하나에 300MB 넘게 든다).
    """

    def __init__(self, session: ort.InferenceSession, cfg: dict):
        self.session = session
        self.M = cfg["num_mixtures"]
        self.hidden = cfg["hidden"]
        self.num_layers = cfg["num_layers"]
        self.input_dim = cfg["input_dim"]

    @classmethod
    def load(cls, ckpt_path, device="cpu"):
        """device 인자는 호출부 호환을 위해 남겨둔다 (CPU 추론만 한다)."""
        onnx_path = _resolve_onnx(ckpt_path)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"{onnx_path} 없음 — `uv run --extra dev python -m mouse_model.export_onnx` 필요"
            )
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = session.get_modelmeta().custom_metadata_map
        cfg = {k: int(meta[k]) for k in ("input_dim", "hidden", "num_layers", "num_mixtures")}
        return cls(session, cfg)

    def _sample_step(self, raw_step, bias, temperature, rng):
        """model.sample_step 의 numpy 판. 오프셋 1개와 EOM 확률을 반환."""
        M = self.M
        pi_logits = raw_step[0:M] * (1.0 + bias)
        mu = raw_step[M:3 * M].reshape(M, 2)
        log_sigma = np.clip(raw_step[3 * M:5 * M].reshape(M, 2), -7.0, 5.0)
        sigma = np.exp(log_sigma - bias) * temperature
        rho = np.tanh(raw_step[5 * M:6 * M]) * 0.999

        eom_logit = float(raw_step[6 * M])
        eom_prob = 1.0 / (1.0 + np.exp(-eom_logit))

        # log_softmax 후 categorical 샘플 (torch.distributions.Categorical 과 동일)
        shifted = pi_logits - pi_logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()
        k = int(rng.choice(M, p=probs))

        sx, sy = float(sigma[k, 0]), float(sigma[k, 1])
        r = float(rho[k])
        z1, z2 = rng.standard_normal(2)
        dx = float(mu[k, 0]) + sx * z1
        dy = float(mu[k, 1]) + sy * (r * z1 + np.sqrt(max(1.0 - r * r, 1e-6)) * z2)
        return np.array([dx, dy], dtype=np.float32), eom_prob

    def generate_canonical(self, L, bias=0.0, temperature=1.0,
                           max_steps=600, min_steps=3, seed=None):
        """canonical 오프셋 시퀀스를 샘플 → canonical points (n+1,2) 반환."""
        rng = np.random.default_rng(seed)
        log_dist = float(np.log(L + 1.0))

        pos = np.zeros(2, dtype=np.float32)
        prev_off = np.zeros(2, dtype=np.float32)
        h = np.zeros((self.num_layers, 1, self.hidden), dtype=np.float32)
        c = np.zeros((self.num_layers, 1, self.hidden), dtype=np.float32)
        offsets = []

        target = np.array([K, 0.0], dtype=np.float32)

        for step in range(max_steps):
            remaining = target - pos
            rem_dist = float(np.linalg.norm(remaining))
            prev_speed = float(np.linalg.norm(prev_off))
            feat = np.array([
                prev_off[0], prev_off[1],
                pos[0] / K, pos[1] / K,
                remaining[0] / K, remaining[1] / K,
                rem_dist / K,
                prev_speed,
                log_dist,
            ], dtype=np.float32).reshape(1, 1, -1)

            raw, h, c = self.session.run(None, {"x": feat, "h0": h, "c0": c})
            off, eom_prob = self._sample_step(raw[0, 0], bias, temperature, rng)

            pos = pos + off
            prev_off = off
            offsets.append(off)

            rem_now = float(np.linalg.norm(target - pos))
            if step + 1 >= min_steps and (
                rem_now < common.EOM_DIST_THRESH or rng.random() < eom_prob
            ):
                break

        offs = np.stack(offsets) if offsets else np.zeros((1, 2))
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
