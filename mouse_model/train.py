"""
P3 학습 루프: MDN-LSTM을 packed 합성 데이터(dataset.py)로 학습.

  uv run python -m mouse_model.train --data mouse_model/data/synth/synth.npz --epochs 25

체크포인트는 export_onnx.py가 기대하는 형식으로 저장한다:
  {"config": {"input_dim","hidden","num_layers","num_mixtures"}, "state_dict": ...}
"""
from __future__ import annotations

import os
import sys
import argparse
import time

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mouse_model.model import MDNLSTM, mdn_loss, FEATURE_DIM, HIDDEN, NUM_LAYERS, NUM_MIXTURES  # noqa: E402
from mouse_model.dataset import TrajectoryDataset, collate_fn  # noqa: E402
from mouse_model import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(common.SYNTH_DIR, "synth.npz"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"데이터 로드: {args.data}")
    ds = TrajectoryDataset(args.data)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(args.seed))
    print(f"  전체={len(ds)}  train={n_train}  val={n_val}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MDNLSTM(FEATURE_DIM, HIDDEN, NUM_LAYERS, NUM_MIXTURES)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_nll = tr_eom = 0.0
        n_batches = 0
        for feats, targets, eom, mask in train_dl:
            raw, _ = model(feats)
            loss, nll, eom_l = mdn_loss(raw, model, targets, eom, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item(); tr_nll += nll.item(); tr_eom += eom_l.item()
            n_batches += 1
        sched.step()

        model.eval()
        va_loss = va_nll = va_eom = 0.0
        vb = 0
        with torch.no_grad():
            for feats, targets, eom, mask in val_dl:
                raw, _ = model(feats)
                loss, nll, eom_l = mdn_loss(raw, model, targets, eom, mask)
                va_loss += loss.item(); va_nll += nll.item(); va_eom += eom_l.item()
                vb += 1

        tr_loss /= max(n_batches, 1); tr_nll /= max(n_batches, 1); tr_eom /= max(n_batches, 1)
        va_loss /= max(vb, 1); va_nll /= max(vb, 1); va_eom /= max(vb, 1)
        dt = time.time() - t0
        print(f"[{epoch:3d}/{args.epochs}] train loss={tr_loss:.3f}(nll={tr_nll:.3f},eom={tr_eom:.3f})  "
              f"val loss={va_loss:.3f}(nll={va_nll:.3f},eom={va_eom:.3f})  {dt:.1f}s", flush=True)

        if va_loss < best_val:
            best_val = va_loss
            cfg = dict(input_dim=FEATURE_DIM, hidden=HIDDEN, num_layers=NUM_LAYERS, num_mixtures=NUM_MIXTURES)
            torch.save({"config": cfg, "state_dict": model.state_dict()}, args.out)
            print(f"  -> 체크포인트 저장 (val={va_loss:.3f}): {args.out}")

    print("학습 완료.")


if __name__ == "__main__":
    main()
