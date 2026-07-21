"""
model.pt (MDN-LSTM) → model.onnx 변환.

추론은 8ms 스텝마다 한 스텝씩 진행하는 autoregressive 방식이라, LSTM 은닉상태를
모델 입출력으로 빼서 스텝 단위로 export 한다. 샘플링(혼합분포에서 오프셋 뽑기)은
numpy 로 sample.py 가 담당하므로 그래프에는 넣지 않는다.

  uv run --extra dev python -m mouse_model.export_onnx

학습으로 model.pt 를 갱신했을 때만 다시 돌리면 된다. 런타임(HID)은 .onnx 만 읽는다.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mouse_model.model import MDNLSTM  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(BASE, "model.pt")
OUT = os.path.join(BASE, "model.onnx")


class StepModel(torch.nn.Module):
    """한 스텝 실행: (x, h0, c0) → (raw, hn, cn)."""

    def __init__(self, m: MDNLSTM):
        super().__init__()
        self.lstm = m.lstm
        self.head = m.head

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return self.head(out), hn, cn


def main() -> int:
    if not os.path.exists(CKPT):
        print(f"  !! 체크포인트 없음: {CKPT}")
        return 1

    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ckpt["config"]
    model = MDNLSTM(cfg["input_dim"], cfg["hidden"], cfg["num_layers"], cfg["num_mixtures"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    step = StepModel(model).eval()
    x = torch.zeros(1, 1, cfg["input_dim"])
    h0 = torch.zeros(cfg["num_layers"], 1, cfg["hidden"])
    c0 = torch.zeros(cfg["num_layers"], 1, cfg["hidden"])

    with torch.no_grad():
        torch.onnx.export(
            step, (x, h0, c0), OUT,
            input_names=["x", "h0", "c0"],
            output_names=["raw", "hn", "cn"],
            opset_version=17,
            dynamo=False,
        )

    # sample.py 가 혼합성분 개수 등을 알아야 해서 메타데이터로 함께 심는다.
    import onnx
    m = onnx.load(OUT)
    for k in ("input_dim", "hidden", "num_layers", "num_mixtures"):
        entry = m.metadata_props.add()
        entry.key, entry.value = k, str(cfg[k])
    onnx.save(m, OUT)

    print(f"  OK  model.pt ({os.path.getsize(CKPT)/1048576:.1f} MB)"
          f" -> model.onnx ({os.path.getsize(OUT)/1048576:.1f} MB)")
    print(f"      cfg = {dict((k, cfg[k]) for k in ('input_dim','hidden','num_layers','num_mixtures'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
