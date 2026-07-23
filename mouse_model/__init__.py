"""
mouse_model — MDN-LSTM 마우스 궤적 모델(학습 + 런타임 추론).

  common.py            → canonical frame 기하 변환 / 설정 (numpy)
  model.py             → MDN-LSTM 정의 (torch)
  sample.py            → TrajectoryGenerator (HID.py 진입점). model.onnx 로드.
  gather_synthetic.py  → HumanCursor(Bezier) 기반 합성 학습 데이터 생성
  dataset.py           → packed npz → (feature, target, eom, mask) 배치
  train.py             → MDN-LSTM 학습 루프, model.pt 저장
  export_onnx.py        → model.pt → model.onnx 변환 (학습 후 재실행 필요)
"""
