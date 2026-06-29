"""
mouse_model — 학습된 MDN-LSTM 마우스 궤적 모델의 런타임 추론 패키지.

  common.py  → canonical frame 기하 변환 / 설정 (numpy)
  model.py   → MDN-LSTM 정의 (torch)
  sample.py  → TrajectoryGenerator (HID.py 진입점). checkpoints/model.pt 로드.

학습/데이터 수집 도구는 제거됨 — 추론에 필요한 위 3개 모듈 + model.pt 만 유지.
"""
