import ctypes
import math
import os
import struct
import time
from base import Input_Q, ardu_lock

user32 = ctypes.windll.user32

class key:
    '''
    사용 예시
    ## press_key(key.a)
    ## release_key(key.f1)
    '''

    # 알파벳
    a              = 0x61
    b              = 0x62
    c              = 0x63
    d              = 0x64
    e              = 0x65
    f              = 0x66
    g              = 0x67
    h              = 0x68
    i              = 0x69
    j              = 0x6a
    k              = 0x6b
    l              = 0x6c
    m              = 0x6d
    n              = 0x6e
    o              = 0x6f
    p              = 0x70
    q              = 0x71
    r              = 0x72
    s              = 0x73
    t              = 0x74
    u              = 0x75
    v              = 0x76
    w              = 0x77
    x              = 0x78
    y              = 0x79
    z              = 0x7a



    # 펑션키 (F1~F12)
    f1              = 0xc2
    f2              = 0xc3
    f3              = 0xc4
    f4              = 0xc5
    f5              = 0xc6
    f6              = 0xc7
    f7              = 0xc8
    f8              = 0xc9
    f9              = 0xca
    f10             = 0xcb
    f11             = 0xcc
    f12             = 0xcd

    # 숫자열 (키보드 상단 1~0, -, =)
    num0            = 0x30      # (X1, Y11)
    num1            = 0x31      # (X3, Y12)
    num2            = 0x32      # (X7, Y12)
    num3            = 0x33      # (X0, Y7)
    num4            = 0x34      # (X6, Y11)
    num5            = 0x35      # (X6, Y14)
    num6            = 0x36      # (X7, Y7)
    num7            = 0x37      # (X4, Y14)
    num8            = 0x38      # (X1, Y10)
    num9            = 0x39      # (X4, Y12)

    num_minus       = 0x2d
    num_equal       = 0x3d


    # 숫자패드 (Numpad)
    numpad1         = 0xe1
    numpad2         = 0xe2
    numpad3         = 0xe3
    numpad4         = 0xe4
    numpad5         = 0xe5
    numpad6         = 0xe6
    numpad7         = 0xe7
    numpad8         = 0xe8
    numpad9         = 0xe9
    numpad0         = 0xea
    numpad_enter    = 0xe0
    numpad_plus     = 0xdf
    numpad_minus    = 0xde
    numpad_multiply = 0xdd
    numpad_divide   = 0xdc
    numpad_period   = 0xeb
    numpad_numlock  = 0xdb

    # 편집 및 블록 이동키
    insert          = 0xd1
    delete          = 0xd4
    home            = 0xd2
    pageup          = 0xd3
    pagedown        = 0xd6
    end             = 0xd5

    # 화살표
    up              = 0xda
    down            = 0xd9
    left            = 0xd8
    right           = 0xd7

    # 특수키 (제어키 등)
    left_ctrl       = 0x80
    left_shift      = 0x81
    left_alt        = 0x82
    right_ctrl      = 0x84
    right_shift     = 0x85
    right_alt       = 0x86

    space           = 0xb4
    enter           = 0xb0
    backspace       = 0xb2
    tab             = 0xb3
    capslock        = 0xc1
    esc             = 0xb1

    printscreen     = 0xce
    scrolllock      = 0xcf
    pause           = 0xd0

    # 각종 기호
    backtick       = 0xbd
    left_bracket   = 0x5b
    right_bracket  = 0x5d
    backslash      = 0x5c
    semicolon      = 0x3b
    quote          = 0x27
    comma          = 0x2c
    period         = 0x2e
    slash          = 0x2f

    # -----------------------------------
    # 주의사항
    # 1) 실제 키보드 물리 배열과 다를 수 있으므로,
    #    이 표준 좌표는 상황에 맞게 조정이 필요합니다.
    # 2) 일부 키(예: /, numpad_divide) 등은 표 상에서
    #    좌표 충돌이 날 수도 있으니, 사용 시 유의하세요.
    # 3) Shift, Ctrl, Alt 등은 좌/우 구분이 필요하다면
    #    각각 다른 코드를 부여해 관리할 수 있습니다.
    # -----------------------------------

def get_mouse_pos():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def get_key_code(key_name: str) -> int:
    """
    key 클래스에서 키 이름으로 키 코드를 가져오는 함수
    예시: get_key_code("a") -> 0x61
    """
    if hasattr(key, key_name):
        return getattr(key, key_name)
    else:
        raise ValueError(f"Invalid key name: {key_name}")


class HumanMouseController:
    """
    학습된 MDN-LSTM(mouse_model) 궤적으로 절대좌표 마우스 이동을 수행.

    move_to_px_human(x, y):
      1. 현재 커서 위치(get_mouse_pos)에서 목표까지 모델이 8ms cadence 궤적 생성
      2. perf_counter busy-wait 기반 정밀 pacing 으로 Arduino에 직접 전송
      3. closed-loop 보정으로 목표에 정확히 안착

    모델(checkpoints/model.pt)이 필수 — 없으면 RuntimeError.
    """

    # 학습 모델(mouse_model) 캐시 — 클래스 단위로 1회만 로드
    _model_gen = None
    _model_load_failed = False

    # ============ 학습 모델 로더 ============ #

    def _get_model(self):
        """mouse_model/model.pt 를 1회 로드해 TrajectoryGenerator 반환. 실패 시 None."""
        if HumanMouseController._model_gen is not None:
            return HumanMouseController._model_gen
        if HumanMouseController._model_load_failed:
            return None
        try:
            ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "mouse_model", "model.pt")
            if not os.path.exists(ckpt):
                HumanMouseController._model_load_failed = True
                return None
            from mouse_model.sample import TrajectoryGenerator
            HumanMouseController._model_gen = TrajectoryGenerator.load(ckpt, device="cpu")
            print(f"[HumanMouse] 학습 모델 로드: {ckpt}")
            return HumanMouseController._model_gen
        except Exception as e:
            print(f"[HumanMouse] 모델 로드 실패: {e}")
            HumanMouseController._model_load_failed = True
            return None

    def _generate_curve_model(self, start_pt, target_pt):
        """학습 모델로 화면 좌표 궤적(8ms cadence) 생성. 모델 없거나 실패 시 None."""
        gen = self._get_model()
        if gen is None:
            return None
        try:
            pts = gen.generate(start_pt, target_pt)
            if pts is None or len(pts) < 2:
                return None
            return pts
        except Exception as e:
            print(f"[HumanMouse] 모델 추론 실패: {e}")
            return None

    # ============ 포인트 → dMouse 커맨드 변환 ============ #

    def _points_to_commands(self, points):
        """
        연속 포인트 시퀀스의 차분을 dMouse 커맨드로 변환.
        dMouse 1 = 1 pixel이므로 소수점 누적만 관리.
        """
        commands = []
        extra_x = 0.0
        extra_y = 0.0

        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]

            cmd_fx = dx + extra_x
            cmd_fy = dy + extra_y

            cmd_x = int(cmd_fx)
            cmd_y = int(cmd_fy)

            extra_x = cmd_fx - cmd_x
            extra_y = cmd_fy - cmd_y

            if cmd_x != 0 or cmd_y != 0:
                commands.append((cmd_x, cmd_y))

        # 잔여 소수점 처리
        rx = int(round(extra_x))
        ry = int(round(extra_y))
        if rx != 0 or ry != 0:
            commands.append((rx, ry))

        return commands

    # ============ 메인: move_to_px_human ============ #

    # USB HID 폴링 주기를 고려한 최소 커맨드 간격
    # Arduino가 Mouse.move() 후 delay(2)하므로, Python에서도 충분한 간격 확보
    _CMD_INTERVAL_S = 0.008  # 8ms — USB 125Hz polling + TCP 전송 마진

    @staticmethod
    def _precise_sleep(seconds: float):
        """
        Windows 타이머 해상도를 1ms로 설정 후 sleep.
        """
        if seconds <= 0:
            return
        # Windows 타이머 해상도를 1ms로 설정
        ctypes.windll.winmm.timeBeginPeriod(1)
        try:
            time.sleep(seconds)
        finally:
            ctypes.windll.winmm.timeEndPeriod(1)

    @staticmethod
    def _wait_queue_drain(timeout: float = 5.0):
        """Input_Q가 빌 때까지 대기."""
        t0 = time.perf_counter()
        while len(Input_Q) > 0:
            if time.perf_counter() - t0 > timeout:
                break
            time.sleep(0.0005)

    def move_to_px_human(
        self,
        target_x: int,
        target_y: int,
        duration: float | None = None,
        stop_dist: float = 1.0,
    ):
        """
        학습 모델 궤적 + 직접 pacing + closed-loop 보정으로 절대좌표 이동.
        """
        # 0) 기존 큐 명령이 Arduino에서 완전히 소화될 때까지 대기
        #    (OP_SLEEP 중 dMouse가 TCP 버퍼에 쌓이는 문제 방지)
        self._wait_queue_drain(timeout=10.0)
        time.sleep(0.03)  # Arduino 측 잔여 OP 처리 여유

        start_x, start_y = get_mouse_pos()

        total_dist = math.hypot(target_x - start_x, target_y - start_y)
        if total_dist <= stop_dist:
            return

        # 1) 학습 모델로 8ms cadence 궤적 생성 (모델 필수)
        model_pts = self._generate_curve_model(
            (float(start_x), float(start_y)),
            (float(target_x), float(target_y)),
        )
        if model_pts is None:
            raise RuntimeError(
                "학습 모델 궤적 생성 실패. mouse_model/checkpoints/model.pt 확인 "
                "(python -m mouse_model.train 으로 생성)."
            )

        # 모델은 이미 8ms cadence로 출력 → 리샘플 불필요. duration은 스텝 수에서 자연 결정.
        resampled = [(float(x), float(y)) for x, y in model_pts]
        resampled[-1] = (float(target_x), float(target_y))
        if duration is None:
            duration = max((len(resampled) - 1) * self._CMD_INTERVAL_S,
                           self._CMD_INTERVAL_S)

        # 2) 포인트 → dMouse 커맨드 변환
        commands = self._points_to_commands(resampled)
        if not commands:
            return

        # 3) Arduino에 직접 전송 + 정밀 pacing
        #    consumer 큐를 우회하고 lock으로 소켓 접근 동기화
        from arduino import ardu

        n_cmds = len(commands)
        interval = duration / n_cmds

        move_start = time.perf_counter()

        for i, (cx, cy) in enumerate(commands):
            # 이 커맨드의 목표 시각까지 정밀 대기
            target_time = move_start + interval * i
            now = time.perf_counter()
            wait = target_time - now
            if wait > 0:
                self._precise_sleep(wait)

            # lock으로 consumer와 소켓 충돌 방지 후 직접 전송
            packet = struct.pack("<Bii", 0x06, cx, cy)
            with ardu_lock:
                ardu.send(packet)

        # duration 끝까지 대기
        remaining = (move_start + duration) - time.perf_counter()
        if remaining > 0:
            self._precise_sleep(remaining)

        # 4) Closed-loop 정밀 보정 (작은 step으로 분할)
        MAX_CORR_STEP = 5  # 보정 1회 최대 이동량 (px)
        time.sleep(0.02)
        for attempt in range(200):
            cur_x, cur_y = get_mouse_pos()
            ex = target_x - cur_x
            ey = target_y - cur_y

            if abs(ex) <= 0 and abs(ey) <= 0:
                break

            # 오차 벡터를 MAX_CORR_STEP 이하로 축소
            err_mag = math.hypot(ex, ey)
            if err_mag > MAX_CORR_STEP:
                scale = MAX_CORR_STEP / err_mag
                cmd_x = int(round(ex * scale))
                cmd_y = int(round(ey * scale))
            else:
                cmd_x = ex
                cmd_y = ey

            # 최소 1px은 보내야 진행됨
            if cmd_x == 0 and cmd_y == 0:
                cmd_x = 1 if ex > 0 else (-1 if ex < 0 else 0)
                cmd_y = 1 if ey > 0 else (-1 if ey < 0 else 0)

            packet = struct.pack("<Bii", 0x06, cmd_x, cmd_y)
            with ardu_lock:
                ardu.send(packet)
            self._precise_sleep(0.008)


human_mouse = HumanMouseController()
