import ctypes
import math
import random
import struct
import time
import numpy as np
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
    HumanCursor(riflosnake/HumanCursor)의 파이프라인을 그대로 차용:
      1. offset boundary 랜덤 설정
      2. knot 개수 가중치 확률분포
      3. 고차 Bezier (Bernstein polynomial)
      4. distort (정규분포 노이즈)
      5. tween (easeOut 계열 감속)
      6. target_points (거리 기반 포인트 수)
    
    출력은 dMouse 커맨드 시퀀스 → Input_Q 투입.
    """

    # ============ easeOut tween 함수들 (pytweening 대체) ============ #
    @staticmethod
    def _ease_out_quad(t):   return t * (2 - t)
    @staticmethod
    def _ease_out_cubic(t):  return 1 - (1 - t) ** 3
    @staticmethod
    def _ease_out_quart(t):  return 1 - (1 - t) ** 4
    @staticmethod
    def _ease_out_quint(t):  return 1 - (1 - t) ** 5
    @staticmethod
    def _ease_out_expo(t):   return 1.0 if t == 1.0 else 1 - 2 ** (-10 * t)
    @staticmethod
    def _ease_out_circ(t):   return math.sqrt(1 - (t - 1) ** 2)
    @staticmethod
    def _ease_out_sine(t):   return math.sin(t * math.pi / 2)
    @staticmethod
    def _ease_in_out_quad(t):
        return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
    @staticmethod
    def _ease_in_out_cubic(t):
        return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    @staticmethod
    def _ease_in_out_quart(t):
        return 8 * t ** 4 if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2
    @staticmethod
    def _ease_in_out_sine(t):
        return -(math.cos(math.pi * t) - 1) / 2
    @staticmethod
    def _ease_in_out_quint(t):
        return 16 * t ** 5 if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2
    @staticmethod
    def _ease_in_out_expo(t):
        if t == 0.0: return 0.0
        if t == 1.0: return 1.0
        return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2
    @staticmethod
    def _ease_in_out_circ(t):
        if t < 0.5:
            return (1 - math.sqrt(1 - (2 * t) ** 2)) / 2
        return (math.sqrt(1 - (-2 * t + 2) ** 2) + 1) / 2
    @staticmethod
    def _linear(t):          return t

    # HumanCursor 원본과 동일한 13개 tween (pytweening 함수 순서 그대로)
    TWEEN_OPTIONS = [
        _ease_out_expo.__func__,
        _ease_in_out_quint.__func__,
        _ease_in_out_sine.__func__,
        _ease_in_out_quart.__func__,
        _ease_in_out_expo.__func__,
        _ease_in_out_cubic.__func__,
        _ease_in_out_circ.__func__,
        _linear.__func__,
        _ease_out_sine.__func__,
        _ease_out_quart.__func__,
        _ease_out_quint.__func__,
        _ease_out_cubic.__func__,
        _ease_out_circ.__func__,
    ]

    def __init__(self):
        # dMouse 1 단위 = 1 pixel (Arduino HID 기준)
        self.px_per_cmd = 1.0

        # 마지막 이동 동선 기록
        self.last_trajectory: list[tuple[int, int]] = []

    # ============ Bezier 수학 (HumanCursor BezierCalculator 동일) ============ #

    @staticmethod
    def _binomial(n, k):
        return math.factorial(n) / float(math.factorial(k) * math.factorial(n - k))

    @staticmethod
    def _bernstein_polynomial_point(x, i, n):
        return HumanMouseController._binomial(n, i) * (x ** i) * ((1 - x) ** (n - i))

    @staticmethod
    def _bernstein_polynomial(points):
        def bernstein(t):
            n = len(points) - 1
            bx = by = 0.0
            for i, pt in enumerate(points):
                bern = HumanMouseController._bernstein_polynomial_point(t, i, n)
                bx += pt[0] * bern
                by += pt[1] * bern
            return bx, by
        return bernstein

    @staticmethod
    def _bezier_curve_points(n, points):
        """n개의 포인트를 Bezier 곡선 위에서 균등 t로 샘플링."""
        bern = HumanMouseController._bernstein_polynomial(points)
        return [bern(i / max(n - 1, 1)) for i in range(n)]

    # ============ HumanCursor 파이프라인 (generate_curve 동일) ============ #

    def _generate_random_params(self, from_pt, to_pt):
        """
        HumanCursor generate_random_curve_parameters() 와 동일한 확률분포.
        """
        tween = random.choice(self.TWEEN_OPTIONS)

        # HumanCursor 원본과 동일한 가중치 (원본: [0.2, 0.65, 15])
        offset_boundary_x = random.choice(
            random.choices(
                [range(20, 45), range(45, 75), range(75, 100)],
                [0.2, 0.65, 15]
            )[0]
        )
        offset_boundary_y = random.choice(
            random.choices(
                [range(20, 45), range(45, 75), range(75, 100)],
                [0.2, 0.65, 15]
            )[0]
        )

        knots_count = random.choices(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [0.15, 0.36, 0.17, 0.12, 0.08, 0.04, 0.03, 0.02, 0.015, 0.005],
        )[0]

        distortion_mean = random.choice(range(80, 110)) / 100
        distortion_st_dev = random.choice(range(85, 110)) / 100
        distortion_frequency = random.choice(range(25, 70)) / 100

        target_points = max(
            int(math.sqrt(
                (from_pt[0] - to_pt[0]) ** 2 + (from_pt[1] - to_pt[1]) ** 2
            )),
            2
        )

        return (
            offset_boundary_x, offset_boundary_y,
            knots_count,
            distortion_mean, distortion_st_dev, distortion_frequency,
            tween, target_points,
        )

    def _generate_internal_knots(self, l, r, d, u, count):
        """HumanCursor generate_internal_knots 동일 (np.random.choice 사용)."""
        l, r, d, u = int(l), int(r), int(d), int(u)
        if l > r:
            l, r = r, l
        if d > u:
            d, u = u, d
        if l == r:
            r = l + 1
        if d == u:
            u = d + 1
        kx = np.random.choice(range(l, r), size=count)
        ky = np.random.choice(range(d, u), size=count)
        return list(zip(kx.tolist(), ky.tolist()))

    def _distort_points(self, points, mean, st_dev, frequency):
        """HumanCursor distort_points 동일 (np.random.normal 사용)."""
        distorted = []
        for i in range(1, len(points) - 1):
            x, y = points[i]
            delta = (
                np.random.normal(mean, st_dev)
                if random.random() < frequency
                else 0
            )
            distorted.append((x, y + delta))
        return [points[0]] + distorted + [points[-1]]

    def _tween_points(self, points, tween, target_points):
        """HumanCursor tween_points 동일."""
        if target_points < 2:
            target_points = 2
        res = []
        for i in range(target_points):
            idx = int(tween(float(i) / (target_points - 1)) * (len(points) - 1))
            idx = max(0, min(idx, len(points) - 1))
            res.append(points[idx])
        return res

    def _generate_curve(self, from_pt, to_pt):
        """
        HumanCursor HumanizeMouseTrajectory.generate_curve() 와 동일 파이프라인.
        반환: [(x, y), ...] 포인트 시퀀스
        """
        (
            off_bx, off_by, knots_count,
            dist_mean, dist_std, dist_freq,
            tween, target_points,
        ) = self._generate_random_params(from_pt, to_pt)

        left   = min(from_pt[0], to_pt[0]) - off_bx
        right  = max(from_pt[0], to_pt[0]) + off_bx
        down   = min(from_pt[1], to_pt[1]) - off_by
        up     = max(from_pt[1], to_pt[1]) + off_by

        knots = self._generate_internal_knots(left, right, down, up, knots_count)

        control_points = [from_pt] + knots + [to_pt]

        mid_count = max(
            abs(int(from_pt[0]) - int(to_pt[0])),
            abs(int(from_pt[1]) - int(to_pt[1])),
            2
        )
        points = self._bezier_curve_points(mid_count, control_points)

        points = self._distort_points(points, dist_mean, dist_std, dist_freq)

        points = self._tween_points(points, tween, target_points)

        return points

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

    # ============ 리샘플링 ============ #

    @staticmethod
    def _resample_curve(points, n_out):
        """
        가변 간격의 포인트 시퀀스를 누적 호장(arc-length) 기반으로
        n_out 개의 등간격 포인트로 리샘플링.
        """
        if n_out <= 2:
            return [points[0], points[-1]]

        # 누적 거리 계산
        dists = [0.0]
        for i in range(1, len(points)):
            d = math.hypot(points[i][0] - points[i-1][0],
                           points[i][1] - points[i-1][1])
            dists.append(dists[-1] + d)

        total_len = dists[-1]
        if total_len < 1e-9:
            return [points[0]] * n_out

        result = []
        seg = 0
        for k in range(n_out):
            target_dist = (k / (n_out - 1)) * total_len

            while seg < len(dists) - 2 and dists[seg + 1] < target_dist:
                seg += 1

            seg_len = dists[seg + 1] - dists[seg]
            if seg_len < 1e-9:
                t = 0.0
            else:
                t = (target_dist - dists[seg]) / seg_len

            x = points[seg][0] + t * (points[seg + 1][0] - points[seg][0])
            y = points[seg][1] + t * (points[seg + 1][1] - points[seg][1])
            result.append((x, y))

        return result

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

    def _send_one(self, dx: int, dy: int):
        """dMouse 커맨드 1개를 큐에 넣고 소진될 때까지 대기."""
        Input_Q.append(f"dMouse {dx} {dy}")
        self._wait_queue_drain(timeout=0.5)

    def move_to_px_human(
        self,
        target_x: int,
        target_y: int,
        duration: float | None = None,
        stop_dist: float = 1.0,
    ):
        """
        HumanCursor 파이프라인 + 직접 pacing.
        1. HumanCursor 곡선 생성 (고차 Bezier + tween)
        2. duration / CMD_INTERVAL 개로 리샘플링 → 커맨드 수 제한
        3. perf_counter busy-wait 기반 정밀 pacing
        4. closed-loop 보정
        """
        self.last_trajectory = []

        # 0) 기존 큐 명령이 Arduino에서 완전히 소화될 때까지 대기
        #    (OP_SLEEP 중 dMouse가 TCP 버퍼에 쌓이는 문제 방지)
        self._wait_queue_drain(timeout=10.0)
        time.sleep(0.03)  # Arduino 측 잔여 OP 처리 여유

        start_x, start_y = get_mouse_pos()
        self.last_trajectory.append((start_x, start_y))

        total_dist = math.hypot(target_x - start_x, target_y - start_y)
        if total_dist <= stop_dist:
            return

        # 1) HumanCursor 파이프라인으로 곡선 포인트 생성
        curve_points = self._generate_curve(
            (float(start_x), float(start_y)),
            (float(target_x), float(target_y))
        )

        # 마지막 포인트를 정확히 목표로 고정
        curve_points[-1] = (float(target_x), float(target_y))

        # 2) duration 계산 (HumanCursor와 동일: 0.5~2.0초 랜덤)
        if duration is None:
            duration = random.uniform(0.5, 2.0)

        # 3) 커맨드 수를 duration에 맞게 제한
        max_cmds = max(int(duration / self._CMD_INTERVAL_S), 2)
        n_out = min(len(curve_points), max_cmds)

        # 리샘플링으로 커맨드 수 제한
        resampled = self._resample_curve(curve_points, n_out)

        # trajectory 기록
        for pt in resampled:
            self.last_trajectory.append((int(pt[0]), int(pt[1])))

        # 4) 리샘플링된 포인트 → dMouse 커맨드 변환
        commands = self._points_to_commands(resampled)

        if not commands:
            return

        # 5) Arduino에 직접 전송 + 정밀 pacing
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

        # 6) Closed-loop 정밀 보정 (작은 step으로 분할)
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

    # ============ 동선 조회/플롯 ============ #

    def get_last_trajectory(self):
        return list(self.last_trajectory)

    def plot_last_trajectory(self, show: bool = True, save_path: str | None = None):
        if not self.last_trajectory:
            print("[HumanMouseController] last_trajectory 가 비어 있습니다.")
            return

        import matplotlib.pyplot as plt

        xs, ys = zip(*self.last_trajectory)

        plt.figure()
        plt.plot(xs, ys, marker='o', markersize=2, linewidth=1)
        plt.title("Last Mouse Trajectory")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.gca().invert_yaxis()
        plt.axis("equal")

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


human_mouse = HumanMouseController()

