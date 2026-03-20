from arduino import ardu
from HID import get_key_code
import struct
from base import Input_Q, ardu_lock
import time

def input_queue_consumer() :
    while True :
        try :
            while True :
                if len(Input_Q) == 0 :
                    time.sleep(0.001)  # 큐 비어있을 때 1ms 폴링 (CPU 절약)
                    continue

                cmd_start = time.time()
                data = Input_Q.popleft().split()
                if data[0] == "on" :
                    with ardu_lock:
                        ardu.send(bytes([0x02]))

                elif data[0] == "off" :
                    with ardu_lock:
                        ardu.send(bytes([0x03]))

                elif data[0] == "releaseAll" :
                    with ardu_lock:
                        ardu.send(bytes([0x04]))

                elif data[0] == "delay" :
                    if len(data) != 2 :
                        print("Invalid delay command")
                        continue
                    ms = int(data[1])
                    ms = max(0, min(ms, 65535))
                    with ardu_lock:
                        ardu.send(bytes([0x05, ms & 0xFF, (ms >> 8) & 0xFF]))

                elif data[0] == "sleep" :
                    # Python측에서 sleep (마우스 속도 제어용)
                    if len(data) != 2 :
                        continue
                    ms = int(data[1])
                    if ms > 0 :
                        time.sleep(ms / 1000.0)

                elif data[0] == "press_key" :
                    if len(data) != 2 :
                        print("Invalid press_key command")
                        continue
                    k = get_key_code(data[1])
                    with ardu_lock:
                        ardu.send(bytes([0x00, k & 0xFF]))

                elif data[0] == "release_key" :
                    if len(data) != 2 :
                        print("Invalid release_key command")
                        continue
                    k = get_key_code(data[1])
                    with ardu_lock:
                        ardu.send(bytes([0x01, k & 0xFF]))

                elif data[0] == "dMouse" :
                    if len(data) != 3 :
                        print("Invalid dMouse command")
                        continue
                    dx = int(data[1])
                    dy = int(data[2])
                    packet = struct.pack("<Bii", 0x06, dx, dy)
                    with ardu_lock:
                        ardu.send(packet)

                elif data[0] == "cMouse" :
                    if len(data) != 3 :
                        print("Invalid cMouse command")
                        continue
                    click_mode = int(data[1])
                    delay = int(data[2])
                    delay_ms1 = delay & 0xFF
                    delay_ms2 = (delay >> 8) & 0xFF
                    with ardu_lock:
                        ardu.send(bytes([0x07, click_mode, delay_ms1, delay_ms2]))

                # 명령 간 최소 2ms 간격 유지 (USB bInterval)
                elapsed = time.time() - cmd_start
                remaining = 0.002 - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as e :
            Input_Q.clear()
            continue