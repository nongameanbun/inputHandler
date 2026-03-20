from collections import deque
import threading

Input_Q = deque()
ardu_lock = threading.Lock()  # Arduino 소켓 전송 동기화
q_lock = threading.Lock()     # Input_Q 복합 삽입 동기화 (press+delay+release 등)