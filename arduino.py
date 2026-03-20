# InputHandler.py
import os
import socket
import time
import dotenv

dotenv.load_dotenv()

ARDUINO_IP = str(os.getenv('ARDUINO_IP'))
ARDUINO_PORT = int(os.getenv('ARDUINO_PORT'))

class ArduinoClient:
    def __init__(self, ip, port):
        # print(f"[ArduinoClient] Connecting to {ip}:{port}...")
        self.ip = ip
        self.port = port
        self.sock = None
        self._connect()

    def _connect(self):
        """소켓 연결 (재연결 시에도 사용)"""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self.sock.connect((self.ip, self.port))
            # print(f"[ArduinoClient] Connected successfully!")
        except Exception as e:
            print(f"[ArduinoClient] Connection FAILED: {e}")
            raise

    def _reconnect(self, max_retries=10, retry_delay=1.0):
        """프레이밍 오류 등으로 Arduino가 연결을 끊었을 때 자동 재연결"""
        for attempt in range(max_retries):
            try:
                print(f"[ArduinoClient] Reconnecting... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                self._connect()
                return
            except Exception as e:
                print(f"[ArduinoClient] Reconnect attempt {attempt + 1} failed: {e}")
        raise ConnectionError(f"[ArduinoClient] Failed to reconnect after {max_retries} attempts")

    def send(self, data: bytes):
        try:
            self.sock.sendall(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            print(f"[ArduinoClient.send] Connection lost: {e}, attempting reconnect...")
            self._reconnect()
            # 재연결 후 재전송
            self.sock.sendall(data)
        except Exception as e:
            print(f"[ArduinoClient.send] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close(self):
        print(f"[ArduinoClient] Closing connection...")
        if self.sock:
            self.sock.close()
        print(f"[ArduinoClient] Closed!")

ardu = ArduinoClient(ARDUINO_IP, ARDUINO_PORT)
