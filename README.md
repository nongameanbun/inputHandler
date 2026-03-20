# nongameanbun - inputHandler


## 프로젝트 구조
```text
.
├── main.py # 모든 API 라우터 및 동작 엔드포인트 정의.
├── consumer.py # 입력 큐 관리
├── arduino.py # 실제 아두이노 gateway
├── base.py # 전역 변수 설정
├── ArduinoInterface.ino # 아두이노에 업로드되는 소스 코드.
└── HID.py # 키보드 및 마우스 입력 로직
```

## 사전 요구 사항

### 환경 변수 세팅 (`.env`)
해당 정보를 ArduinoInterface.ino 에 일관되게 입력 후 아래와 같이 .env 파일 생성


```powershell
Copy-Item env.example .env
```

`.env` 을 다음과 같이 수정
```ini
ARDUINO_IP=192.168.x.x  # 아두이노 이더넷 쉴드에 할당된 IP 주소
ARDUINO_PORT=80         # 아두이노 TCP 연결 대기 포트
```




## 실행 방법

```bash
pip install -r requirements.txt
python main.py
```

localhost:8000/docs 로 swagger 확인 가능