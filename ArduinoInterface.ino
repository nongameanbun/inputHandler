// GG

#include <SPI.h>
#include <Ethernet.h>
#include "Mouse.h"
#include "Keyboard.h"

byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };
IPAddress ip(192, 168, 219, 99);
EthernetServer server(80);
EthernetClient client;

enum OPCODE : uint8_t {
  OP_PRESS       = 0x00, OP_RELEASE     = 0x01,
  OP_RUN_ON      = 0x02, OP_RUN_OFF     = 0x03,
  OP_RELEASE_ALL = 0x04, OP_SLEEP       = 0x05,
  OP_MOUSEMOVE   = 0x06, OP_MOUSECLICK  = 0x07,
};

volatile bool running = false;

const uint16_t USB_TICK_MS = 2;

volatile int32_t acc_dx = 0;
volatile int32_t acc_dy = 0;

const int32_t ACC_SOFT_LIMIT = 32767;

uint32_t last_send_ms = 0;

static inline int8_t clampDeltaToReportRange(int32_t v) {
    if (v > 127)  return 127;
    if (v < -127) return -127;

    return (int8_t)v;
}

static inline void pumpHIDFrames() {
    const uint32_t now = millis();
    while ((uint32_t)(now - last_send_ms) >= USB_TICK_MS) {
        int8_t sx = 0, sy = 0;

        // 누적된 Δ를 한 프레임이 담을 수 있는 범위로 슬라이스
        int32_t send_dx = acc_dx;
        int32_t send_dy = acc_dy;
        sx = clampDeltaToReportRange(send_dx);
        sy = clampDeltaToReportRange(send_dy);

        if (sx != 0 || sy != 0) {
            Mouse.move(sx, sy);     // wheel은 기본 0
            acc_dx -= sx;
            acc_dy -= sy;
        }
        else {
            // 필요 시 keep-alive를 원하면 아래 주석 해제
            // Mouse.move(0, 0);
        }
        last_send_ms += USB_TICK_MS;
    }
}

// 정확히 N바이트 읽기 (대기 중에도 펌프를 계속 돌려서 HID 틱 유지)
bool readExact(EthernetClient& c, uint8_t* buf, size_t n, uint32_t timeout_ms = 500) {
    uint32_t start = millis();
    size_t got = 0;
    while (got < n) {
        if (c.available()) {
            int b = c.read();
            if (b < 0) continue;
            buf[got++] = (uint8_t)b;
        }
        else {
            pumpHIDFrames();  // 대기 중에도 누적 배출
            if ((uint32_t)(millis() - start) > timeout_ms) return false;
            delay(1);
        }
    }
    return true;
}

// ======================= Arduino 표준 =======================
void setup() {
    Ethernet.init(10);          // W5100 SS(=CS) 핀
    Ethernet.begin(mac, ip);
    delay(1000);
    server.begin();

    Keyboard.begin();
    Mouse.begin();

    acc_dx = 0;
    acc_dy = 0;
    last_send_ms = millis();
    running = false;
}

void loop() {
    // 1) 연결 수락/관리
    if (!client || !client.connected()) {
        client = server.available();
    }

    // 2) 수신 처리
    if (client && client.connected()) {
        while (client.available()) {
        int opi = client.read();
        if (opi < 0) break;
        uint8_t op = (uint8_t)opi;

        switch (op) {
            case OP_RUN_ON:
            running = true;
            break;

            case OP_RUN_OFF:
            // 여기에 acc를 0으로 날리면 아직 못 보낸 Δ가 유실됨 → 배출되도록 그대로 둔다.
            running = false;
            break;

            case OP_RELEASE_ALL:
            Keyboard.releaseAll();
            // 마우스 버튼은 click/release API로 제어 (여기선 별도 없음)
            break;

            case OP_SLEEP: {
            uint8_t buf[2];
            if (!readExact(client, buf, 2)) goto framing_error;
            uint16_t ms = (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
            uint32_t t0 = millis();
            while ((uint32_t)(millis() - t0) < ms) {
                pumpHIDFrames();   // 슬립 중에도 HID 틱 유지
                delay(1);
            }
            break;
            }

            case OP_PRESS: {
            uint8_t key;
            if (!readExact(client, &key, 1)) goto framing_error;
            if (running) Keyboard.press(key);
            break;
            }

            case OP_RELEASE: {
            uint8_t key;
            if (!readExact(client, &key, 1)) goto framing_error;
            if (running) Keyboard.release(key);
            break;
            }

            case OP_MOUSEMOVE: {
            // 8바이트: int32 x, int32 y (LE)
            uint8_t buf[8];
            if (!readExact(client, buf, 8)) goto framing_error;

            int32_t x = (int32_t)buf[0] | ((int32_t)buf[1] << 8) | ((int32_t)buf[2] << 16) | ((int32_t)buf[3] << 24);
            int32_t y = (int32_t)buf[4] | ((int32_t)buf[5] << 8) | ((int32_t)buf[6] << 16) | ((int32_t)buf[7] << 24);

            if (running) {
                int32_t nx = acc_dx + x;
                int32_t ny = acc_dy + y;

                if (nx >  ACC_SOFT_LIMIT) nx = ACC_SOFT_LIMIT;
                if (nx < -ACC_SOFT_LIMIT) nx = -ACC_SOFT_LIMIT;
                if (ny >  ACC_SOFT_LIMIT) ny = ACC_SOFT_LIMIT;
                if (ny < -ACC_SOFT_LIMIT) ny = -ACC_SOFT_LIMIT;

                acc_dx = nx;
                acc_dy = ny;
            }

            // MOVE 수신 후 즉시 펌프 (누적분 배출)
            pumpHIDFrames();
            break;
            }

            case OP_MOUSECLICK: {
            // 패킷: [OP_MOUSECLICK][button][ms_lo][ms_hi]
            uint8_t buf[3];
            if (!readExact(client, buf, 3)) goto framing_error;

            uint8_t  button = buf[0];
            uint16_t ms     = (uint16_t)buf[1] | ((uint16_t)buf[2] << 8);

            if (running) {
                // 클릭 전에 남은 이동(acc_dx, acc_dy) 먼저 다 배출해도 됨 (선택)
                while (acc_dx != 0 || acc_dy != 0) {
                pumpHIDFrames();
                }

                // 그냥 Mouse.click 사용 (블로킹, 내부에서 delay(ms))
                Mouse.click(button, ms);
            }
            break;
            }

            default:
            // 알 수 없는 opcode → 프레이밍 오류 가능성, 연결 끊기
            goto framing_error;
        }

        // 스위치 간에도 펌프 (긴 처리 중 틱 손실 방지)
        pumpHIDFrames();
        }
    }

    // 3) 루프 하단에서 최종 캐치업
    pumpHIDFrames();
    return;

framing_error:
    // readExact 타임아웃 또는 알 수 없는 opcode → 프레이밍 오정렬
    // 연결을 끊어서 클라이언트가 재연결하도록 강제
    Keyboard.releaseAll();
    client.stop();
    acc_dx = 0;
    acc_dy = 0;
}
