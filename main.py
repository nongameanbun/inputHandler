from fastapi import FastAPI, APIRouter
from typing import Literal, Optional
from base import Input_Q, q_lock
from HID import human_mouse
from threading import Thread
import random
import uvicorn

app = FastAPI(title="Input Handler API", description="게임 입력 처리 서버")

def Rdelay(base_ms) :
    delay = base_ms * random.uniform(1, 1.2)
    return int(delay)

def send_and_get_Rdelay(base_ms) :
    rd = Rdelay(base_ms)
    Input_Q.append(f"delay {rd}")
    return rd

def send_and_get_Rdelay_raw(base_ms) :
    """delay 값만 계산하여 반환 (큐에 넣지 않음, q_lock 내부에서 직접 삽입용)"""
    return Rdelay(base_ms)

# ───────────── Input API Endpoints ─────────────

@app.post("/on", summary="입력 제어 시작", tags=["키 입력"])
async def turn_on():
    Input_Q.append("on")
    return {"resp": 0}

@app.post("/off", summary="입력 제어 종료", tags=["키 입력"])
async def turn_off():
    Input_Q.append("off")
    return {"resp": 0}

@app.post("/press_key", summary="키 누르기", tags=["키 입력"])
async def press_key(key_name: str):
    Input_Q.append(f"press_key {key_name}")
    return {"resp": 0}

@app.post("/delay", summary="딜레이 추가", tags=["키 입력"])
async def delay(delay: int):
    final_delay_ms = send_and_get_Rdelay(delay)
    return {"resp": final_delay_ms}

@app.post("/release_key", summary="키 떼기", tags=["키 입력"])
async def release_key(key_name: str):
    Input_Q.append(f"release_key {key_name}")
    return {"resp": 0}

@app.post("/releaseAll", summary="모든 키 떼기", tags=["키 입력"])
async def release_all():
    Input_Q.append("releaseAll")
    return {"resp": 0}

@app.post("/press_key_with_delay", summary="키 누르기 + 딜레이 + 해제")
async def press_key_with_delay(key_name: str, delay: int):
    with q_lock:
        rd = send_and_get_Rdelay_raw(delay)
        Input_Q.append(f"press_key {key_name}")
        Input_Q.append(f"delay {rd}")
        Input_Q.append(f"release_key {key_name}")

    return {"resp": rd}


@app.post("/press_two_key", summary="2키 조합 입력")
async def press_two_key(key1: str, key2: str):
    final_delay_ms = 0

    with q_lock:
        rd = send_and_get_Rdelay_raw(50)
        Input_Q.append(f"delay {rd}")
        final_delay_ms += rd

        Input_Q.append(f"press_key {key1}")

        rd = send_and_get_Rdelay_raw(300)
        Input_Q.append(f"delay {rd}")
        final_delay_ms += rd

        Input_Q.append(f"press_key {key2}")

        rd = send_and_get_Rdelay_raw(100)
        Input_Q.append(f"delay {rd}")
        final_delay_ms += rd

        Input_Q.append(f"release_key {key2}")

        rd = send_and_get_Rdelay_raw(100)
        Input_Q.append(f"delay {rd}")
        final_delay_ms += rd

        Input_Q.append(f"release_key {key1}")

        rd = send_and_get_Rdelay_raw(50)
        Input_Q.append(f"delay {rd}")
        final_delay_ms += rd

    return {"resp": final_delay_ms}

mouse_router = APIRouter(prefix="/mouse", tags=["마우스"])

@mouse_router.post("/dmove", summary="마우스 이동")
async def mouse_dmove(dx: int, dy: int):
    Input_Q.append(f"dMouse {dx} {dy}")
    return {"resp": 0}

@mouse_router.post("/move", summary="마우스 이동")
async def mouse_move(x: int, y: int):
    import time as _t
    t0 = _t.perf_counter()
    print(f"[mouse_move] START x={x}, y={y}")
    human_mouse.move_to_px_human(x, y)
    t1 = _t.perf_counter()
    print(f"[mouse_move] move_to_px_human took {(t1-t0)*1000:.1f}ms")

    return {"resp": 0}
    
@mouse_router.post("/click", summary="마우스 클릭")
async def mouse_click(click_mode: Literal["left", "right", "middle"], delay: int, x: Optional[int] = None, y: Optional[int] = None):
    final_delay_ms = 0

    if x is not None and y is not None:
        human_mouse.move_to_px_human(x, y)
        # move 완료 후 Arduino 측 잔여 acc 배출 대기 (30ms)
        import time as _t
        _t.sleep(0.03)

    mode_dict = {"left": 1, "right": 2, "middle": 4}
    Input_Q.append(f"cMouse {mode_dict[click_mode]} {delay}")
    final_delay_ms += send_and_get_Rdelay(delay + 20)

    return {"resp": final_delay_ms}

app.include_router(mouse_router)

# run app
if __name__ == "__main__":
    try:
        from consumer import input_queue_consumer
        from dotenv import load_dotenv
        import os

        load_dotenv()
        port_num = int(os.getenv("inputHandler_API_PORT"))
        assert port_num
    except Exception as e:
        print(e)
        exit(1)
    
    Thread(target=input_queue_consumer, daemon=True).start()
    Input_Q.append("releaseAll")
    Input_Q.append("on")
    uvicorn.run(app, host="0.0.0.0", port=port_num, log_level="warning")