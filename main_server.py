"""
BMS 메인 서버 v2.2 (UART 통신 통합판)
==================================================
[v2.2 변경사항]
1. UART 통신 모듈 추가 (ESP32 ↔ 라즈베리파이)
2. is_real_mode = True 시 ESP32에서 센서 데이터 수신
3. AI 판단 후 ESP32로 제어 명령 전송
4. 통신 실패 시 자동 fallback (시뮬 모드)
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import asyncio
import json
import random
import os
import joblib
import pandas as pd

# UART 통신용 (실제 모드에서만 사용, 시뮬 모드에서는 import 실패해도 OK)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial 미설치 - 시뮬 모드 전용 (실제 모드 시 'pip install pyserial' 필요)")

# ============================================================
# UART 설정 (실제 하드웨어 연결 시 수정)
# ============================================================
UART_PORT = '/dev/ttyUSB0'   # 라즈베리파이에 ESP32가 USB로 연결된 경우
                              # 또는 '/dev/ttyACM0' (ESP32 native USB)
                              # GPIO UART면 '/dev/serial0'
UART_BAUDRATE = 115200        # ESP32 표준 baud rate
UART_TIMEOUT = 0.5            # 수신 타임아웃 (초)

# ============================================================
# 설정
# ============================================================
PACK_FEATURES = ['Max_Mosfet_T', 'Max_Battery_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_T', 'Battery_T', 'Cell_V', 'Delta_From_Min']

# 발열 모델 (현실 P = I × V_DS_drop 근사)
HEAT_DAC_MOSFET = 1.0    # 2.0 → 1.0 (선형 모델로 바꾸면서 계수 절반)
HEAT_PWM_MOSFET = 0.5    # 그대로 (PWM은 평균 전류 기반이라 변경 없음)

# 자연 방열 (실내 + 약방열판)
COOL_K_MOSFET = 0.014    # 0.020 → 0.014 (통풍 약함)
COOL_K_BATTERY = 0.004   # 0.006 → 0.004

# 안전 임계
MOSFET_T_STOP = 65.0     # 70 → 65 (Op-Amp 70°C 한계 보호)
BATTERY_T_STOP = 55.0

TEMP_AMBIENT = 28.0       
HEAT_BATTERY_CONDUCT = 0.02     
HEAT_BATTERY_INTERNAL = 0.1     

# 자가방전 외란 
SELF_DISCHARGE_NOISE = 0.0003    

app = FastAPI()

# ============================================================
# AI 모델 3개 로드
# ============================================================
try:
    clf = joblib.load('bms_mode_ai.pkl')
    reg_dac = joblib.load('bms_dac_ai.pkl')
    reg_pwm = joblib.load('bms_pwm_ai.pkl')
    AI_LOADED = True
    print("[AI] 모델 3개 로드 완료")
except Exception as e:
    clf = reg_dac = reg_pwm = None
    AI_LOADED = False
    print(f"[WARN] 모델 파일 없음: {e}")

# ============================================================
# UART 통신 (ESP32 ↔ 라즈베리파이)
# ============================================================
uart_conn = None  # 시리얼 객체 (지연 초기화)


def uart_connect():
    """
    UART 포트 연결. 실패 시 None 반환 (시뮬 모드로 fallback).
    실제 모드 첫 진입 시 호출.
    """
    global uart_conn
    if not SERIAL_AVAILABLE:
        return None
    if uart_conn is not None and uart_conn.is_open:
        return uart_conn
    try:
        uart_conn = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=UART_TIMEOUT)
        print(f"[UART] 연결 성공: {UART_PORT} @ {UART_BAUDRATE}")
        return uart_conn
    except Exception as e:
        print(f"[UART] 연결 실패: {e}")
        uart_conn = None
        return None


def uart_read_sensor():
    """
    ESP32에서 센서 데이터 1줄 읽기 (JSON 한 줄).
    형식: {"v":[...], "mt":[...], "bt":[...], "i":[...]}
    실패 시 None 반환.
    """
    if uart_conn is None or not uart_conn.is_open:
        return None
    try:
        # in_waiting > 0이면 데이터 있음. 없으면 기다리지 않고 None 반환 (non-blocking)
        if uart_conn.in_waiting == 0:
            return None
        line = uart_conn.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            return None
        return json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[UART] 데이터 파싱 실패 (무시): {e}")
        return None
    except Exception as e:
        print(f"[UART] 수신 에러: {e}")
        return None


def uart_send_command(mode, dac_vals, pwm_duty):
    """
    ESP32로 제어 명령 전송 (JSON 한 줄).
    형식: {"mode":"DAC", "dac":[1500,800,400,0], "pwm":[0,0,0,0]}
    """
    if uart_conn is None or not uart_conn.is_open:
        return False
    try:
        cmd = {
            "mode": mode,
            "dac": list(dac_vals),
            "pwm": list(pwm_duty),
        }
        msg = json.dumps(cmd) + "\n"
        uart_conn.write(msg.encode('utf-8'))
        uart_conn.flush()
        return True
    except Exception as e:
        print(f"[UART] 송신 에러: {e}")
        return False


# ============================================================
# 시뮬레이션 상태
# ============================================================
is_real_mode = False

sim_state = {
    "v": [4.20, 4.10, 4.05, 3.90],          
    "mosfet_t": [30.0, 30.0, 30.0, 30.0],   
    "battery_t": [30.0, 30.0, 30.0, 30.0],  
    "i": [0.0, 0.0, 0.0, 0.0],              

    "dac_vals": [0, 0, 0, 0],     
    "pwm_duty": [0, 0, 0, 0],     

    "pwm_v": [4.20, 4.10, 4.05, 3.90],
    "pwm_i": [0.0, 0.0, 0.0, 0.0],
    "pwm_mosfet_t": [30.0, 30.0, 30.0, 30.0],
    "pwm_battery_t": [30.0, 30.0, 30.0, 30.0],

    "dac_v": [4.20, 4.10, 4.05, 3.90],
    "dac_i": [0.0, 0.0, 0.0, 0.0],
    "dac_mosfet_t": [30.0, 30.0, 30.0, 30.0],
    "dac_battery_t": [30.0, 30.0, 30.0, 30.0],
}

@app.get("/")
async def get_index():
    if not os.path.exists("index.html"):
        return {"error": "index.html not found"}
    return FileResponse("index.html")

# ============================================================
# AI 판단
# ============================================================
def ai_decide_mode(max_mosfet_t, max_battery_t, pack_delta_v):
    if max_mosfet_t >= MOSFET_T_STOP or max_battery_t >= BATTERY_T_STOP:
        return "STOP", 1.0
    # 2. 전압차 안전 체크 (추가: 0.01V 이하로 평탄화 완료 시 정지)
    if pack_delta_v <= 0.01:
        return "STOP", 1.0

    if AI_LOADED:
        feat = pd.DataFrame([[max_mosfet_t, max_battery_t, pack_delta_v]], columns=PACK_FEATURES)
        proba = clf.predict_proba(feat)[0]
        cls = clf.classes_
        p_dac = float(proba[list(cls).index('DAC')]) if 'DAC' in cls else 0.5
        mode = clf.predict(feat)[0]
        return mode, p_dac
    
    #AI 비로드시 비상용 판단 로직 (간단한 휴리스틱)
    else:
        if max_mosfet_t >= 50 or max_battery_t >= 45 or pack_delta_v <= 0.02:
            return "DAC", 0.9
        elif pack_delta_v >= 0.05:
            return "PWM", 0.1
        else:
            return "PWM", 0.3

def ai_decide_cell_outputs(mode, mosfet_ts, battery_ts, voltages):
    if mode == "STOP":
        return [0, 0, 0, 0], [0, 0, 0, 0]

    min_v = min(voltages)
    deltas = [v - min_v for v in voltages]

    batch = pd.DataFrame([
        [mosfet_ts[i], battery_ts[i], voltages[i], deltas[i]] for i in range(4)
    ], columns=CELL_FEATURES)

    if AI_LOADED:
        if mode == "DAC":
            dac_vals = reg_dac.predict(batch)
            dac_vals = [int(max(0, min(4095, v))) for v in dac_vals]
            pwm_duty = [0, 0, 0, 0]
        else:
            pwm_duty = reg_pwm.predict(batch)
            pwm_duty = [int(max(0, min(255, v))) for v in pwm_duty]
            dac_vals = [0, 0, 0, 0]

    # AI 비로드 시 비상용 코드
    else:
        if mode == "DAC":
            dac_vals = [int(min(4095, deltas[i] / 0.15 * 4095)) for i in range(4)]
            pwm_duty = [0, 0, 0, 0]
        else:
            pwm_duty = [int(min(255, deltas[i] / 0.15 * 255)) for i in range(4)]
            dac_vals = [0, 0, 0, 0]

    return dac_vals, pwm_duty

# ============================================================
# 셀 전압 변화 (물리법칙 강제 적용 - 노이즈 제거)
# ============================================================
def discharge_cell(v, current, smoothness):
    """단일 셀 전압 변화 (상태 오염 방지)"""
    self_discharge = -random.uniform(0, SELF_DISCHARGE_NOISE)
    if current <= 0.001:
        true_v = v + self_discharge
        return min(v, true_v)
    
    decay = current * smoothness * 0.05
    true_v = v - decay + self_discharge
    return min(v, true_v) # 절대로 이전 전압(v)보다 커질 수 없음

def discharge_cells_balanced(cells, currents, smoothness):
    return [discharge_cell(cells[i], currents[i], smoothness) for i in range(len(cells))]

def balance_toward_min(cells, currents, smoothness):
    """가상 우주용 평탄화 (상태 오염 방지)"""
    min_v = min(cells)
    new_cells = []
    
    for i, v in enumerate(cells):
        self_discharge = -random.uniform(0, SELF_DISCHARGE_NOISE)
        if currents[i] <= 0.001:
            true_v = v + self_discharge
            new_cells.append(min(v, true_v))
            continue
            
        diff = v - min_v
        decay = diff * currents[i] * smoothness * 0.004
        true_v = v - decay + self_discharge
        
        # 🌟 핵심 방어 로직: 진짜 전압은 절대로 상승할 수 없음 🌟
        new_cells.append(min(v, true_v))
        
    return new_cells

# ============================================================
# WebSocket
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global is_real_mode

    async def listen_commands():
        global is_real_mode
        try:
            while True:
                msg = await websocket.receive_text()
                if json.loads(msg).get("command") == "toggle_mode":
                    is_real_mode = not is_real_mode
                    print(f"[INFO] 시스템 모드: {'REAL' if is_real_mode else 'SIM'}")
        except Exception:
            pass

    asyncio.create_task(listen_commands())
    pwm_phase = True
    prev_mode = "PWM"

    try:
        prev_real_mode = is_real_mode

        #실제 모드 데이터 수신 받아올 코드
        while True:

            #모드 토클 시 시뮬레이션 재동기화
            if is_real_mode != prev_real_mode:
                print(f"[INFO] 데이터 소스 전환: {'SIM->REAL' if is_real_mode else 'REAL->SIM'}")
                if is_real_mode:
                    # 실제 모드 진입 시 UART 연결 시도
                    uart_connect()
                prev_real_mode = is_real_mode

            if is_real_mode:
                # ----------------------------------------------------------
                # ESP32에서 센서 데이터 수신 (UART)
                # ----------------------------------------------------------
                # 받는 데이터 형식 (ESP32에서 1초마다 보내는 JSON 한 줄):
                #   {"v":[4.20,4.10,4.05,3.90], "mt":[30.5,...], "bt":[29.8,...], "i":[0.05,...]}
                #
                # v  = 셀 전압 4개 (ADS1115)
                # mt = MOSFET 온도 4개 (NTC 서미스터)
                # bt = 배터리 온도 4개 (DS18B20)
                # i  = 셀 밸런싱 전류 4개 (Rsense)
                received = uart_read_sensor()
                if received is not None:
                    # 데이터 도착 -> sim_state 덮어쓰기
                    try:
                        if "v" in received and len(received["v"]) == 4:
                            sim_state["v"] = [float(x) for x in received["v"]]
                        if "mt" in received and len(received["mt"]) == 4:
                            sim_state["mosfet_t"] = [float(x) for x in received["mt"]]
                        if "bt" in received and len(received["bt"]) == 4:
                            sim_state["battery_t"] = [float(x) for x in received["bt"]]
                        if "i" in received and len(received["i"]) == 4:
                            sim_state["i"] = [float(x) for x in received["i"]]
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"[UART] 데이터 형식 오류 (무시): {e}")
                # 데이터 없으면 이전 sim_state 그대로 유지 (1초 주기에서 자연스러움)

            cells_v = sim_state["v"]
            cells_mt = sim_state["mosfet_t"]
            cells_bt = sim_state["battery_t"]

            pack_dv = max(cells_v) - min(cells_v)
            max_mosfet_t = max(cells_mt)
            max_battery_t = max(cells_bt)
            min_v = min(cells_v)

            mode, p_dac = ai_decide_mode(max_mosfet_t, max_battery_t, pack_dv)
            dac_vals, pwm_duty = ai_decide_cell_outputs(mode, cells_mt, cells_bt, cells_v)

            sim_state["dac_vals"] = dac_vals
            sim_state["pwm_duty"] = pwm_duty

                       
            # ----------------------------------------------------------
            # [실제 모드] AI 출력을 ESP32로 전송 (UART)
            # ----------------------------------------------------------
            # 보내는 형식 (ESP32가 받아서 DAC/PWM 출력 적용):
            #   {"mode":"DAC", "dac":[1500,800,400,0], "pwm":[0,0,0,0]}
            #
            # mode: "PWM" / "DAC" / "STOP"
            # dac : 셀별 MCP4725 출력값 (0~4095)
            # pwm : 셀별 듀티비 (0~255), STOP 시 모두 0
            if is_real_mode:
                uart_send_command(mode, dac_vals, pwm_duty)
            #ㄴ 송신 실패해도 시뮬은 계속 (안전한 fallback)

            #모드 전환 시 시뮬레이션 재동기화
            if mode != prev_mode:
                sim_state["pwm_mosfet_t"] = list(cells_mt)
                sim_state["dac_mosfet_t"] = list(cells_mt)
                sim_state["pwm_battery_t"] = list(cells_bt)
                sim_state["dac_battery_t"] = list(cells_bt)
                sim_state["pwm_v"] = list(cells_v)
                sim_state["dac_v"] = list(cells_v)
            prev_mode = mode

            pwm_phase = not pwm_phase
            cell_currents = [0.0] * 4
            cell_dt_mosfet = [0.0] * 4
            cell_dt_battery = [0.0] * 4

            for i in range(4):
                if mode == "STOP":
                    cell_currents[i] = 0.0
                elif mode == "PWM":
                    duty_ratio = pwm_duty[i] / 255.0
                    avg_current = duty_ratio * 0.5
                    if random.random() < duty_ratio:
                        cell_currents[i] = 0.5
                    else:
                        cell_currents[i] = 0.0
                    cell_dt_mosfet[i] = avg_current * HEAT_PWM_MOSFET + random.uniform(-0.02, 0.02)
                    cell_dt_battery[i] = (cells_mt[i] - cells_bt[i]) * HEAT_BATTERY_CONDUCT + avg_current * HEAT_BATTERY_INTERNAL
                else: 
                    target_i = (dac_vals[i] / 4095.0) * 0.4
                    cell_currents[i] = target_i
                    #미세 전류 구간에서(1A 미만) 선형 제어 발열은 전류랑에 거의 정비례하므로 제곱 안 함
                    cell_dt_mosfet[i] = target_i * HEAT_DAC_MOSFET + random.uniform(-0.03, 0.03)
                    cell_dt_battery[i] = (cells_mt[i] - cells_bt[i]) * HEAT_BATTERY_CONDUCT + target_i * HEAT_BATTERY_INTERNAL

            sim_state["i"] = cell_currents

            for i in range(4):
                cells_mt[i] += cell_dt_mosfet[i]
                cells_mt[i] -= COOL_K_MOSFET * (cells_mt[i] - TEMP_AMBIENT)
                cells_mt[i] = max(TEMP_AMBIENT, cells_mt[i])

                cells_bt[i] += cell_dt_battery[i]
                cells_bt[i] -= COOL_K_BATTERY * (cells_bt[i] - TEMP_AMBIENT)
                cells_bt[i] = max(TEMP_AMBIENT, cells_bt[i])

            # 실제 모드일 때는 위에서 받은 데이터로 sim_state 이미 갱신됨 -> 시뮬 계산 스킵
            # 시뮬 모드일 때만 발열/평탄화 시뮬 적용
            if not is_real_mode:
                if mode == "PWM":
                    cells_v_new = balance_toward_min(cells_v, cell_currents, smoothness=1.0)
                elif mode == "DAC":
                    cells_v_new = balance_toward_min(cells_v, cell_currents, smoothness=0.5)
                else: 
                    cells_v_new = [v - random.uniform(0, SELF_DISCHARGE_NOISE) for v in cells_v]

                sim_state["v"] = [max(3.0, min(4.25, v)) for v in cells_v_new]
                sim_state["mosfet_t"] = cells_mt
                sim_state["battery_t"] = cells_bt
            # 실제 모드: sim_state는 ESP32 데이터 그대로 사용 (덮어쓰기 안 함)

            # ============================================================
            # 6. 가상 우주
            # ============================================================
            if mode == "PWM":
                sim_state["pwm_v"] = list(sim_state["v"])
                sim_state["pwm_i"] = list(cell_currents)
                sim_state["pwm_mosfet_t"] = list(cells_mt)
                sim_state["pwm_battery_t"] = list(cells_bt)

                dac_virt_min = min(sim_state["dac_v"])
                virt_dv = max(sim_state["dac_v"]) - dac_virt_min
                dac_virt_i_target = min(0.4, virt_dv * 2.5)
                dac_virt_i_base = max(0.30, dac_virt_i_target)  

                if virt_dv > 0.001:
                    dac_currents = [max(0.05, dac_virt_i_base * (v - dac_virt_min) / virt_dv) for v in sim_state["dac_v"]]
                else:
                    dac_currents = [0.05] * 4  

                for i in range(4):
                    ti = dac_currents[i]
                    #미세 전류 구간에서(1A 미만) 선형 제어 발열은 전류랑에 거의 정비례하므로 제곱 안 하고 곱으로만 처리
                    dt_m = ti * HEAT_DAC_MOSFET + random.uniform(-0.03, 0.03)
                    dt_b = ((sim_state["dac_mosfet_t"][i] - sim_state["dac_battery_t"][i]) * HEAT_BATTERY_CONDUCT + ti * HEAT_BATTERY_INTERNAL)
                    sim_state["dac_mosfet_t"][i] = max(TEMP_AMBIENT, sim_state["dac_mosfet_t"][i] + dt_m - COOL_K_MOSFET * (sim_state["dac_mosfet_t"][i] - TEMP_AMBIENT))
                    sim_state["dac_battery_t"][i] = max(TEMP_AMBIENT, sim_state["dac_battery_t"][i] + dt_b - COOL_K_BATTERY * (sim_state["dac_battery_t"][i] - TEMP_AMBIENT))

                sim_state["dac_v"] = balance_toward_min(sim_state["dac_v"], dac_currents, smoothness=0.5)
                sim_state["dac_i"] = list(dac_currents)

            elif mode == "DAC":
                sim_state["dac_v"] = list(sim_state["v"])
                sim_state["dac_i"] = list(cell_currents)
                sim_state["dac_mosfet_t"] = list(cells_mt)
                sim_state["dac_battery_t"] = list(cells_bt)

                pwm_virt_min = min(sim_state["pwm_v"])
                pwm_virt_dv = max(sim_state["pwm_v"]) - pwm_virt_min

                pwm_currents = [0.0] * 4
                for i in range(4):
                    if pwm_virt_dv > 0.001:
                        virt_duty = 0.3 + 0.7 * (sim_state["pwm_v"][i] - pwm_virt_min) / pwm_virt_dv
                    else:
                        virt_duty = 0.3
                    pwm_currents[i] = 0.5 if random.random() < virt_duty else 0.0

                    avg_i = virt_duty * 0.5
                    dt_m = avg_i * HEAT_PWM_MOSFET + random.uniform(-0.02, 0.02)
                    dt_b = ((sim_state["pwm_mosfet_t"][i] - sim_state["pwm_battery_t"][i]) * HEAT_BATTERY_CONDUCT + avg_i * HEAT_BATTERY_INTERNAL)
                    sim_state["pwm_mosfet_t"][i] = max(TEMP_AMBIENT, sim_state["pwm_mosfet_t"][i] + dt_m - COOL_K_MOSFET * (sim_state["pwm_mosfet_t"][i] - TEMP_AMBIENT))
                    sim_state["pwm_battery_t"][i] = max(TEMP_AMBIENT, sim_state["pwm_battery_t"][i] + dt_b - COOL_K_BATTERY * (sim_state["pwm_battery_t"][i] - TEMP_AMBIENT))

                avg_currents = [pwm_currents[i] if pwm_currents[i] > 0 else 0.05 for i in range(4)]
                sim_state["pwm_v"] = balance_toward_min(sim_state["pwm_v"], avg_currents, smoothness=1.0)
                sim_state["pwm_i"] = list(pwm_currents)

            else:  
                sim_state["pwm_v"] = list(sim_state["v"])
                sim_state["dac_v"] = list(sim_state["v"])
                sim_state["pwm_i"] = [0.0] * 4
                sim_state["dac_i"] = [0.0] * 4
                sim_state["pwm_mosfet_t"] = list(cells_mt)
                sim_state["pwm_battery_t"] = list(cells_bt)
                sim_state["dac_mosfet_t"] = list(cells_mt)
                sim_state["dac_battery_t"] = list(cells_bt)

            # ============================================================
            # 7. WebSocket payload (출력 시에만 센서 흔들림 연출)
            # ============================================================
            def add_display_noise(v_list, is_pwm=False):
                # 출력용 가짜 노이즈 (상태 오염 X)
                n_range = 0.015 if is_pwm else 0.001
                return [round(v + random.uniform(-n_range, n_range), 3) for v in v_list]

            payload = {
                "system_mode": "REAL_HARDWARE" if is_real_mode else "DUMMY_SIMULATION",
                "ai_loaded": AI_LOADED,
                "ai_mode": mode,
                "p_dac": round(p_dac, 3),
                "pack_delta_v": round(pack_dv, 4),
                "pack_min_v": round(min_v, 3),
                "real": {
                    "v":         add_display_noise(sim_state["v"], is_pwm=(mode=="PWM")),
                    "i":         [round(c, 3) for c in cell_currents],
                    "mosfet_t":  [round(t, 2) for t in cells_mt],
                    "battery_t": [round(t, 2) for t in cells_bt],
                    "dac_vals":  list(dac_vals),
                    "pwm_duty":  list(pwm_duty),
                },
                "pwm": {
                    "v":         add_display_noise(sim_state["pwm_v"], is_pwm=True),
                    "i":         [round(c, 3) for c in sim_state["pwm_i"]],
                    "mosfet_t":  [round(t, 2) for t in sim_state["pwm_mosfet_t"]],
                    "battery_t": [round(t, 2) for t in sim_state["pwm_battery_t"]],
                },
                "dac": {
                    "v":         add_display_noise(sim_state["dac_v"], is_pwm=False),
                    "i":         [round(c, 3) for c in sim_state["dac_i"]],
                    "mosfet_t":  [round(t, 2) for t in sim_state["dac_mosfet_t"]],
                    "battery_t": [round(t, 2) for t in sim_state["dac_battery_t"]],
                },
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)

    except Exception as e:
        print(f"[ERROR] 웹소켓: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
