"""
A²-BMS 메인 서버 (소전류 실측 + 듀얼 MOSFET 측정 기반 AI)
============================================================
[시스템 개요]
ESP32가 측정한 셀 상태를 받아 AI가 밸런싱 모드(PWM/DAC)와 출력값을 결정,
다시 ESP32로 제어 명령을 보낸다. 동시에 웹 대시보드로 실시간 상태를 송신한다.

[측정/입력 - 듀얼 MOSFET]
  · 셀 전압 (V)
  · Q_PWM 모스펫 온도   : ESP32 GPIO로 직접 제어하는 스위칭 MOSFET
  · Q_LINEAR(DAC) 모스펫 온도 : I²C→DAC→Op-Amp로 제어하는 선형 MOSFET
  ※ 배터리 온도는 측정 안 함 — 소전류(~0.5A)에서 셀 발열 I²R≈0.006W로
    무시 가능, "배터리가 타지 않는다"는 신뢰 기반.

[전류 파형]
  실제로 전류 센서값을 쓰지 않고, AI가 결정한 모드/출력값으로부터
  전류를 역산해 웹에 파형으로 표시한다 (PWM=펄스, DAC=선형).

[AI 모델 3개]
  bms_mode_ai.pkl (모드분류) / bms_dac_ai.pkl / bms_pwm_ai.pkl

[모드별 발열 원리 - AI가 학습한 핵심]
  · DAC 모드: Q_LINEAR가 선형영역 손실(P=V_DS×I)로 크게 발열
  · PWM 모드: Q_PWM이 스위칭손실(P=I²×R_DS_on)이라 발열 미미
  → DAC 모스펫이 뜨거우면 AI가 PWM으로 전환해 발열을 분산
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import asyncio
import json
import random
import os
import csv
import time
from datetime import datetime
import joblib
import pandas as pd

# UART 통신용 (실제 하드웨어 연결 시에만 사용, 없으면 시뮬레이션)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial 미설치 - 시뮬 모드 전용")

# ============================================================
# 설정값
# ============================================================
# UART (라즈베리파이 ↔ ESP32)
UART_PORT = '/dev/ttyUSB0'
UART_BAUDRATE = 115200
UART_TIMEOUT = 0.5

# 실측 로깅 (재학습용 데이터 축적)
LOGGING_ENABLED = True
LOG_DIR = 'logs'
# 누적 로그 파일 (매번 새 파일 X, 한 파일에 계속 append)
LOG_FILE = 'logs/real_data_accumulated.csv'

# 배포 환경 구분 (Render = 클라우드, LOCAL = 라즈베리파이/PC)
# 환경변수 DEPLOY_ENV로 제어. Render에선 자동학습 비활성(파일 휘발+자원 제한)
DEPLOY_ENV = os.environ.get('DEPLOY_ENV', 'RENDER')  # 'LOCAL' 또는 'RENDER'
# 로컬에서만 STOP 시 자동 재학습 가능
AUTO_RETRAIN_ON_STOP = (DEPLOY_ENV == 'LOCAL')

# AI feature 컬럼 (학습 때와 반드시 동일한 순서)
PACK_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Cell_V', 'Delta_From_Min']

# --- 발열 모델 계수 (시뮬레이션 fallback 시 사용) ---
# DAC 모스펫: 선형손실 커서 발열 계수 큼
HEAT_DAC_MOSFET = 2.0
# PWM 모스펫: 스위칭이라 발열 계수 작음 (DAC의 약 1/4)
HEAT_PWM_MOSFET = 0.5
# 자연 방열 계수
COOL_K_MOSFET = 0.014

# 안전 임계
MOSFET_T_STOP = 65.0      # 모스펫 65°C 넘으면 강제 STOP (Op-Amp 보호)
TEMP_AMBIENT = 28.0       # 상온

# PCB 열 결합 (이웃 셀 모스펫 발열이 약간 전이)
PCB_HEAT_COUPLING = 0.015

# 셀별 자가방전 속도 (V/sec) — 시뮬레이션용, 1회 초기화
SELF_DISCHARGE_RATES = [random.uniform(0.0000003, 0.0000008) for _ in range(4)]
MEASUREMENT_NOISE = 0.0005   # ±0.5mV 측정 노이즈

app = FastAPI()

# Render 등 클라우드 배포 시 실제 하드웨어 없으므로 기본 시뮬 모드
is_real_mode = False

# ============================================================
# AI 모델 3개 로드 (없으면 룰 기반 fallback)
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
    print(f"[WARN] 모델 없음 (룰 기반 가동): {e}")

# ============================================================
# UART 통신
# ============================================================
uart_conn = None

def uart_connect():
    """UART 포트 연결. 실패 시 None (시뮬 모드로 fallback)."""
    global uart_conn
    if not SERIAL_AVAILABLE:
        return None
    try:
        uart_conn = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=UART_TIMEOUT)
        print(f"[UART] 연결 성공: {UART_PORT}")
        return uart_conn
    except Exception as e:
        print(f"[UART] 연결 실패 (시뮬 모드): {e}")
        return None


def uart_read_sensor():
    """
    ESP32에서 센서 JSON 1줄 읽기.
    형식: {"v":[...], "mt":[...], "dt":[...], "i":[...]}
      v  = 셀 전압
      mt = PWM 모스펫 온도 (Q_PWM)
      dt = DAC 모스펫 온도 (Q_LINEAR)
      i  = 전류 (있으면 참고, 없어도 됨)
    ※ ESP32 펌웨어가 아직 'bt'(배터리)만 보내는 과도기면 mt만 받고 dt는 추정.
    """
    if uart_conn is None or not uart_conn.is_open:
        return None
    try:
        if uart_conn.in_waiting == 0:
            return None
        line = uart_conn.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            return None
        return json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[UART] 파싱 실패 (무시): {e}")
        return None
    except Exception as e:
        print(f"[UART] 수신 에러: {e}")
        return None


def uart_send_command(mode, dac_vals, pwm_duty):
    """
    ESP32로 제어 명령 송신.
    형식: {"mode":"DAC", "dac":[1500,...], "pwm":[0,...]}
    """
    if uart_conn is None or not uart_conn.is_open:
        return
    try:
        cmd = {"mode": mode, "dac": list(dac_vals), "pwm": list(pwm_duty)}
        uart_conn.write((json.dumps(cmd) + "\n").encode('utf-8'))
    except Exception as e:
        print(f"[UART] 송신 에러: {e}")


# ============================================================
# 실측 데이터 로깅 (재학습용 CSV 축적)
# ============================================================
logger_state = {'file': None, 'writer': None, 'start_time': None, 'tick': 0}

LOG_HEADER = [
    'timestamp', 'tick',
    'pwm_mosfet_t_0', 'pwm_mosfet_t_1', 'pwm_mosfet_t_2', 'pwm_mosfet_t_3',
    'dac_mosfet_t_0', 'dac_mosfet_t_1', 'dac_mosfet_t_2', 'dac_mosfet_t_3',
    'v_0', 'v_1', 'v_2', 'v_3',
    'pack_delta_v', 'ai_mode', 'p_dac',
    'dac_0', 'dac_1', 'dac_2', 'dac_3',
    'pwm_0', 'pwm_1', 'pwm_2', 'pwm_3',
]

def init_logger():
    """
    누적 로그 파일 열기.
    - 파일이 이미 있으면 append 모드로 이어서 기록 (헤더 안 씀)
    - 없으면 새로 만들고 헤더 기록
    → 여러 번 실행해도 한 파일에 계속 쌓임 (재학습용 데이터 축적)
    """
    if not LOGGING_ENABLED:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)
    # append 모드로 열기
    logger_state['file'] = open(LOG_FILE, 'a', newline='', encoding='utf-8')
    logger_state['writer'] = csv.writer(logger_state['file'])
    # 새 파일이면 헤더 기록
    if not file_exists:
        logger_state['writer'].writerow(LOG_HEADER)
        logger_state['file'].flush()
    logger_state['start_time'] = time.time()
    print(f"[LOG] 누적 로깅 → {LOG_FILE} ({'이어쓰기' if file_exists else '새 파일'})")


def log_tick(state, mode, p_dac, dac_vals, pwm_duty):
    if not LOGGING_ENABLED or logger_state['writer'] is None:
        return
    logger_state['tick'] += 1
    pwm_t = state['pwm_mosfet_t']
    dac_t = state['dac_mosfet_t']
    v = state['v']
    pdv = max(v) - min(v)
    row = [
        datetime.now().isoformat(), logger_state['tick'],
        round(pwm_t[0], 2), round(pwm_t[1], 2), round(pwm_t[2], 2), round(pwm_t[3], 2),
        round(dac_t[0], 2), round(dac_t[1], 2), round(dac_t[2], 2), round(dac_t[3], 2),
        round(v[0], 4), round(v[1], 4), round(v[2], 4), round(v[3], 4),
        round(pdv, 4), mode, round(p_dac, 3),
        dac_vals[0], dac_vals[1], dac_vals[2], dac_vals[3],
        pwm_duty[0], pwm_duty[1], pwm_duty[2], pwm_duty[3],
    ]
    logger_state['writer'].writerow(row)
    logger_state['file'].flush()


def close_logger():
    if logger_state['file']:
        logger_state['file'].close()
        print("[LOG] 로깅 종료")


def trigger_retrain():
    """
    누적 로그 기반 재학습 실행 (로컬 전용).
    train_system.py를 호출 → 합성 + 실측로그(물리규칙 재라벨) 병합 학습.
    Render에선 호출 안 됨 (AUTO_RETRAIN_ON_STOP=False).
    """
    if not AUTO_RETRAIN_ON_STOP:
        print("[재학습] 비활성 환경 (RENDER) - 건너뜀")
        return False
    if not os.path.exists('train_system.py'):
        print("[재학습] train_system.py 없음 - 건너뜀")
        return False
    try:
        # 로그 파일 먼저 flush (최신 데이터 반영)
        if logger_state['file']:
            logger_state['file'].flush()
        import subprocess
        print("[재학습] 시작 - train_system.py 실행 (합성+실측 재라벨 병합)...")
        result = subprocess.run(
            ['python', 'train_system.py'],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout[-500:] if result.stdout else "")
        if result.returncode == 0:
            print("[재학습] 완료 - 새 .pkl 생성됨. 서버 재시작 시 반영")
            return True
        else:
            print(f"[재학습] 실패: {result.stderr[-300:]}")
            return False
    except Exception as e:
        print(f"[재학습] 에러: {e}")
        return False


# ============================================================
# 웹 페이지 서빙
# ============================================================
@app.get("/")
async def get_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found"}


# ============================================================
# AI 판단 1: 모드 분류 (PWM/DAC/STOP)
# ============================================================
def ai_decide_mode(max_pwm_t, max_dac_t, pack_delta_v):
    """
    팩 상태(두 모스펫 최고온도 + 전압편차)로 밸런싱 모드 결정.
    - 모스펫 과열 시 STOP (안전 우선)
    - 평탄화 완료(편차 미미) 시 STOP
    - 그 외 AI 추론 (또는 룰 fallback)
    """
    # 안전: 둘 중 하나라도 임계 초과면 정지
    if max_pwm_t >= MOSFET_T_STOP or max_dac_t >= MOSFET_T_STOP:
        return "STOP", 1.0
    # 평탄화 완료
    if pack_delta_v <= 0.01:
        return "STOP", 1.0

    if AI_LOADED:
        feat = pd.DataFrame([[max_pwm_t, max_dac_t, pack_delta_v]], columns=PACK_FEATURES)
        proba = clf.predict_proba(feat)[0]
        cls = clf.classes_
        p_dac = float(proba[list(cls).index('DAC')]) if 'DAC' in cls else 0.5
        mode = clf.predict(feat)[0]
        return mode, p_dac
    else:
        # 룰 기반 fallback: DAC 모스펫 뜨겁거나 편차 작으면 신중하게
        if max_dac_t >= 50.0 or pack_delta_v <= 0.02:
            return "DAC", 0.9
        elif pack_delta_v >= 0.05:
            return "PWM", 0.1
        else:
            return "PWM", 0.3


# ============================================================
# AI 판단 2: 셀별 출력값 (DAC값 0~4095 / PWM듀티 0~255)
# ============================================================
def ai_decide_cell_outputs(mode, pwm_ts, dac_ts, voltages):
    if mode == "STOP":
        return [0, 0, 0, 0], [0, 0, 0, 0]

    min_v = min(voltages)
    deltas = [v - min_v for v in voltages]
    # 4셀을 한 batch로 추론
    batch = pd.DataFrame([
        [pwm_ts[i], dac_ts[i], voltages[i], deltas[i]] for i in range(4)
    ], columns=CELL_FEATURES)

    if AI_LOADED:
        if mode == "DAC":
            vals = reg_dac.predict(batch)
            dac_vals = [int(max(0, min(4095, v))) for v in vals]
            pwm_duty = [0, 0, 0, 0]
        else:  # PWM
            vals = reg_pwm.predict(batch)
            pwm_duty = [int(max(0, min(255, v))) for v in vals]
            dac_vals = [0, 0, 0, 0]
    else:
        # 룰 fallback: 편차 비례
        if mode == "DAC":
            dac_vals = [int(min(4095, deltas[i] / 0.15 * 4095)) for i in range(4)]
            pwm_duty = [0, 0, 0, 0]
        else:
            pwm_duty = [int(min(255, deltas[i] / 0.15 * 255)) for i in range(4)]
            dac_vals = [0, 0, 0, 0]

    return dac_vals, pwm_duty


# ============================================================
# 시뮬레이션 헬퍼 (실제 데이터 없을 때 fallback)
# ============================================================
def balance_toward_min(cells, currents, smoothness):
    """전류 흘린 셀의 전압을 최소 셀 쪽으로 수렴시킴 (밸런싱 모사)"""
    min_v = min(cells)
    new_cells = []
    for i, v in enumerate(cells):
        # 전류가 흐르면 그만큼 방전 (높은 셀일수록 빨리 내려감)
        drop = currents[i] * 0.0002 * smoothness
        nv = v - drop
        # 자가방전
        nv -= SELF_DISCHARGE_RATES[i]
        new_cells.append(max(min_v - 0.001, nv))
    return new_cells


def add_measurement_noise(cells_v):
    return [v + random.uniform(-MEASUREMENT_NOISE, MEASUREMENT_NOISE) for v in cells_v]


def apply_pcb_heat_coupling(temps):
    """이웃 셀 모스펫 발열이 PCB 통해 약하게 전이"""
    avg = sum(temps) / len(temps)
    return [t + (avg - t) * PCB_HEAT_COUPLING for t in temps]


def add_display_noise(v_list, is_pwm=False):
    """웹 표시용 미세 노이즈 (PWM은 스위칭 리플로 약간 더 큼)"""
    amp = 0.004 if is_pwm else 0.0015
    return [round(v + random.uniform(-amp, amp), 4) for v in v_list]


# ============================================================
# 시뮬레이션 상태 (실제 데이터 없을 때만 사용)
# ============================================================
sim_state = {
    "v": [4.20, 4.10, 4.05, 3.90],          # 셀 전압
    "pwm_mosfet_t": [30.0, 30.0, 30.0, 30.0],  # Q_PWM 온도
    "dac_mosfet_t": [30.0, 30.0, 30.0, 30.0],  # Q_LINEAR 온도
    "i": [0.0, 0.0, 0.0, 0.0],              # 전류 (계산값)
    "dac_vals": [0, 0, 0, 0],
    "pwm_duty": [0, 0, 0, 0],
}


# ============================================================
# WebSocket 메인 루프
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global is_real_mode

    # 클라이언트 → 서버 명령 수신 (실제/시뮬 토글, 재학습)
    async def listen_commands():
        global is_real_mode
        try:
            while True:
                msg = await websocket.receive_text()
                cmd = json.loads(msg).get("command")
                if cmd == "toggle_mode":
                    is_real_mode = not is_real_mode
                    print(f"[INFO] 모드: {'REAL' if is_real_mode else 'SIM'}")
                elif cmd == "retrain":
                    # STOP 후 누적 로그로 재학습 (로컬 전용)
                    print("[INFO] 재학습 명령 수신")
                    ok = trigger_retrain()
                    try:
                        await websocket.send_text(json.dumps({
                            "event": "retrain_done", "success": ok
                        }))
                    except Exception:
                        pass
        except Exception:
            pass

    asyncio.create_task(listen_commands())
    prev_mode = "PWM"

    if logger_state['file'] is None:
        init_logger()

    if is_real_mode:
        uart_connect()

    try:
        while True:
            # ===== 1. 데이터 획득 =====
            if is_real_mode:
                received = uart_read_sensor()
                if received:
                    try:
                        if "v" in received and len(received["v"]) == 4:
                            sim_state["v"] = [float(x) for x in received["v"]]
                        # mt = PWM 모스펫 온도
                        if "mt" in received and len(received["mt"]) == 4:
                            sim_state["pwm_mosfet_t"] = [float(x) for x in received["mt"]]
                        # dt = DAC 모스펫 온도
                        if "dt" in received and len(received["dt"]) == 4:
                            sim_state["dac_mosfet_t"] = [float(x) for x in received["dt"]]
                        # 전류: ESP32가 R_SENSE로 측정한 실측값 (있으면 사용)
                        if "i" in received and len(received["i"]) == 4:
                            sim_state["i_real"] = [float(x) for x in received["i"]]
                        else:
                            sim_state["i_real"] = None
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"[UART] 형식 오류 (무시): {e}")
                # 데이터 없으면 이전 상태 유지

            cells_v = sim_state["v"]
            pwm_ts = sim_state["pwm_mosfet_t"]
            dac_ts = sim_state["dac_mosfet_t"]

            pack_dv = max(cells_v) - min(cells_v)
            max_pwm_t = max(pwm_ts)
            max_dac_t = max(dac_ts)
            min_v = min(cells_v)

            # ===== 2. AI 판단 =====
            mode, p_dac = ai_decide_mode(max_pwm_t, max_dac_t, pack_dv)
            dac_vals, pwm_duty = ai_decide_cell_outputs(mode, pwm_ts, dac_ts, cells_v)
            sim_state["dac_vals"] = dac_vals
            sim_state["pwm_duty"] = pwm_duty

            # ===== 3. ESP32로 명령 송신 (실제 모드) =====
            if is_real_mode:
                uart_send_command(mode, dac_vals, pwm_duty)

            # ===== 4. 전류 계산 (전류 센서 없으므로 모드/출력에서 역산) =====
            # PWM: 듀티비 확률로 0↔0.5A 펄스 (웹에 펄스 파형 표시)
            # DAC: DAC값 비례 연속 전류 (선형 파형)
            cell_currents = [0.0] * 4
            for i in range(4):
                if mode == "STOP":
                    cell_currents[i] = 0.0
                elif mode == "PWM":
                    duty_ratio = pwm_duty[i] / 255.0
                    if random.random() < duty_ratio:
                        cell_currents[i] = 0.5 + random.uniform(-0.01, 0.01)  # ON
                    else:
                        cell_currents[i] = 0.0   # OFF
                else:  # DAC
                    cell_currents[i] = (dac_vals[i] / 4095.0) * 0.4 + random.uniform(-0.003, 0.003)
                    cell_currents[i] = max(0.0, cell_currents[i])
            sim_state["i"] = cell_currents

            # 실제 모드 + ESP32가 실측 전류를 보냈으면 그 값 우선 사용
            # (R_SENSE로 측정한 진짜 전류 → 웹에 실측 파형 표시)
            i_for_display = cell_currents
            if is_real_mode and sim_state.get("i_real") is not None:
                i_for_display = sim_state["i_real"]

            # ===== 5. 시뮬레이션 모드: 다음 상태 갱신 =====
            # (실제 모드면 ESP32 측정값을 그대로 쓰므로 갱신 안 함)
            if not is_real_mode:
                # 발열 계산 — 평균 전류 기준 (PWM은 펄스 평균)
                for i in range(4):
                    if mode == "PWM":
                        avg_i = (pwm_duty[i] / 255.0) * 0.5
                        # PWM 모스펫만 발열 (스위칭손실, 작음), DAC 모스펫은 식음
                        pwm_ts[i] += avg_i * HEAT_PWM_MOSFET + random.uniform(-0.02, 0.02)
                        dac_ts[i] -= COOL_K_MOSFET * (dac_ts[i] - TEMP_AMBIENT)
                    elif mode == "DAC":
                        avg_i = (dac_vals[i] / 4095.0) * 0.4
                        # DAC 모스펫만 발열 (선형손실, 큼), PWM 모스펫은 식음
                        dac_ts[i] += avg_i * HEAT_DAC_MOSFET + random.uniform(-0.03, 0.03)
                        pwm_ts[i] -= COOL_K_MOSFET * (pwm_ts[i] - TEMP_AMBIENT)
                    else:  # STOP: 둘 다 식음
                        pwm_ts[i] -= COOL_K_MOSFET * (pwm_ts[i] - TEMP_AMBIENT)
                        dac_ts[i] -= COOL_K_MOSFET * (dac_ts[i] - TEMP_AMBIENT)
                    # 자연 방열
                    pwm_ts[i] -= COOL_K_MOSFET * (pwm_ts[i] - TEMP_AMBIENT)
                    dac_ts[i] -= COOL_K_MOSFET * (dac_ts[i] - TEMP_AMBIENT)
                    pwm_ts[i] = max(TEMP_AMBIENT, pwm_ts[i])
                    dac_ts[i] = max(TEMP_AMBIENT, dac_ts[i])

                # PCB 열 결합
                pwm_ts = apply_pcb_heat_coupling(pwm_ts)
                dac_ts = apply_pcb_heat_coupling(dac_ts)

                # 전압 밸런싱 (전류 흘린 만큼 수렴)
                cells_v = balance_toward_min(cells_v, cell_currents,
                                             smoothness=1.0 if mode == "PWM" else 0.5)
                cells_v = [max(3.0, min(4.25, v)) for v in cells_v]

                sim_state["v"] = cells_v
                sim_state["pwm_mosfet_t"] = pwm_ts
                sim_state["dac_mosfet_t"] = dac_ts

            # ===== 6. WebSocket payload 송신 =====
            payload = {
                "system_mode": "REAL_HARDWARE" if is_real_mode else "DUMMY_SIMULATION",
                "ai_loaded": AI_LOADED,
                "ai_mode": mode,
                "p_dac": round(p_dac, 3),
                "pack_delta_v": round(pack_dv, 4),
                "pack_min_v": round(min_v, 3),
                "real": {
                    "v":          add_display_noise(sim_state["v"], is_pwm=(mode == "PWM")),
                    "i":          [round(c, 3) for c in i_for_display],
                    "pwm_mosfet_t": [round(t, 2) for t in sim_state["pwm_mosfet_t"]],
                    "dac_mosfet_t": [round(t, 2) for t in sim_state["dac_mosfet_t"]],
                    "dac_vals":   list(dac_vals),
                    "pwm_duty":   list(pwm_duty),
                },
            }
            await websocket.send_text(json.dumps(payload))

            # 실측 로깅
            log_tick(sim_state, mode, p_dac, dac_vals, pwm_duty)

            # 전압 조정 완료(STOP 진입) 시 자동 재학습 (실측 모드 + 로컬 + 1회만)
            # prev_mode가 STOP이 아니었다가 STOP이 된 순간 = 막 평탄화 완료
            if (mode == "STOP" and prev_mode != "STOP"
                    and is_real_mode and AUTO_RETRAIN_ON_STOP):
                print("[자동재학습] 전압 조정 완료 감지 → 재학습 트리거")
                # 블로킹 방지: 별도 스레드에서 실행
                asyncio.get_event_loop().run_in_executor(None, trigger_retrain)

            prev_mode = mode
            await asyncio.sleep(1.0)

    except Exception as e:
        print(f"[ERROR] 웹소켓: {e}")


if __name__ == "__main__":
    import uvicorn
    print(f"[ENV] 배포 환경: {DEPLOY_ENV} (자동 재학습: {AUTO_RETRAIN_ON_STOP})")
    # 로컬: 8000번 고정 / Render: 환경변수 PORT 사용
    port = int(os.environ.get('PORT', 8000))
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    finally:
        close_logger()
