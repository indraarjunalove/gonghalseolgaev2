"""
A²-BMS 메인 서버 - 방안 B (소전류 + 가상 대전류 통합)
==================================================
[방안 B 변경사항]
1. AI 모델 6개 로드 (소전류 3 + 대전류 3)
2. 가상 PWM-only / DAC-only 시뮬 제거
3. 가상 대전류 시뮬 추가 (×20 스케일링, 데이터시트 물리식)
4. 환경별 적응형 AI: 소전류 AI는 실측에, 대전류 AI는 가상 환경에 적용
5. 웹 대시보드: 12 그래프 → 8 그래프 (실측 4 + 가상 대전류 4)

[데이터 흐름]
실측 (0.5A) → 소전류 AI → 모드/출력 결정 → ESP32
실측 (0.5A) → 물리식 ×20 → 가상 10A 환경 → 대전류 AI → 가상 모드/출력
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

# UART 통신용 (실제 모드에서만 사용, 시뮬레이션 시 무시)
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
# 실측 데이터 자동 로깅(csv)
# ============================================================
# is_real_mode=True 일 때만.. 실제측정값 저장
# 매 tick의 입력+AI출력+결과(다음 tick 측정값)을 CSV로 저장
# 추후 train_system.py로 재학습 시켜야함
LOGGING_ENABLED = True              # 로깅 on/off
LOG_DIR = 'logs'                    # 로그 폴더
LOG_FILE_PREFIX = 'real_data'       # 파일명 prefix
LOG_ROTATE_HOURS = 24               # N시간마다 새 파일

# ============================================================
# 설정 (실측 데이터 얻고 수정 필요)
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

# PCB 열 결합 계수
# 이웃 셀 MOSFET이 뜨거우면 PCB 통해 이쪽도 따뜻해짐
# 0 = 완전 독립 (이전 가정), 1 = 완전 결합 (현실)
# 0.05 = 약한 결합 (PCB 가정)
PCB_HEAT_COUPLING = 0.015   # 약한 결합 (현실적 PCB)

# 자가방전: 단조 감소 모델 가정
# 셀마다 다른 속도 (개체 차이)
# 시간당 1~3mV 떨어짐 (실제 18650 자가방전률)

# 셀별 자가방전 속도 (V/sec), 한 번만 초기화
SELF_DISCHARGE_RATES = [
    random.uniform(0.0000003, 0.0000008)  # 약 1~3 mV/hour
    for _ in range(4)
]
# 측정 노이즈 (자가방전과 분리)
MEASUREMENT_NOISE = 0.0005  # ±0.5mV 측정 노이즈

app = FastAPI()

# ============================================================
# AI 모델 6개 로드 (방안 B: 환경별 적응형 AI)
# ============================================================
# 소전류 환경 (학부 실험 0.1~0.5A)
try:
    clf_low = joblib.load('bms_mode_ai_low.pkl')
    reg_dac_low = joblib.load('bms_dac_ai_low.pkl')
    reg_pwm_low = joblib.load('bms_pwm_ai_low.pkl')
    AI_LOW_LOADED = True
    print("[AI 소전류] 모델 3개 로드 완료")
except Exception as e:
    clf_low = reg_dac_low = reg_pwm_low = None
    AI_LOW_LOADED = False
    print(f"[WARN] 소전류 모델 없음: {e}")

# 대전류 환경 (가상 HILS 10A~)
try:
    clf_high = joblib.load('bms_mode_ai_high.pkl')
    reg_dac_high = joblib.load('bms_dac_ai_high.pkl')
    reg_pwm_high = joblib.load('bms_pwm_ai_high.pkl')
    AI_HIGH_LOADED = True
    print("[AI 대전류] 모델 3개 로드 완료")
except Exception as e:
    clf_high = reg_dac_high = reg_pwm_high = None
    AI_HIGH_LOADED = False
    print(f"[WARN] 대전류 모델 없음: {e}")

# 전체 AI 로드 상태 (기존 코드 호환용)
AI_LOADED = AI_LOW_LOADED

# ============================================================
# [방안 B] 대전류 환경 물리 상수 (데이터시트 기반)
# Plant Model용 - AI 학습 대상 아님 (객관적 ground truth)
# ============================================================
class PhysicsConstants:
    """대전류 환경 BMS 표준 상수"""
    # ===== Samsung INR18650-30Q (Samsung SDI 데이터시트) =====
    R_CELL_INTERNAL = 0.022       # 22 mΩ
    C_CELL_THERMAL = 42.0         # 셀당 42 J/K (논문 기준)
    R_THETA_CELL = 8.0            # 자연방열 °C/W
    
    # ===== Infineon IRFB4110 (Infineon 데이터시트) =====
    R_DS_ON = 0.0037              # 3.7 mΩ
    V_DS_SAT = 2.5                # saturation V_DS drop
    R_THETA_MOSFET = 15.0         # 방열판 가정 °C/W
    C_MOSFET_THERMAL = 5.3        # TO-220 열용량
    T_J_MAX = 175.0               # 절대 한계
    
    # 스케일 팩터 (실측 ×K = 가상 대전류)
    SCALE_FACTOR = 20             # 0.5A × 20 = 10A (30Q 약 3C, 급속충전급)


class HighCurrentSimulator:
    """가상 대전류 환경 시뮬레이터 (Plant Model)
    실측 소전류 → 데이터시트 물리식 → 가상 대전류 상태"""
    def __init__(self):
        self.C = PhysicsConstants()
        self.virtual_mosfet_t = [TEMP_AMBIENT] * 4
        self.virtual_cell_t = [TEMP_AMBIENT] * 4
    
    def reset(self):
        self.virtual_mosfet_t = [TEMP_AMBIENT] * 4
        self.virtual_cell_t = [TEMP_AMBIENT] * 4
    
    def scale_current(self, i_real_list):
        """실측 전류 → 가상 대전류 (×K)"""
        return [i * self.C.SCALE_FACTOR for i in i_real_list]
    
    def virtual_cell_voltage(self, v_open_list, i_scaled_list):
        """대전류 시 IR drop 적용 (옴의 법칙)"""
        return [v - i * self.C.R_CELL_INTERNAL 
                for v, i in zip(v_open_list, i_scaled_list)]
    
    def mosfet_power_loss(self, i_scaled, mode):
        """MOSFET 발열 (W) - 모드별"""
        if mode == "DAC":
            return self.C.V_DS_SAT * abs(i_scaled)
        elif mode == "PWM":
            return i_scaled ** 2 * self.C.R_DS_ON
        return 0.0
    
    def cell_power_loss(self, i_scaled):
        """셀 내부 발열 P = I²R"""
        return i_scaled ** 2 * self.C.R_CELL_INTERNAL
    
    def update_virtual_temperatures(self, i_scaled_list, mode, dt=1.0, t_amb=None):
        """뉴턴 냉각 법칙으로 가상 환경 온도 업데이트"""
        if t_amb is None:
            t_amb = TEMP_AMBIENT
        
        p_totals = []
        for i in range(4):
            i_s = abs(i_scaled_list[i])
            
            # MOSFET
            p_mosfet = self.mosfet_power_loss(i_s, mode)
            dt_heat = (p_mosfet * dt) / self.C.C_MOSFET_THERMAL
            dt_cool = (self.virtual_mosfet_t[i] - t_amb) * dt / (self.C.R_THETA_MOSFET * self.C.C_MOSFET_THERMAL)
            self.virtual_mosfet_t[i] += dt_heat - dt_cool
            self.virtual_mosfet_t[i] = max(t_amb, self.virtual_mosfet_t[i])
            
            # 셀
            p_cell = self.cell_power_loss(i_s)
            dt_heat_c = (p_cell * dt) / self.C.C_CELL_THERMAL
            dt_cool_c = (self.virtual_cell_t[i] - t_amb) * dt / (self.C.R_THETA_CELL * self.C.C_CELL_THERMAL)
            self.virtual_cell_t[i] += dt_heat_c - dt_cool_c
            self.virtual_cell_t[i] = max(t_amb, self.virtual_cell_t[i])
            
            p_totals.append(p_mosfet + p_cell)
        
        return list(self.virtual_mosfet_t), list(self.virtual_cell_t), p_totals


# 글로벌 인스턴스
hc_sim = HighCurrentSimulator()

# ============================================================
# UART 통신 (ESP32 ↔ 라즈베리파이) 확인!! 확인!!! 굴러가는지봐야됨!!!
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
# 시뮬레이션 초기값 가상 설정
# ============================================================
is_real_mode = False

sim_state = {
    # ===== 실측 (소전류, 안전 범위) =====
    "v": [4.20, 4.10, 4.05, 3.90],
    "mosfet_t": [30.0, 30.0, 30.0, 30.0],
    "battery_t": [30.0, 30.0, 30.0, 30.0],
    "i": [0.0, 0.0, 0.0, 0.0],

    # AI 출력 (실제 회로 인가 - 소전류 AI 기준)
    "dac_vals": [0, 0, 0, 0],
    "pwm_duty": [0, 0, 0, 0],

    # ===== [방안 B] 가상 대전류 환경 (Plant Model 출력) =====
    "v_high": [4.20, 4.10, 4.05, 3.90],          # IR drop 적용 후 가상 전압
    "i_high": [0.0, 0.0, 0.0, 0.0],              # ×20 스케일 가상 전류
    "mosfet_t_high": [30.0, 30.0, 30.0, 30.0],   # 가상 MOSFET 온도
    "battery_t_high": [30.0, 30.0, 30.0, 30.0],  # 가상 배터리 온도

    # 가상 환경 AI 출력 (별개)
    "ai_mode_high": "DAC",
    "p_dac_high": 0.5,
    "dac_vals_high": [0, 0, 0, 0],
    "pwm_duty_high": [0, 0, 0, 0],
}

# ============================================================
# 실측 데이터 로깅 시스템
# ============================================================
# 매 tick 데이터를 CSV로 저장 → 실측 데이터 축적 → 재학습에 써먹을 거 (이건수동)
# 헤더: 시간 + 입력(V/MT/BT) + AI출력(mode/DAC/PWM) + 결과지표(다음 tick ΔV 변화량)
LOG_HEADER = [
    'timestamp', 'tick',
    # 입력 (현재 상태)
    'v1', 'v2', 'v3', 'v4',
    'mt1', 'mt2', 'mt3', 'mt4',
    'bt1', 'bt2', 'bt3', 'bt4',
    'pack_delta_v',
    # AI 출력 (현재 결정)
    'ai_mode', 'p_dac',
    'dac1', 'dac2', 'dac3', 'dac4',
    'pwm1', 'pwm2', 'pwm3', 'pwm4',
    # 결과 (직전 tick 대비 ΔV 변화 — 이게 진짜 효과)
    'delta_v_change',     # 양수 = ΔV 증가 (나빠짐), 음수 = ΔV 감소 (좋아짐)
    'mt_max_change',      # MOSFET 최고 온도 변화
    'bt_max_change',      # 배터리 최고 온도 변화
    'data_source',        # 'real' or 'sim'
]

logger_state = {
    'file': None,
    'writer': None,
    'tick': 0,
    'start_time': None,
    'prev_pdv': None,
    'prev_mt_max': None,
    'prev_bt_max': None,
}

def init_logger():
    """로그 폴더/파일 생성"""
    if not LOGGING_ENABLED:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{LOG_DIR}/{LOG_FILE_PREFIX}_{timestamp}.csv"
    logger_state['file'] = open(filename, 'w', newline='', encoding='utf-8')
    logger_state['writer'] = csv.writer(logger_state['file'])
    logger_state['writer'].writerow(LOG_HEADER)
    logger_state['file'].flush()
    logger_state['start_time'] = time.time()
    print(f"[LOG] 로깅 시작 → {filename}")

def log_tick(state, mode, p_dac, dac_vals, pwm_duty, is_real):
    """매 tick 데이터 1줄 기록"""
    if not LOGGING_ENABLED or logger_state['writer'] is None:
        return
    
    pdv = max(state['v']) - min(state['v'])
    mt_max = max(state['mosfet_t'])
    bt_max = max(state['battery_t'])
    
    # 이전 tick과 비교 (결과 지표)
    dv_change = (pdv - logger_state['prev_pdv']) if logger_state['prev_pdv'] is not None else 0
    mt_change = (mt_max - logger_state['prev_mt_max']) if logger_state['prev_mt_max'] is not None else 0
    bt_change = (bt_max - logger_state['prev_bt_max']) if logger_state['prev_bt_max'] is not None else 0
    
    row = [
        datetime.now().isoformat(),
        logger_state['tick'],
        # 입력
        round(state['v'][0], 4), round(state['v'][1], 4), round(state['v'][2], 4), round(state['v'][3], 4),
        round(state['mosfet_t'][0], 2), round(state['mosfet_t'][1], 2), round(state['mosfet_t'][2], 2), round(state['mosfet_t'][3], 2),
        round(state['battery_t'][0], 2), round(state['battery_t'][1], 2), round(state['battery_t'][2], 2), round(state['battery_t'][3], 2),
        round(pdv, 4),
        # AI 출력
        mode, round(p_dac, 3),
        dac_vals[0], dac_vals[1], dac_vals[2], dac_vals[3],
        pwm_duty[0], pwm_duty[1], pwm_duty[2], pwm_duty[3],
        # 결과 변화
        round(dv_change, 4),
        round(mt_change, 2),
        round(bt_change, 2),
        'real' if is_real else 'sim',
    ]
    logger_state['writer'].writerow(row)
    
    # 매 10 tick마다 flush (저장 보장)
    if logger_state['tick'] % 10 == 0:
        logger_state['file'].flush()
    
    logger_state['tick'] += 1
    logger_state['prev_pdv'] = pdv
    logger_state['prev_mt_max'] = mt_max
    logger_state['prev_bt_max'] = bt_max
    
    # 24시간마다 새 파일 (rotation)
    if logger_state['start_time'] and (time.time() - logger_state['start_time']) > LOG_ROTATE_HOURS * 3600:
        logger_state['file'].close()
        init_logger()

def close_logger():
    """프로그램 종료 시 호출"""
    if logger_state['file']:
        logger_state['file'].close()
        print(f"[LOG] 로깅 종료, 총 {logger_state['tick']}개 tick 저장")

@app.get("/")
async def get_index():
    if not os.path.exists("index.html"):
        return {"error": "index.html not found"}
    return FileResponse("index.html")

# ============================================================
# AI 판단
# ============================================================
def ai_decide_mode(max_mosfet_t, max_battery_t, pack_delta_v, env='low'):
    """모드 분류기 추론
    env: 'low' (소전류) 또는 'high' (대전류)
    """
    # 환경별 STOP 임계 (대전류는 더 높은 온도까지 허용)
    if env == 'high':
        mt_stop = 100.0
        bt_stop = 60.0
    else:
        mt_stop = MOSFET_T_STOP
        bt_stop = BATTERY_T_STOP
    
    if max_mosfet_t >= mt_stop or max_battery_t >= bt_stop:
        return "STOP", 1.0
    # 전압차 체크 (0.01V 이하로 평탄화 완료 시 정지)
    if pack_delta_v <= 0.01:
        return "STOP", 1.0

    # 환경별 모델 선택
    if env == 'high':
        ai_ready = AI_HIGH_LOADED
        model = clf_high
    else:
        ai_ready = AI_LOW_LOADED
        model = clf_low

    if ai_ready:
        feat = pd.DataFrame([[max_mosfet_t, max_battery_t, pack_delta_v]], columns=PACK_FEATURES)
        proba = model.predict_proba(feat)[0]
        cls = model.classes_
        p_dac = float(proba[list(cls).index('DAC')]) if 'DAC' in cls else 0.5
        mode = model.predict(feat)[0]
        return mode, p_dac
    
    # AI 비로드시 비상용 (환경별 다른 임계)
    else:
        mt_high = 80.0 if env == 'high' else 50.0
        bt_high = 50.0 if env == 'high' else 45.0
        if max_mosfet_t >= mt_high or max_battery_t >= bt_high or pack_delta_v <= 0.02:
            return "DAC", 0.9
        elif pack_delta_v >= 0.05:
            return "PWM", 0.1
        else:
            return "PWM", 0.3


def ai_decide_cell_outputs(mode, mosfet_ts, battery_ts, voltages, env='low'):
    """셀별 출력값 회귀기 추론
    env: 'low' (소전류) 또는 'high' (대전류)
    """
    if mode == "STOP":
        return [0, 0, 0, 0], [0, 0, 0, 0]

    min_v = min(voltages)
    deltas = [v - min_v for v in voltages]

    batch = pd.DataFrame([
        [mosfet_ts[i], battery_ts[i], voltages[i], deltas[i]] for i in range(4)
    ], columns=CELL_FEATURES)

    # 환경별 모델 선택
    if env == 'high':
        ai_ready = AI_HIGH_LOADED
        dac_model = reg_dac_high
        pwm_model = reg_pwm_high
    else:
        ai_ready = AI_LOW_LOADED
        dac_model = reg_dac_low
        pwm_model = reg_pwm_low

    if ai_ready:
        if mode == "DAC":
            dac_vals = dac_model.predict(batch)
            dac_vals = [int(max(0, min(4095, v))) for v in dac_vals]
            pwm_duty = [0, 0, 0, 0]
        else:
            pwm_duty = pwm_model.predict(batch)
            pwm_duty = [int(max(0, min(255, v))) for v in pwm_duty]
            dac_vals = [0, 0, 0, 0]

    # AI 비로드 시 비상용
    else:
        if mode == "DAC":
            dac_vals = [int(min(4095, deltas[i] / 0.15 * 4095)) for i in range(4)]
            pwm_duty = [0, 0, 0, 0]
        else:
            pwm_duty = [int(min(255, deltas[i] / 0.15 * 255)) for i in range(4)]
            dac_vals = [0, 0, 0, 0]

    return dac_vals, pwm_duty

# ============================================================
# 셀 전압 변화 (물리법칙에 의해 가정 - 실측 데이터 뽑고 다시 함 봐야됨)
# [자가방전 단조 감소 + 측정 노이즈 분리
# ============================================================
def discharge_cell(v, current, smoothness, cell_idx=0):
    """단일 셀 전압 변화 (상태 오염 방지)"""
    # 자가방전: 항상 일정한 단조 감소 (셀마다 고유 속도)
    self_discharge = -SELF_DISCHARGE_RATES[cell_idx]
    if current <= 0.001:
        true_v = v + self_discharge
        return min(v, true_v)
    
    decay = current * smoothness * 0.05
    true_v = v - decay + self_discharge
    return min(v, true_v) # 절대로 이전 전압(v)보다 커질 수 없음

def discharge_cells_balanced(cells, currents, smoothness):
    return [discharge_cell(cells[i], currents[i], smoothness, cell_idx=i) for i in range(len(cells))]

def balance_toward_min(cells, currents, smoothness):
    """가상 우주용 평탄화 (상태 오염 방지)"""
    min_v = min(cells)
    new_cells = []
    
    for i, v in enumerate(cells):
        # [5번] 단조 감소 자가방전
        self_discharge = -SELF_DISCHARGE_RATES[i]
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

def add_measurement_noise(cells_v):
    """표시용 측정 노이즈 (ADC 노이즈 + 양자화)
    실제 셀 전압엔 영향 없고 표시값만 흔들림"""
    return [v + random.uniform(-MEASUREMENT_NOISE, MEASUREMENT_NOISE) for v in cells_v]

# ============================================================
# PCB 열 결합 — 이웃 셀 MOSFET 발열이 PCB 통해 전파 - 실측 후 수정~~ 해야할듯?
# ============================================================
def apply_pcb_heat_coupling(temps):
    """4개 MOSFET이 같은 PCB에 있을 때 열 평형 보정
    각 셀 온도 = (1-k) × 자기 자신 + k × 이웃 평균"""
    if PCB_HEAT_COUPLING <= 0:
        return list(temps)
    
    coupled = []
    n = len(temps)
    for i in range(n):
        # 이웃 셀 (자기 자신 제외) 평균
        neighbors = [temps[j] for j in range(n) if j != i]
        neighbor_avg = sum(neighbors) / len(neighbors)
        # 가중 평균: 본인 (1-k) + 이웃 평균 k
        new_t = (1 - PCB_HEAT_COUPLING) * temps[i] + PCB_HEAT_COUPLING * neighbor_avg
        coupled.append(new_t)
    return coupled

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
    
    # 로거 초기화 (서버 시작 시 1회)
    if logger_state['file'] is None:
        init_logger()

    # 통신코드이므로 일단 통신 해보고 다시 검토
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

            # ============================================================
            # [소전류 AI] 실측 소전류 환경에 대한 판단
            # → 실제 ESP32 회로에 인가될 값
            # ============================================================
            mode, p_dac = ai_decide_mode(max_mosfet_t, max_battery_t, pack_dv, env='low')
            dac_vals, pwm_duty = ai_decide_cell_outputs(mode, cells_mt, cells_bt, cells_v, env='low')

            sim_state["dac_vals"] = dac_vals
            sim_state["pwm_duty"] = pwm_duty

            # ============================================================
            # [방안 B - 가상 대전류 AI] 데이터시트 물리식 ×20 스케일 환경
            # → 실제 회로엔 인가 안 함 (가상 시뮬만)
            # ============================================================
            # 1) 평균 전류로 가상 스케일 (PWM 펄스 튐 방지)
            #    sim_state["i"]는 PWM 모드일 때 펄스(0↔0.5A)라 그래프가 튐
            #    → AI 명령값(dac_vals/pwm_duty)에서 평균 전류 역산해서 사용
            if mode == "PWM":
                # PWM 펄스 시뮬: 듀티비 확률로 ON/OFF
                # 발열 계산용 평균 (i_avg_for_heat)과 그래프 표시용 펄스값(i_for_display) 분리
                i_avg_for_heat = [(pwm_duty[i] / 255.0) * 0.5 for i in range(4)]
                # 셀별 펄스값 생성 (4셀 각각)
                i_for_display = []
                for i in range(4):
                    duty_ratio = pwm_duty[i] / 255.0
                    # 듀티비를 확률로 사용해 매 tick ON/OFF 결정 (PWM 펄스 시뮬레이션)
                    # → 1초 단위 샘플링이라 실제 5kHz 펄스를 직접 못 그리므로,
                    #   듀티비 확률로 ON(0.5A)/OFF(0A)를 뽑아 펄스 특성을 시각화
                    if random.random() < duty_ratio:
                        pulse_val = 0.5 + random.uniform(-0.03, 0.03)  # ON 상태 + 스위칭 노이즈
                    else:
                        pulse_val = 0.0 + random.uniform(0, 0.02)      # OFF 상태 + 미세 노이즈
                    i_for_display.append(pulse_val)  # ★ 루프 안에서 4셀 모두 append

                # 발열 계산은 평균 전류 기준 (실제 PWM의 RMS 발열에 근사)
                i_virtual_for_heat = hc_sim.scale_current(i_avg_for_heat)
                # 그래프 표시는 펄스 전류 기준 (PWM 특유의 출렁임 표현)
                i_virtual = hc_sim.scale_current(i_for_display)
            elif mode == "DAC":
                i_avg = [(dac_vals[i] / 4095.0) * 0.4 for i in range(4)]
                # DAC는 부드러운 노이즈만 (작은 ±)
                i_avg = [v + random.uniform(-0.005, 0.005) for v in i_avg]
                i_virtual = hc_sim.scale_current(i_avg)
                i_virtual_for_heat = i_virtual
            else:
                i_avg = [0.0, 0.0, 0.0, 0.0]
                i_virtual = hc_sim.scale_current(i_avg)
                i_virtual_for_heat = i_virtual
            
            # 가상 전압 노이즈 판단용: 직전 tick의 대전류 모드 참조
            mode_high_prev_for_noise = sim_state.get("ai_mode_high", "DAC")

            # 2) 가상 셀 전압 (IR drop 적용 - 옴의 법칙: V = V_OCV - I×R_cell)
            v_virtual = hc_sim.virtual_cell_voltage(cells_v, i_virtual)
            # PWM 모드면 대전류 스위칭으로 인한 전압 리플 추가 (현실적 노이즈)
            # DAC 모드는 선형 제어라 리플 거의 없음 → 노이즈 미세하게만
            if mode_high_prev_for_noise == "PWM":
                v_virtual = [v + random.uniform(-0.04, 0.04) for v in v_virtual]
            else:
                v_virtual = [v + random.uniform(-0.005, 0.005) for v in v_virtual]
            
            # 3) 이전 모드로 가상 환경 온도 갱신 (물리식)
            prev_mode_high = sim_state.get("ai_mode_high", "DAC")
            # 발열은 평균값으로 (실제 PWM의 평균 발열 = RMS 발열)
            v_mt_high, v_bt_high, _ = hc_sim.update_virtual_temperatures(
                i_virtual_for_heat, prev_mode_high, dt=1.0
            )
            
            # 4) 대전류 AI 판단 (가상 상태 입력)
            pack_dv_high = max(v_virtual) - min(v_virtual)
            mode_high, p_dac_high = ai_decide_mode(
                max(v_mt_high), max(v_bt_high), pack_dv_high, env='high'
            )
            dac_vals_high, pwm_duty_high = ai_decide_cell_outputs(
                mode_high, v_mt_high, v_bt_high, v_virtual, env='high'
            )
            
            # 5) sim_state에 가상 결과 저장
            sim_state["v_high"] = v_virtual
            sim_state["i_high"] = i_virtual
            sim_state["mosfet_t_high"] = v_mt_high
            sim_state["battery_t_high"] = v_bt_high
            sim_state["ai_mode_high"] = mode_high
            sim_state["p_dac_high"] = p_dac_high
            sim_state["dac_vals_high"] = dac_vals_high
            sim_state["pwm_duty_high"] = pwm_duty_high

                       
            # ----------------------------------------------------------
            # AI 출력을 ESP32로 전송 (UART) - 소전류 AI 결과만
            # ----------------------------------------------------------
            if is_real_mode:
                uart_send_command(mode, dac_vals, pwm_duty)
            #ㄴ 송신 실패해도 시뮬은 계속 (안전한 fallback)

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
            
            # PCB 열 결합 — MOSFET들이 PCB 통해 서로 영향
            # 한 셀이 풀파워면 옆 셀도 살짝 따뜻해지는 거 반영
            cells_mt = apply_pcb_heat_coupling(cells_mt)
            # 배터리는 물리적으로 떨어져 있어 열 결합 거의 없음 (그대로)

            # 실제 모드일 때는 위에서 받은 데이터로 sim_state 이미 갱신됨 -> 시뮬 계산 스킵
            # 시뮬 모드일 때만 발열/평탄화 시뮬 적용
            if not is_real_mode:
                if mode == "PWM":
                    cells_v_new = balance_toward_min(cells_v, cell_currents, smoothness=1.0)
                elif mode == "DAC":
                    cells_v_new = balance_toward_min(cells_v, cell_currents, smoothness=0.5)
                else: 
                    # [5번] STOP 모드 자가방전 — 단조 감소 (랜덤 아님)
                    cells_v_new = [v - SELF_DISCHARGE_RATES[i] for i, v in enumerate(cells_v)]

                sim_state["v"] = [max(3.0, min(4.25, v)) for v in cells_v_new]
                sim_state["mosfet_t"] = cells_mt
                sim_state["battery_t"] = cells_bt
            # 실제 모드: sim_state는 ESP32 데이터 그대로 사용 (덮어쓰기 안 함)

            # ============================================================
            # 6. [방안 B] 가상 대전류 환경은 이미 위에서 hc_sim으로 처리 완료
            #    (sim_state["v_high"], ["mosfet_t_high"] 등에 저장됨)
            # ============================================================

            # ============================================================
            # 7. WebSocket payload (출력 시 노이즈 연출용)
            # ============================================================
            def add_display_noise(v_list, is_pwm=False):
                # 출력용 가짜 노이즈 (상태 오염 X)
                n_range = 0.015 if is_pwm else 0.001
                return [round(v + random.uniform(-n_range, n_range), 3) for v in v_list]

            payload = {
                "system_mode": "REAL_HARDWARE" if is_real_mode else "DUMMY_SIMULATION",
                "ai_loaded": AI_LOW_LOADED,
                "ai_high_loaded": AI_HIGH_LOADED,
                "ai_mode": mode,
                "p_dac": round(p_dac, 3),
                "pack_delta_v": round(pack_dv, 4),
                "pack_min_v": round(min_v, 3),

                # ===== 실측 (소전류, 안전 범위) =====
                "real": {
                    "v":         add_display_noise(sim_state["v"], is_pwm=(mode=="PWM")),
                    "i":         [round(c, 3) for c in cell_currents],
                    "mosfet_t":  [round(t, 2) for t in cells_mt],
                    "battery_t": [round(t, 2) for t in cells_bt],
                    "dac_vals":  list(dac_vals),
                    "pwm_duty":  list(pwm_duty),
                },

                # ===== [방안 B] 가상 대전류 환경 (Plant Model + 대전류 AI) =====
                "high": {
                    "scale_factor": hc_sim.C.SCALE_FACTOR,
                    "ai_mode":   sim_state["ai_mode_high"],
                    "p_dac":     round(sim_state["p_dac_high"], 3),
                    "v":         [round(v, 3) for v in sim_state["v_high"]],
                    "i":         [round(c, 2) for c in sim_state["i_high"]],
                    "mosfet_t":  [round(t, 2) for t in sim_state["mosfet_t_high"]],
                    "battery_t": [round(t, 2) for t in sim_state["battery_t_high"]],
                    "dac_vals":  list(sim_state["dac_vals_high"]),
                    "pwm_duty":  list(sim_state["pwm_duty_high"]),
                    "i_real_avg": round(sum(sim_state["i"])/4, 3),
                    "i_virtual_avg": round(sum(sim_state["i_high"])/4, 2),
                },
            }

            await websocket.send_text(json.dumps(payload))
            
            # [1번 대응] 매 tick 데이터 로깅
            log_tick(sim_state, mode, p_dac, dac_vals, pwm_duty, is_real_mode)
            
            await asyncio.sleep(1.0)

    except Exception as e:
        print(f"[ERROR] 웹소켓: {e}")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        close_logger()
