"""
- CSV 2개 생성:
  1. bms_pack_data.csv  : 분류기용 (팩 단위)
     입력: [max_mosfet_t, max_battery_t, pack_delta_v]
     출력: PWM or DAC

  2. bms_cell_data.csv  : 회귀기용 (셀 단위)
     입력: [cell_mosfet_t, cell_battery_t, cell_v, delta_from_min]
     출력: dac_val (0~4095), pwm_duty (0~255)

확률적 라벨링: 시그모이드 기반 부드러운 모드 전환
Thermal Throttling: 온도 높을 때 출력값 자동 감쇄
"""
import csv
import random
import math

NUM_PACK_SAMPLES = 6000
NUM_CELL_SAMPLES = 8000
PACK_FILE = "bms_pack_data.csv"
CELL_FILE = "bms_cell_data.csv"

# 임계값
MOSFET_T_HIGH = 50.0
MOSFET_T_STOP = 70.0
BATTERY_T_HIGH = 45.0
BATTERY_T_STOP = 55.0
DV_LOW = 0.02
DV_HIGH = 0.08


def sigmoid(x, center, sharpness=50):
    """경계 근처에서 부드럽게 전환되는 확률 함수"""
    return 1.0 / (1.0 + math.exp(-sharpness * (x - center)))


# ============================================================
# Part 1: 팩 단위 데이터 (분류기용)
# ============================================================
def get_pack_dac_probability(max_m_t, max_b_t, pack_dv):
    """팩 상태 -> DAC 모드일 확률"""
    # MOSFET 온도 기반 (50도 근처에서 전환)
    p_mosfet = sigmoid(max_m_t, MOSFET_T_HIGH, sharpness=0.6)
    # 배터리 온도 기반 (45도 근처)
    p_battery = sigmoid(max_b_t, BATTERY_T_HIGH, sharpness=0.6)
    # 편차 기반 (작을수록 DAC)
    center_dv = (DV_LOW + DV_HIGH) / 2
    p_dv = 1.0 - sigmoid(pack_dv, center_dv, sharpness=120)

    # 어느 하나라도 DAC 강하게 원하면 DAC (max 결합)
    p_dac = max(p_mosfet, p_battery, p_dv)
    p_dac += random.uniform(-0.05, 0.05)
    return max(0.02, min(0.98, p_dac))


def generate_pack_sample():
    max_m_t = round(random.uniform(25, 65), 1)
    max_b_t = round(random.uniform(25, 50), 1)  # 배터리는 더 좁은 범위
    pack_dv = round(random.uniform(0, 0.15), 4)

    p_dac = get_pack_dac_probability(max_m_t, max_b_t, pack_dv)
    label = "DAC" if random.random() < p_dac else "PWM"
    return max_m_t, max_b_t, pack_dv, label, round(p_dac, 3)


def make_pack_csv():
    dac_list, pwm_list = [], []
    target = NUM_PACK_SAMPLES // 2
    attempts = 0
    while (len(dac_list) < target or len(pwm_list) < target) and attempts < NUM_PACK_SAMPLES * 10:
        s = generate_pack_sample()
        if s[3] == "DAC" and len(dac_list) < target:
            dac_list.append(s)
        elif s[3] == "PWM" and len(pwm_list) < target:
            pwm_list.append(s)
        attempts += 1

    all_data = dac_list + pwm_list
    random.shuffle(all_data)
    with open(PACK_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Max_Mosfet_T", "Max_Battery_T", "Pack_Delta_V", "Label", "P_DAC"])
        w.writerows(all_data)
    print(f"[PACK] {len(all_data)}개 (PWM:{len(pwm_list)} / DAC:{len(dac_list)}) -> {PACK_FILE}")


# ============================================================
# Part 2: 셀 단위 데이터 (회귀기용)
# ============================================================
def get_dac_value(cell_m_t, cell_b_t, cell_v, delta):
    """
    DAC 출력값 (0~4095)
    - delta 비례
    - MOSFET 온도 derating (saturation 영역 발열)
    - 배터리 온도 derating
    - 셀 위험 영역 보호
    """
    # 기본: delta 비례 (delta=0.15에서 4095 도달)
    val = (delta / 0.15) * 4095

    # MOSFET 온도 derating (DAC 모드는 MOSFET 발열 큼)
    if cell_m_t >= MOSFET_T_HIGH:
        penalty = 1.0 - (cell_m_t - MOSFET_T_HIGH) * 0.04
        val *= max(0.1, penalty)

    # 배터리 온도 derating
    if cell_b_t >= BATTERY_T_HIGH:
        penalty = 1.0 - (cell_b_t - BATTERY_T_HIGH) * 0.06
        val *= max(0.1, penalty)

    # 셀 전압 위험 영역 (3.3V 미만)
    if cell_v < 3.3:
        val *= 0.4

    # 노이즈 추가
    val += random.uniform(-50, 50)
    return int(min(4095, max(0, val)))


def get_pwm_duty(cell_m_t, cell_b_t, cell_v, delta):
    """
    PWM 듀티비 (0~255)
    - delta 비례 (PWM은 큰 편차에서 동작)
    - 온도 derating은 DAC보다 약함 (PWM은 효율 좋아서 발열 덜함)
    - 셀 위험 영역 보호
    """
    # 기본: delta 비례 (delta=0.15에서 255 도달)
    val = (delta / 0.15) * 255

    # MOSFET 온도 derating (PWM은 약하게)
    if cell_m_t >= MOSFET_T_HIGH:
        penalty = 1.0 - (cell_m_t - MOSFET_T_HIGH) * 0.025
        val *= max(0.2, penalty)

    # 배터리 온도 derating
    if cell_b_t >= BATTERY_T_HIGH:
        penalty = 1.0 - (cell_b_t - BATTERY_T_HIGH) * 0.04
        val *= max(0.2, penalty)

    # 셀 전압 위험 영역
    if cell_v < 3.3:
        val *= 0.5

    # 노이즈
    val += random.uniform(-5, 5)
    return int(min(255, max(0, val)))


def generate_cell_sample():
    cell_m_t = round(random.uniform(25, 70), 1)
    cell_b_t = round(random.uniform(25, 55), 1)
    cell_v = round(random.uniform(3.0, 4.2), 3)
    # delta는 0~0.30 (큰 편차도 학습)
    delta = round(random.uniform(0, 0.30), 4)

    dac_val = get_dac_value(cell_m_t, cell_b_t, cell_v, delta)
    pwm_duty = get_pwm_duty(cell_m_t, cell_b_t, cell_v, delta)

    return cell_m_t, cell_b_t, cell_v, delta, dac_val, pwm_duty


def make_cell_csv():
    samples = [generate_cell_sample() for _ in range(NUM_CELL_SAMPLES)]
    with open(CELL_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Mosfet_T", "Battery_T", "Cell_V", "Delta_From_Min", "DAC_Val", "PWM_Duty"])
        w.writerows(samples)
    print(f"[CELL] {len(samples)}개 -> {CELL_FILE}")


# ============================================================
# 실행
# ============================================================
def main():
    print("[1/2] 팩 단위 분류기 데이터 생성...")
    make_pack_csv()
    print("[2/2] 셀 단위 회귀기 데이터 생성...")
    make_cell_csv()

    # 미리보기
    print("\n[PACK 샘플 5개]")
    with open(PACK_FILE) as f:
        for i, line in enumerate(f):
            if i > 5: break
            print("  " + line.strip())
    print("\n[CELL 샘플 5개]")
    with open(CELL_FILE) as f:
        for i, line in enumerate(f):
            if i > 5: break
            print("  " + line.strip())


if __name__ == "__main__":
    main()
