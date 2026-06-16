"""
A²-BMS 학습 데이터 생성기 (소전류 + 듀얼 MOSFET 측정 기반)
============================================================
[설계 배경]
- 측정 대상: 셀 전압 + Q_PWM MOSFET 온도 + Q_LINEAR(DAC) MOSFET 온도
- 배터리 온도 제외: 소전류(~0.5A) 환경에서 셀 발열은 I²R_cell ≈ 0.006W로
  무시 가능 → 배터리가 타지 않는다는 신뢰 기반
- 두 MOSFET을 따로 측정하는 이유: 모드별 발열 특성이 정반대이기 때문
    · DAC 모드: Q_LINEAR가 선형영역 손실(V_DS×I)로 크게 발열
    · PWM 모드: Q_PWM이 스위칭(I²R_DS_on)이라 발열 미미
  → 두 온도를 비교하면 AI가 "어느 모드가 발열 유리한지" 직접 학습 가능

[인과 사슬]
  ΔV(전압편차) → 밸런싱 전류량 → 두 MOSFET 발열
    · DAC 모스펫 온도: ΔV에 강하게 비례 (선형손실 큼)
    · PWM 모스펫 온도: ΔV에 약하게 비례 (스위칭손실 작음)
  → 같은 ΔV라도 DAC 모스펫이 항상 더 뜨겁게 생성 (물리 반영)

[데이터셋]
  - bms_pack_data.csv : 모드 분류기용 (팩 단위)
  - bms_cell_data.csv : 출력 회귀기용 (셀 단위)

[비율] 80% 인과 데이터 + 20% 독립 random (과적합 방지)
"""
import csv, random, math

NUM_PACK_SAMPLES = 6000      # 분류기용 샘플 수
NUM_CELL_SAMPLES = 8000      # 회귀기용 샘플 수
INDEPENDENT_RATIO = 0.20     # 독립 random 비율 (나머지는 인과 기반)
AMBIENT_T = 25.0             # 상온 (발열 누적 기준점)

PACK_FILE = "bms_pack_data.csv"
CELL_FILE = "bms_cell_data.csv"


def sigmoid(x, center, sharpness=50):
    """경계 부근에서 부드럽게 0→1 전환되는 확률 함수 (확률적 라벨링용)"""
    return 1.0 / (1.0 + math.exp(-sharpness * (x - center)))


# ============================================================
# 환경 파라미터 (소전류 학부 환경)
# ============================================================
class Params:
    # --- 안전/판단 임계값 ---
    MOSFET_T_HIGH = 50.0     # 이 온도 넘으면 발열 위험 구간 (derating 시작)
    MOSFET_T_STOP = 65.0     # 강제 정지 임계 (Op-Amp 70°C 한계 보호)
    DV_LOW = 0.02            # 이 이하 편차는 거의 평탄 → DAC 정밀 선호
    DV_HIGH = 0.08           # 이 이상 편차는 큼 → PWM 고속 선호

    # --- 데이터 분포 범위 (독립 random 생성 시) ---
    PACK_DV_RANGE = (0, 0.15)
    CELL_V_RANGE = (3.0, 4.2)
    DELTA_RANGE = (0, 0.30)
    MOSFET_PWM_T_RANGE = (25, 55)    # PWM 모스펫: 발열 작아 범위 좁음
    MOSFET_DAC_T_RANGE = (25, 70)    # DAC 모스펫: 발열 커서 범위 넓음

    # --- 인과 발열 계수 (ΔV → 온도 상승) ---
    # DAC 모스펫: 선형손실(V_DS×I)이라 발열 계수 큼
    DAC_HEAT_COEF_PACK = (200, 350)
    DAC_HEAT_COEF_CELL = (150, 280)
    # PWM 모스펫: 스위칭손실(I²R_DS)이라 발열 계수 작음 (DAC의 약 1/4)
    PWM_HEAT_COEF_PACK = (40, 90)
    PWM_HEAT_COEF_CELL = (30, 70)

    # --- derating 계수 (온도 높을 때 출력 감쇄) ---
    DAC_DERATE = 0.04
    PWM_DERATE = 0.025
    LOW_V_PENALTY = (2, 6)   # 셀 전압 낮을 때(<3.3V) 내부저항↑ 추가 발열


# ============================================================
# 모드 라벨 확률 (팩 상태 → DAC일 확률)
# ============================================================
def get_pack_dac_probability(pwm_t, dac_t, pack_dv, P):
    """
    DAC 모드를 택할 확률.
    - DAC 모스펫이 이미 뜨거우면 → DAC 회피 (PWM 선호) → 확률 낮춤
    - 편차가 작으면 → 정밀한 DAC 선호 → 확률 높임
    - PWM 모스펫이 뜨거우면 → PWM 회피 (DAC 선호) → 확률 높임
    """
    # 편차 작을수록 DAC 선호 (정밀 제어)
    center_dv = (P.DV_LOW + P.DV_HIGH) / 2
    p_dv = 1.0 - sigmoid(pack_dv, center_dv, sharpness=120)
    # DAC 모스펫 뜨거우면 DAC 회피
    p_dac_hot = 1.0 - sigmoid(dac_t, P.MOSFET_T_HIGH, sharpness=0.5)
    # PWM 모스펫 뜨거우면 PWM 회피 → DAC 선호
    p_pwm_hot = sigmoid(pwm_t, P.MOSFET_T_HIGH, sharpness=0.5)

    # 종합: 편차 기반을 주로 하되, 발열 상태로 보정
    p_dac = 0.6 * p_dv + 0.25 * p_dac_hot + 0.15 * p_pwm_hot
    p_dac += random.uniform(-0.05, 0.05)
    return max(0.02, min(0.98, p_dac))


# ============================================================
# 셀 출력값 (회귀 타깃)
# ============================================================
def get_dac_value(pwm_t, dac_t, cell_v, delta, P):
    """DAC 출력값 (0~4095). 편차 비례 + DAC 모스펫 과열 시 감쇄."""
    val = (delta / 0.15) * 4095
    if dac_t >= P.MOSFET_T_HIGH:
        val *= max(0.1, 1.0 - (dac_t - P.MOSFET_T_HIGH) * P.DAC_DERATE)
    if cell_v < 3.3:
        val *= 0.4
    val += random.uniform(-50, 50)
    return int(min(4095, max(0, val)))


def get_pwm_duty(pwm_t, dac_t, cell_v, delta, P):
    """PWM 듀티비 (0~255). 편차 비례 + PWM 모스펫 과열 시 감쇄(약하게)."""
    val = (delta / 0.15) * 255
    if pwm_t >= P.MOSFET_T_HIGH:
        val *= max(0.2, 1.0 - (pwm_t - P.MOSFET_T_HIGH) * P.PWM_DERATE)
    if cell_v < 3.3:
        val *= 0.5
    val += random.uniform(-5, 5)
    return int(min(255, max(0, val)))


# ============================================================
# 인과 기반 샘플 (80%) — ΔV로부터 두 모스펫 온도 생성
# ============================================================
def generate_pack_sample_causal(P):
    pack_dv = round(random.uniform(*P.PACK_DV_RANGE), 4)
    # DAC 모스펫: ΔV에 강하게 비례 (선형손실 큼)
    dac_t = AMBIENT_T + pack_dv * random.uniform(*P.DAC_HEAT_COEF_PACK) + random.uniform(-3, 3)
    dac_t = max(AMBIENT_T, min(P.MOSFET_DAC_T_RANGE[1], dac_t))
    # PWM 모스펫: ΔV에 약하게 비례 (스위칭손실 작음)
    pwm_t = AMBIENT_T + pack_dv * random.uniform(*P.PWM_HEAT_COEF_PACK) + random.uniform(-2, 2)
    pwm_t = max(AMBIENT_T, min(P.MOSFET_PWM_T_RANGE[1], pwm_t))

    p_dac = get_pack_dac_probability(pwm_t, dac_t, pack_dv, P)
    label = "DAC" if random.random() < p_dac else "PWM"
    return round(pwm_t, 1), round(dac_t, 1), pack_dv, label, round(p_dac, 3)


def generate_cell_sample_causal(P):
    delta = round(random.uniform(*P.DELTA_RANGE), 4)
    # 셀 전압: 편차 클수록 방전 진행 → 전압 낮음
    base_v = random.uniform(3.5, 4.2)
    cell_v = base_v - delta * random.uniform(0.3, 0.7)
    cell_v = round(max(P.CELL_V_RANGE[0], min(P.CELL_V_RANGE[1], cell_v)), 3)

    # DAC 모스펫 온도 (강한 발열)
    dac_t = AMBIENT_T + delta * random.uniform(*P.DAC_HEAT_COEF_CELL) + random.uniform(-3, 3)
    if cell_v < 3.3:
        dac_t += random.uniform(*P.LOW_V_PENALTY)   # 저전압 시 내부저항↑ 추가발열
    dac_t = round(max(AMBIENT_T, min(P.MOSFET_DAC_T_RANGE[1], dac_t)), 1)

    # PWM 모스펫 온도 (약한 발열)
    pwm_t = AMBIENT_T + delta * random.uniform(*P.PWM_HEAT_COEF_CELL) + random.uniform(-2, 2)
    pwm_t = round(max(AMBIENT_T, min(P.MOSFET_PWM_T_RANGE[1], pwm_t)), 1)

    dac_val = get_dac_value(pwm_t, dac_t, cell_v, delta, P)
    pwm_duty = get_pwm_duty(pwm_t, dac_t, cell_v, delta, P)
    return pwm_t, dac_t, cell_v, delta, dac_val, pwm_duty


# ============================================================
# 독립 random 샘플 (20%) — 과적합 방지용 노이즈/엣지케이스
# ============================================================
def generate_pack_sample_independent(P):
    pwm_t = round(random.uniform(*P.MOSFET_PWM_T_RANGE), 1)
    dac_t = round(random.uniform(*P.MOSFET_DAC_T_RANGE), 1)
    pack_dv = round(random.uniform(*P.PACK_DV_RANGE), 4)
    p_dac = get_pack_dac_probability(pwm_t, dac_t, pack_dv, P)
    label = "DAC" if random.random() < p_dac else "PWM"
    return pwm_t, dac_t, pack_dv, label, round(p_dac, 3)


def generate_cell_sample_independent(P):
    pwm_t = round(random.uniform(*P.MOSFET_PWM_T_RANGE), 1)
    dac_t = round(random.uniform(*P.MOSFET_DAC_T_RANGE), 1)
    cell_v = round(random.uniform(*P.CELL_V_RANGE), 3)
    delta = round(random.uniform(*P.DELTA_RANGE), 4)
    dac_val = get_dac_value(pwm_t, dac_t, cell_v, delta, P)
    pwm_duty = get_pwm_duty(pwm_t, dac_t, cell_v, delta, P)
    return pwm_t, dac_t, cell_v, delta, dac_val, pwm_duty


# 80/20 혼합 샘플러
def generate_pack_sample(P):
    if random.random() < INDEPENDENT_RATIO:
        return generate_pack_sample_independent(P)
    return generate_pack_sample_causal(P)


def generate_cell_sample(P):
    if random.random() < INDEPENDENT_RATIO:
        return generate_cell_sample_independent(P)
    return generate_cell_sample_causal(P)


# ============================================================
# CSV 생성
# ============================================================
def make_pack_csv(P):
    """모드 분류기용 데이터 (PWM/DAC 균형 맞춤)"""
    dac_list, pwm_list = [], []
    target = NUM_PACK_SAMPLES // 2
    attempts = 0
    while (len(dac_list) < target or len(pwm_list) < target) and attempts < NUM_PACK_SAMPLES * 15:
        s = generate_pack_sample(P)
        if s[3] == "DAC" and len(dac_list) < target:
            dac_list.append(s)
        elif s[3] == "PWM" and len(pwm_list) < target:
            pwm_list.append(s)
        attempts += 1
    all_data = dac_list + pwm_list
    random.shuffle(all_data)
    with open(PACK_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Mosfet_PWM_T", "Mosfet_DAC_T", "Pack_Delta_V", "Label", "P_DAC"])
        w.writerows(all_data)
    print(f"  [PACK] {len(all_data)}개 (PWM:{len(pwm_list)} / DAC:{len(dac_list)}) -> {PACK_FILE}")


def make_cell_csv(P):
    """출력 회귀기용 데이터"""
    samples = [generate_cell_sample(P) for _ in range(NUM_CELL_SAMPLES)]
    with open(CELL_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Mosfet_PWM_T", "Mosfet_DAC_T", "Cell_V", "Delta_From_Min", "DAC_Val", "PWM_Duty"])
        w.writerows(samples)
    print(f"  [CELL] {len(samples)}개 -> {CELL_FILE}")


def main():
    print("=" * 60)
    print("A²-BMS 학습 데이터 생성 (듀얼 MOSFET, 소전류)")
    print(f"  feature: 전압 + PWM모스펫온도 + DAC모스펫온도")
    print(f"  인과 {int((1-INDEPENDENT_RATIO)*100)}% / 독립 {int(INDEPENDENT_RATIO*100)}%")
    print("=" * 60)
    make_pack_csv(Params)
    make_cell_csv(Params)
    print("\n완료. 다음: python train_system.py")

    # 인과 관계 검증 미리보기
    print("\n[검증] ΔV별 평균 모스펫 온도 (DAC가 PWM보다 뜨거워야 정상)")
    import statistics
    with open(PACK_FILE) as f:
        rows = list(csv.reader(f))[1:]
    for lo, hi in [(0, 0.03), (0.06, 0.09), (0.12, 0.15)]:
        sub = [r for r in rows if lo <= float(r[2]) < hi]
        if sub:
            pwm_avg = statistics.mean(float(r[0]) for r in sub)
            dac_avg = statistics.mean(float(r[1]) for r in sub)
            print(f"  ΔV {lo}~{hi}: PWM모스펫 {pwm_avg:.1f}°C / DAC모스펫 {dac_avg:.1f}°C")


if __name__ == "__main__":
    main()
