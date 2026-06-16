"""
A²-BMS AI 모델 학습 + 재라벨링 통합 (소전류, 듀얼 MOSFET)
============================================================
[하는 일]
1. 합성 데이터(data_maker 생성) 로드 - 라벨 그대로 사용
2. 실측 로그(logs/*.csv) 로드 - AI 판단 라벨 버리고 "물리 규칙"으로 재라벨
3. 둘을 병합하여 모델 3개 학습 (모드분류 + DAC회귀 + PWM회귀)
4. 기존 모델 자동 백업

[왜 재라벨링?]
실측 로그의 라벨(ai_mode)은 "AI가 내린 판단"이다. 이걸 그대로 정답으로
학습하면 자기 판단을 자기가 강화하는 확증편향(feedback loop)이 생긴다.
→ 실측의 입력값(전압, 두 모스펫 온도)은 진짜이므로 그대로 쓰되,
  라벨만 데이터시트 기반 물리 규칙(relabel_mode)으로 다시 매긴다.

[라벨 규칙 튜닝]
relabel_mode() 함수의 숫자(임계값)만 수정하면 라벨 기준이 바뀐다.
데이터가 쌓여 규칙을 바꾸고 싶으면 이 함수만 손대면 됨.

[실행]
  python train_system.py              # 합성 + 실측로그 병합 학습
  python train_system.py --synth-only # 합성만 (최초 학습용)
"""
import os
import sys
import glob
import shutil
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

# 합성 데이터 (data_maker 생성)
PACK_FILE = "bms_pack_data.csv"
CELL_FILE = "bms_cell_data.csv"
# 실측 로그 폴더
LOG_DIR = "logs"

PACK_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Cell_V', 'Delta_From_Min']


# ============================================================
# ★ 재라벨링 규칙 (물리 기반) — 여기 숫자만 바꾸면 라벨 기준 변경
# ============================================================
# 임계값 상수 (튜닝 포인트)
RELABEL_STOP_TEMP = 65.0    # 이 온도 이상이면 학습 제외 (STOP은 안전 룰)
RELABEL_STOP_DV   = 0.01    # 이 편차 이하면 학습 제외 (평탄화 완료)
RELABEL_HOT_TEMP  = 50.0    # 모스펫이 "뜨겁다" 판단하는 온도
RELABEL_COOL_TEMP = 45.0    # 반대 모스펫이 "여유있다" 판단하는 온도
RELABEL_DV_HIGH   = 0.08    # 이 편차 이상이면 큰 편차 → PWM(고속)
RELABEL_DV_LOW    = 0.02    # 이 편차 이하면 작은 편차 → DAC(정밀)


def relabel_mode(pwm_t, dac_t, pack_dv):
    """
    측정값(두 모스펫 온도 + 전압편차)으로 PWM/DAC 라벨을 물리 규칙으로 결정.
    AI 판단을 쓰지 않으므로 확증편향 없음.

    규칙 우선순위:
      1) 안전/완료 → 학습 제외 (None 반환)
      2) 발열 회피 (보조): 한쪽 모스펫 뜨겁고 반대쪽 여유 → 반대 모드
      3) 편차 기반 (주력): 편차 크면 PWM(고속), 작으면 DAC(정밀)
    """
    # 1) 안전/완료 구간은 학습 데이터에서 제외
    if max(pwm_t, dac_t) >= RELABEL_STOP_TEMP:
        return None
    if pack_dv <= RELABEL_STOP_DV:
        return None

    # 2) 발열 회피 (보조 규칙)
    # DAC 모스펫이 뜨겁고 PWM 모스펫은 여유 → PWM 선택 (DAC 식히기)
    if dac_t >= RELABEL_HOT_TEMP and pwm_t < RELABEL_COOL_TEMP:
        return "PWM"
    # PWM 모스펫이 뜨겁고 DAC 모스펫은 여유 → DAC 선택
    if pwm_t >= RELABEL_HOT_TEMP and dac_t < RELABEL_COOL_TEMP:
        return "DAC"

    # 3) 편차 기반 (주력 규칙)
    if pack_dv >= RELABEL_DV_HIGH:
        return "PWM"   # 편차 크면 빠른 PWM
    elif pack_dv <= RELABEL_DV_LOW:
        return "DAC"   # 편차 작으면 정밀 DAC
    else:
        # 중간 구간: 발열 적은 쪽 모스펫의 모드 선택
        return "PWM" if pwm_t <= dac_t else "DAC"


def relabel_cell_outputs(pwm_t, dac_t, cell_v, delta):
    """
    셀 출력값(DAC값/PWM듀티)도 물리 규칙으로 재생성.
    편차 비례 + 해당 모스펫 과열 시 감쇄(derating).
    """
    # 기본: 편차 비례
    dac_val = (delta / 0.15) * 4095
    pwm_duty = (delta / 0.15) * 255

    # 해당 모스펫 과열 시 출력 감쇄
    if dac_t >= RELABEL_HOT_TEMP:
        dac_val *= max(0.1, 1.0 - (dac_t - RELABEL_HOT_TEMP) * 0.04)
    if pwm_t >= RELABEL_HOT_TEMP:
        pwm_duty *= max(0.2, 1.0 - (pwm_t - RELABEL_HOT_TEMP) * 0.025)

    # 저전압 셀 보호
    if cell_v < 3.3:
        dac_val *= 0.4
        pwm_duty *= 0.5

    dac_val = int(min(4095, max(0, dac_val)))
    pwm_duty = int(min(255, max(0, pwm_duty)))
    return dac_val, pwm_duty


# ============================================================
# 실측 로그 → 재라벨링된 학습 데이터로 변환
# ============================================================
def load_and_relabel_logs():
    """
    logs/*.csv 의 실측 로그를 읽어:
      - 입력값(전압, 두 모스펫 온도)은 그대로 사용
      - 라벨/출력값은 물리 규칙으로 재생성 (AI 판단 무시)
    반환: (pack_df, cell_df) 또는 (None, None)
    """
    log_files = glob.glob(f"{LOG_DIR}/real_data*.csv")
    if not log_files:
        print(f"  [실측] {LOG_DIR}/ 로그 없음")
        return None, None

    print(f"  [실측] 로그 {len(log_files)}개 발견")
    pack_rows, cell_rows = [], []
    excluded = 0

    for lf in log_files:
        try:
            df = pd.read_csv(lf)
        except Exception as e:
            print(f"    [무시] {lf}: {e}")
            continue

        for _, row in df.iterrows():
            try:
                # 입력값 (진짜 측정값)
                pwm_ts = [row[f'pwm_mosfet_t_{i}'] for i in range(4)]
                dac_ts = [row[f'dac_mosfet_t_{i}'] for i in range(4)]
                vs = [row[f'v_{i}'] for i in range(4)]
            except KeyError:
                continue

            pack_dv = max(vs) - min(vs)
            max_pwm_t = max(pwm_ts)
            max_dac_t = max(dac_ts)

            # --- 팩 단위 재라벨 (AI 판단 무시!) ---
            new_label = relabel_mode(max_pwm_t, max_dac_t, pack_dv)
            if new_label is None:
                excluded += 1
                continue   # STOP/완료 구간은 제외
            pack_rows.append([max_pwm_t, max_dac_t, pack_dv, new_label])

            # --- 셀 단위 재라벨 ---
            min_v = min(vs)
            for i in range(4):
                delta = vs[i] - min_v
                dac_val, pwm_duty = relabel_cell_outputs(pwm_ts[i], dac_ts[i], vs[i], delta)
                cell_rows.append([pwm_ts[i], dac_ts[i], vs[i], delta, dac_val, pwm_duty])

    if not pack_rows:
        print(f"  [실측] 유효 샘플 없음 (제외 {excluded}개)")
        return None, None

    pack_df = pd.DataFrame(pack_rows, columns=PACK_FEATURES + ['Label'])
    cell_df = pd.DataFrame(cell_rows, columns=CELL_FEATURES + ['DAC_Val', 'PWM_Duty'])
    print(f"  [실측] 재라벨 완료 - 팩 {len(pack_df)}개 / 셀 {len(cell_df)}개 (제외 {excluded}개)")
    return pack_df, cell_df


# ============================================================
# 데이터 병합 (합성 + 재라벨된 실측)
# ============================================================
def build_datasets(synth_only=False):
    # 합성 데이터 (data_maker 라벨 그대로)
    base_pack = pd.read_csv(PACK_FILE)[PACK_FEATURES + ['Label']]
    base_cell = pd.read_csv(CELL_FILE)[CELL_FEATURES + ['DAC_Val', 'PWM_Duty']]
    print(f"  [합성] 팩 {len(base_pack)}개 / 셀 {len(base_cell)}개")

    if synth_only:
        print("  [모드] 합성만 사용 (--synth-only)")
        return base_pack, base_cell

    # 실측 로그 (재라벨)
    real_pack, real_cell = load_and_relabel_logs()
    if real_pack is None:
        print("  [병합] 실측 없음 → 합성만으로 학습")
        return base_pack, base_cell

    # 병합
    pack = pd.concat([base_pack, real_pack], ignore_index=True)
    cell = pd.concat([base_cell, real_cell], ignore_index=True)
    print(f"  [병합] 최종 팩 {len(pack)}개 / 셀 {len(cell)}개")
    return pack, cell


# ============================================================
# 기존 모델 백업
# ============================================================
def backup_models():
    models = ['bms_mode_ai.pkl', 'bms_dac_ai.pkl', 'bms_pwm_ai.pkl']
    existing = [m for m in models if os.path.exists(m)]
    if existing:
        backup_dir = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        for m in existing:
            shutil.copy(m, f"{backup_dir}/{m}")
        print(f"  [백업] 기존 모델 → {backup_dir}/")


# ============================================================
# 학습
# ============================================================
def train_classifier(pack_df):
    print("\n[AI 1] 모드 분류기")
    pwm_n = (pack_df['Label'] == 'PWM').sum()
    dac_n = (pack_df['Label'] == 'DAC').sum()
    print(f"  데이터 {len(pack_df)}개 (PWM:{pwm_n} / DAC:{dac_n})")
    X, y = pack_df[PACK_FEATURES], pack_df['Label']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    print(f"  정확도: {accuracy_score(y_te, clf.predict(X_te))*100:.2f}%")
    for n, imp in zip(PACK_FEATURES, clf.feature_importances_):
        print(f"    {n}: {imp*100:.1f}%")
    joblib.dump(clf, 'bms_mode_ai.pkl')
    print("  → bms_mode_ai.pkl 저장")


def train_regressor(cell_df, target_col, rng, model_out, name):
    print(f"\n[AI] {name} 회귀기")
    X, y = cell_df[CELL_FEATURES], cell_df[target_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    reg.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, reg.predict(X_te))
    print(f"  MAE: {mae:.2f} / {rng} ({mae/rng*100:.2f}%)")
    joblib.dump(reg, model_out)
    print(f"  → {model_out} 저장")


def main():
    synth_only = '--synth-only' in sys.argv
    print("=" * 60)
    print("A²-BMS 학습 (합성 + 실측 재라벨 병합)" if not synth_only else "A²-BMS 학습 (합성만)")
    print("=" * 60)

    backup_models()
    pack_df, cell_df = build_datasets(synth_only=synth_only)

    train_classifier(pack_df)
    train_regressor(cell_df, 'DAC_Val', 4095, 'bms_dac_ai.pkl', 'DAC 출력값')
    train_regressor(cell_df, 'PWM_Duty', 255, 'bms_pwm_ai.pkl', 'PWM 듀티비')

    print("\n" + "=" * 60)
    print("학습 완료 - 3개 모델 저장")
    print("=" * 60)


if __name__ == "__main__":
    main()
