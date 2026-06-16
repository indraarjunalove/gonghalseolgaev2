"""
A²-BMS 재학습 스크립트 (실측 로그 + 기존 합성 데이터 병합)
============================================================
[목적]
실제 하드웨어에서 수집한 로그(logs/real_data_*.csv)를 기존 합성 학습
데이터(bms_pack_data.csv, bms_cell_data.csv)와 병합하여 모델을 다시 학습.

[안전 설계]
- 실측 로그만으로 학습하면 데이터가 적어 과적합 위험 → 기존 합성 데이터와
  합쳐서 학습 (실측이 분포를 보정, 합성이 일반화를 유지)
- 기존 .pkl은 백업 후 덮어씀 (문제 시 롤백 가능)

[로컬 전용]
Render 같은 클라우드에선 실행하지 말 것 (메모리/시간 제한 + 파일 휘발).
라즈베리파이 또는 PC에서 실행.

[실행]
  python retrain.py            # logs/ 의 모든 로그 병합 재학습
  python retrain.py --logs-only  # (옵션) 실측만으로 학습 - 비추천
"""
import os
import glob
import sys
import shutil
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

LOG_DIR = 'logs'
PACK_FILE = 'bms_pack_data.csv'    # 기존 합성 데이터 (팩)
CELL_FILE = 'bms_cell_data.csv'    # 기존 합성 데이터 (셀)

PACK_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Cell_V', 'Delta_From_Min']


# ============================================================
# 1. 실측 로그 → 학습 가능한 형태로 변환
# ============================================================
def load_real_logs():
    """
    logs/real_data_*.csv 파일들을 읽어 팩/셀 학습 데이터로 변환.
    로그 1줄(1초)에는 셀 4개의 상태 + AI가 내린 모드/출력이 들어있음.
    이를 '입력→정답' 형태로 재구성.
    """
    log_files = glob.glob(f'{LOG_DIR}/real_data_*.csv')
    if not log_files:
        print(f"[INFO] {LOG_DIR}/ 에 로그 파일 없음 - 기존 데이터로만 학습")
        return None, None

    print(f"[로그] {len(log_files)}개 파일 발견:")
    for f in log_files:
        print(f"  - {f}")

    pack_rows = []   # 팩 단위 (모드 분류용)
    cell_rows = []   # 셀 단위 (출력 회귀용)

    for lf in log_files:
        try:
            df = pd.read_csv(lf)
        except Exception as e:
            print(f"  [무시] {lf} 읽기 실패: {e}")
            continue

        for _, row in df.iterrows():
            mode = row.get('ai_mode')
            # STOP은 학습 대상 아님 (안전 룰이라 AI 판단 아님)
            if mode not in ('PWM', 'DAC'):
                continue

            # --- 팩 단위 (모드 분류) ---
            # 입력: 두 모스펫 최고온도 + 전압편차 / 정답: 모드
            max_pwm_t = max(row[f'pwm_mosfet_t_{i}'] for i in range(4))
            max_dac_t = max(row[f'dac_mosfet_t_{i}'] for i in range(4))
            pack_dv = row['pack_delta_v']
            pack_rows.append([max_pwm_t, max_dac_t, pack_dv, mode])

            # --- 셀 단위 (출력 회귀) ---
            vs = [row[f'v_{i}'] for i in range(4)]
            min_v = min(vs)
            for i in range(4):
                pwm_t = row[f'pwm_mosfet_t_{i}']
                dac_t = row[f'dac_mosfet_t_{i}']
                cell_v = row[f'v_{i}']
                delta = cell_v - min_v
                dac_val = row[f'dac_{i}']
                pwm_duty = row[f'pwm_{i}']
                cell_rows.append([pwm_t, dac_t, cell_v, delta, dac_val, pwm_duty])

    if not pack_rows:
        print("[INFO] 유효한 실측 샘플 없음 (PWM/DAC 라벨 없음)")
        return None, None

    pack_df = pd.DataFrame(pack_rows, columns=['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Pack_Delta_V', 'Label'])
    cell_df = pd.DataFrame(cell_rows, columns=['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Cell_V', 'Delta_From_Min', 'DAC_Val', 'PWM_Duty'])
    print(f"[로그] 실측 변환 완료 - 팩 {len(pack_df)}개 / 셀 {len(cell_df)}개")
    return pack_df, cell_df


# ============================================================
# 2. 기존 합성 데이터 + 실측 로그 병합
# ============================================================
def merge_data(logs_only=False):
    real_pack, real_cell = load_real_logs()

    if logs_only:
        if real_pack is None:
            print("[ERROR] 실측 로그 없음 - logs-only 불가")
            sys.exit(1)
        print("[병합] 실측 로그만 사용 (비추천)")
        return real_pack, real_cell

    # 기존 합성 데이터 로드
    base_pack = pd.read_csv(PACK_FILE)[['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Pack_Delta_V', 'Label']]
    base_cell = pd.read_csv(CELL_FILE)[CELL_FEATURES + ['DAC_Val', 'PWM_Duty']]
    print(f"[병합] 기존 합성 - 팩 {len(base_pack)}개 / 셀 {len(base_cell)}개")

    if real_pack is None:
        print("[병합] 실측 없음 → 기존 데이터로만 재학습")
        return base_pack, base_cell

    # 합치기
    merged_pack = pd.concat([base_pack, real_pack], ignore_index=True)
    merged_cell = pd.concat([base_cell, real_cell], ignore_index=True)
    print(f"[병합] 최종 - 팩 {len(merged_pack)}개 / 셀 {len(merged_cell)}개")
    return merged_pack, merged_cell


# ============================================================
# 3. 기존 모델 백업
# ============================================================
def backup_models():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f'model_backup_{ts}'
    models = ['bms_mode_ai.pkl', 'bms_dac_ai.pkl', 'bms_pwm_ai.pkl']
    existing = [m for m in models if os.path.exists(m)]
    if existing:
        os.makedirs(backup_dir, exist_ok=True)
        for m in existing:
            shutil.copy(m, f'{backup_dir}/{m}')
        print(f"[백업] 기존 모델 → {backup_dir}/")


# ============================================================
# 4. 재학습
# ============================================================
def retrain(pack_df, cell_df):
    print("\n" + "=" * 60)
    print("재학습 시작")
    print("=" * 60)

    # 모드 분류기
    X = pack_df[PACK_FEATURES]
    y = pack_df['Label']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    print(f"  [모드 분류기] 정확도: {accuracy_score(y_te, clf.predict(X_te))*100:.2f}%")
    joblib.dump(clf, 'bms_mode_ai.pkl')

    # DAC 회귀기
    Xc = cell_df[CELL_FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(Xc, cell_df['DAC_Val'], test_size=0.2, random_state=42)
    reg_dac = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    reg_dac.fit(X_tr, y_tr)
    print(f"  [DAC 회귀기] MAE: {mean_absolute_error(y_te, reg_dac.predict(X_te)):.2f} / 4095")
    joblib.dump(reg_dac, 'bms_dac_ai.pkl')

    # PWM 회귀기
    X_tr, X_te, y_tr, y_te = train_test_split(Xc, cell_df['PWM_Duty'], test_size=0.2, random_state=42)
    reg_pwm = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    reg_pwm.fit(X_tr, y_tr)
    print(f"  [PWM 회귀기] MAE: {mean_absolute_error(y_te, reg_pwm.predict(X_te)):.2f} / 255")
    joblib.dump(reg_pwm, 'bms_pwm_ai.pkl')

    print("=" * 60)
    print("재학습 완료 - 새 .pkl 3개 저장")
    print("=" * 60)


def main():
    logs_only = '--logs-only' in sys.argv
    print("=" * 60)
    print("A²-BMS 재학습 (실측 로그 + 기존 데이터 병합)")
    print("=" * 60)
    backup_models()
    pack_df, cell_df = merge_data(logs_only=logs_only)
    retrain(pack_df, cell_df)


if __name__ == "__main__":
    main()
