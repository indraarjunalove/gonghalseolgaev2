"""
A²-BMS AI 모델 학습 (소전류, 듀얼 MOSFET 측정 기반)
============================================================
3개 모델 학습:
  - bms_mode_ai.pkl : 모드 분류기 (PWM/DAC, 팩 단위)
  - bms_dac_ai.pkl  : DAC 출력값 회귀기 (0~4095, 셀 단위)
  - bms_pwm_ai.pkl  : PWM 듀티비 회귀기 (0~255, 셀 단위)

[입력 feature]
  PACK: PWM모스펫온도, DAC모스펫온도, 전압편차
  CELL: PWM모스펫온도, DAC모스펫온도, 셀전압, 최소셀과의편차
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import joblib

PACK_FILE = "bms_pack_data.csv"
CELL_FILE = "bms_cell_data.csv"

# 듀얼 MOSFET feature (배터리 온도 제외)
PACK_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_PWM_T', 'Mosfet_DAC_T', 'Cell_V', 'Delta_From_Min']


# ============================================================
# AI 1: 모드 분류기 (팩 단위, PWM vs DAC)
# ============================================================
def train_classifier():
    print("=" * 60)
    print("[AI 1] 모드 분류기 (팩 단위)")
    print("=" * 60)
    df = pd.read_csv(PACK_FILE)
    pwm_n = (df['Label'] == 'PWM').sum()
    dac_n = (df['Label'] == 'DAC').sum()
    print(f"  데이터: {len(df)}개 (PWM:{pwm_n} / DAC:{dac_n})")

    X = df[PACK_FEATURES]
    y = df['Label']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    print(f"  정확도: {accuracy_score(y_te, pred)*100:.2f}%")
    print("  Feature 중요도:")
    for n, imp in zip(PACK_FEATURES, clf.feature_importances_):
        print(f"    {n}: {imp*100:.2f}%")

    joblib.dump(clf, 'bms_mode_ai.pkl')
    print("  → bms_mode_ai.pkl 저장")


# ============================================================
# AI 2,3: 출력값 회귀기 (셀 단위)
# ============================================================
def train_regressor(target_col, target_range, model_out, name):
    print(f"\n{'='*60}")
    print(f"[AI] {name} 회귀기 (셀 단위)")
    print("=" * 60)
    df = pd.read_csv(CELL_FILE)
    print(f"  데이터: {len(df)}개")

    X = df[CELL_FEATURES]
    y = df[target_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    print(f"  MAE: {mae:.2f} (전체 {target_range}의 {mae/target_range*100:.2f}%)")
    print("  Feature 중요도:")
    for n, imp in zip(CELL_FEATURES, reg.feature_importances_):
        print(f"    {n}: {imp*100:.2f}%")

    joblib.dump(reg, model_out)
    print(f"  → {model_out} 저장")


def main():
    train_classifier()
    train_regressor('DAC_Val', 4095, 'bms_dac_ai.pkl', 'DAC 출력값')
    train_regressor('PWM_Duty', 255, 'bms_pwm_ai.pkl', 'PWM 듀티비')

    print("\n" + "=" * 60)
    print("학습 완료 - 3개 모델 저장")
    print("  - bms_mode_ai.pkl  (모드 분류기)")
    print("  - bms_dac_ai.pkl   (DAC 회귀기)")
    print("  - bms_pwm_ai.pkl   (PWM 회귀기)")
    print("=" * 60)


if __name__ == "__main__":
    main()
