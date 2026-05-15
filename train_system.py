"""
- AI 1: 모드 분류기 (RandomForestClassifier, 팩 단위)
- AI 2: DAC 출력값 회귀기 (RandomForestRegressor, 셀 단위)
- AI 3: PWM 듀티비 회귀기 (RandomForestRegressor, 셀 단위)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import joblib

PACK_FILE = "bms_pack_data.csv"
CELL_FILE = "bms_cell_data.csv"
PACK_FEATURES = ['Max_Mosfet_T', 'Max_Battery_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_T', 'Battery_T', 'Cell_V', 'Delta_From_Min']

# ============================================================
# AI 1: 모드 분류기 (팩 단위)
# ============================================================
print("=" * 60)
print("[AI 1] 모드 분류기 (팩 단위)")
print("=" * 60)
pack_df = pd.read_csv(PACK_FILE)
print(f"데이터: {len(pack_df)}개 (PWM: {(pack_df['Label']=='PWM').sum()}, DAC: {(pack_df['Label']=='DAC').sum()})")

X = pack_df[PACK_FEATURES]
y = pack_df['Label']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
print(f"\n정확도: {accuracy_score(y_te, pred)*100:.2f}%")
print(classification_report(y_te, pred))
print("Feature 중요도:")
for n, i in zip(PACK_FEATURES, clf.feature_importances_):
    print(f"  {n}: {i*100:.2f}%")

joblib.dump(clf, 'bms_mode_ai2.pkl')
print("\n-> bms_mode_ai2.pkl 저장")

# ============================================================
# AI 2: DAC 출력값 회귀기 (셀 단위)
# ============================================================
print("\n" + "=" * 60)
print("[AI 2] DAC 출력값 회귀기 (셀 단위)")
print("=" * 60)
cell_df = pd.read_csv(CELL_FILE)
print(f"데이터: {len(cell_df)}개")

X = cell_df[CELL_FEATURES]
y_dac = cell_df['DAC_Val']
X_tr, X_te, y_tr, y_te = train_test_split(X, y_dac, test_size=0.2, random_state=42)

reg_dac = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
reg_dac.fit(X_tr, y_tr)
pred = reg_dac.predict(X_te)
mae = mean_absolute_error(y_te, pred)
print(f"\nMAE: {mae:.2f} (전체 범위 4095의 {mae/4095*100:.2f}%)")
print("Feature 중요도:")
for n, i in zip(CELL_FEATURES, reg_dac.feature_importances_):
    print(f"  {n}: {i*100:.2f}%")

joblib.dump(reg_dac, 'bms_dac_ai2.pkl')
print("\n-> bms_dac_ai2.pkl 저장")

# ============================================================
# AI 3: PWM 듀티비 회귀기 (셀 단위)  [신규]
# ============================================================
print("\n" + "=" * 60)
print("[AI 3] PWM 듀티비 회귀기 (셀 단위) [신규]")
print("=" * 60)
y_pwm = cell_df['PWM_Duty']
X_tr, X_te, y_tr, y_te = train_test_split(X, y_pwm, test_size=0.2, random_state=42)

reg_pwm = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
reg_pwm.fit(X_tr, y_tr)
pred = reg_pwm.predict(X_te)
mae = mean_absolute_error(y_te, pred)
print(f"\nMAE: {mae:.2f} (전체 범위 255의 {mae/255*100:.2f}%)")
print("Feature 중요도:")
for n, i in zip(CELL_FEATURES, reg_pwm.feature_importances_):
    print(f"  {n}: {i*100:.2f}%")

joblib.dump(reg_pwm, 'bms_pwm_ai2.pkl')
print("\n-> bms_pwm_ai2.pkl 저장")

print("\n" + "=" * 60)
print("AI 모델 3개 저장 완료")
print("  - bms_mode_ai2.pkl  (분류기)")
print("  - bms_dac_ai2.pkl   (DAC 회귀기)")
print("  - bms_pwm_ai2.pkl   (PWM 회귀기)")
print("=" * 60)

