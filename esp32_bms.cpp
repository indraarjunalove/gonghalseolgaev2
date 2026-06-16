/*
 * ============================================================
 * A2-BMS ESP32 Firmware (듀얼 MOSFET 측정 버전)
 * ============================================================
 *
 * [ESP32 역할]
 * 1. Raspberry Pi로부터 UART(JSON) 명령 수신
 * 2. 셀별 PWM MOSFET(Q_PWM) 제어
 * 3. 셀별 MCP4725 DAC → Op-Amp → Q_LINEAR 제어
 * 4. 셀별 ADS1115 센서 측정
 * 5. 전압 / 전류 / PWM모스펫온도 / DAC모스펫온도 송신
 * 6. 통신 끊김, 과전압, 저전압, 과온 시 출력 차단
 *
 * [듀얼 MOSFET 측정 - 이 버전의 핵심]
 * 모드별 발열 특성이 정반대이므로 두 MOSFET 온도를 따로 측정한다.
 *   · Q_PWM   (스위칭) : PWM 모드에서 발열, I²×R_DS_on (작음)
 *   · Q_LINEAR(선형)   : DAC 모드에서 발열, V_DS×I (큼)
 * → 라즈베리파이 AI가 두 온도를 비교해 발열 유리한 모드를 선택.
 *
 * [ADS1115 채널 매핑]
 * A0 = 셀 전압 BATn+
 * A1 = Vsense (RSENSE 상단, 전류 측정)
 * A2 = Q_PWM 모스펫 NTC      → JSON "mt"
 * A3 = Q_LINEAR 모스펫 NTC   → JSON "dt"   ★ 신규
 *
 * [배터리 온도 제거]
 * 소전류(~0.5A) 환경에서 셀 발열은 무시 가능(I²R≈0.006W)하므로
 * DS18B20 배터리 온도 측정을 제거하고, 발열 주체인 두 MOSFET을 측정한다.
 *
 * [UART 명령 예시]
 * {"mode":"PWM","dac":[0,0,0,0],"pwm":[200,150,100,50]}
 *   STOP = 전체 차단 / PWM = PWM만 / DAC = DAC만 / HYBRID = 동시
 *
 * [송신 JSON 예시]
 * {"v":[..],"i":[..],"mt":[..],"dt":[..],"safe":true,"shutdown":false}
 */

#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_MCP4725.h>

// ============================================================
// ESP32 핀 설정
// ============================================================
#define I2C_SDA 21
#define I2C_SCL 22

// 셀별 PWM MOSFET(Q_PWM) 게이트 제어 핀
// Cell1→GPIO25, Cell2→GPIO26, Cell3→GPIO27, Cell4→GPIO14
const int PWM_PINS[4] = {25, 26, 27, 14};
const int PWM_CHANNELS[4] = {0, 1, 2, 3};   // ESP32 LEDC 채널
const int PWM_FREQ = 5000;   // 5kHz
const int PWM_RES = 8;       // 8bit → duty 0~255

// ============================================================
// I2C 주소
// ============================================================
#define TCA_ADDR 0x70   // TCA9548A 멀티플렉서
#define ADS_ADDR 0x48   // ADS1115 (TCA로 채널 분리되어 주소 같아도 됨)
#define DAC_ADDR 0x60   // MCP4725

// TCA9548A 채널 매핑 (CH0=Cell1 ... CH3=Cell4)
const uint8_t CELL_TCA_CH[4] = {0, 1, 2, 3};

// ============================================================
// ADS1115 채널 매핑 (듀얼 MOSFET)
// ============================================================
#define ADS_CH_CELL_VOLTAGE 0  // A0: 셀 전압
#define ADS_CH_VSENSE       1  // A1: Vsense (전류)
#define ADS_CH_NTC_PWM      2  // A2: Q_PWM 모스펫 NTC
#define ADS_CH_NTC_DAC      3  // A3: Q_LINEAR 모스펫 NTC  ★ 신규

// ============================================================
// 회로 상수
// ============================================================
const float R_SENSE = 0.1;        // 전류 센싱 저항 (I = Vsense / R_SENSE)

// NTC 상수 (VCC_BATn -- 10kΩ -- NTC_sense -- NTC -- BATn-)
const float NTC_R_FIXED = 10000.0;
const float NTC_BETA = 3950.0;
const float NTC_R0 = 10000.0;
const float T0_K = 298.15;

// ============================================================
// 안전 기준값
// ============================================================
const float CELL_OV_LIMIT = 4.25;   // 과전압 차단
const float CELL_UV_LIMIT = 2.80;   // 저전압 차단
const float MOS_TEMP_LIMIT = 70.0;  // 모스펫 과온 차단 (두 MOSFET 모두)

const unsigned long CMD_TIMEOUT_MS = 3000;   // 명령 끊김 차단 시간
const unsigned long SEND_INTERVAL_MS = 1000; // 센서 송신 주기

// ============================================================
// 전역 객체
// ============================================================
Adafruit_ADS1115 ads;
Adafruit_MCP4725 dac;

unsigned long lastCmdTime = 0;
unsigned long lastSendTime = 0;

// ============================================================
// TCA9548A 채널 선택
//   ESP32가 먼저 채널을 고른 뒤 해당 셀의 ADS/DAC와 통신
// ============================================================
bool tcaSelect(uint8_t ch) {
  if (ch > 7) return false;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << ch);   // ch=2이면 00000100
  return Wire.endTransmission() == 0;
}

bool selectCell(uint8_t cell) {
  if (cell >= 4) return false;
  return tcaSelect(CELL_TCA_CH[cell]);
}

// ============================================================
// 셀별 ADS1115/MCP4725 초기화
// ============================================================
bool initCellDevices(uint8_t cell) {
  if (!selectCell(cell)) return false;
  bool adsOk = ads.begin(ADS_ADDR);
  bool dacOk = dac.begin(DAC_ADDR);
  if (dacOk) dac.setVoltage(0, false);   // 부팅 시 DAC 0
  return adsOk && dacOk;
}

// ============================================================
// 전체 출력 차단 (안전)
//   모든 셀의 PWM + DAC를 0으로 → 밸런싱 MOSFET 전부 OFF
// ============================================================
void forceShutdown() {
  for (int cell = 0; cell < 4; cell++) {
    ledcWrite(PWM_CHANNELS[cell], 0);          // Q_PWM OFF
    if (selectCell(cell)) dac.setVoltage(0, false);  // Q_LINEAR OFF
  }
}

// ============================================================
// ADS1115 전압 읽기 (채널별 gain 자동 설정)
//   A0 셀전압: ±6.144V / A1 Vsense: ±0.256V(16배) / NTC: ±4.096V
// ============================================================
float readADSVoltage(uint8_t cell, uint8_t channel) {
  if (!selectCell(cell)) return -1.0;

  if (channel == ADS_CH_CELL_VOLTAGE) {
    ads.setGain(GAIN_TWOTHIRDS);
  } else if (channel == ADS_CH_VSENSE) {
    ads.setGain(GAIN_SIXTEEN);   // 작은 전압이라 고배율
  } else {
    ads.setGain(GAIN_ONE);       // NTC 분압
  }

  delay(2);   // gain 변경 후 안정화
  int16_t raw = ads.readADC_SingleEnded(channel);
  return ads.computeVolts(raw);
}

// ============================================================
// 셀 전압 측정 (A0)
// ============================================================
float readCellVoltage(uint8_t cell) {
  return readADSVoltage(cell, ADS_CH_CELL_VOLTAGE);
}

// ============================================================
// 밸런싱 전류 측정 (A1, I = Vsense / R_SENSE)
// ============================================================
float readBalanceCurrent(uint8_t cell) {
  float vsense = readADSVoltage(cell, ADS_CH_VSENSE);
  if (vsense < 0) return -1.0;
  return vsense / R_SENSE;
}

// ============================================================
// NTC 온도 측정 (공통 함수 - 채널을 인자로 받음)
//   R_NTC = R_FIXED × Vntc / (3.3 - Vntc), 이후 Beta 식
// ============================================================
float readNtcTemp(uint8_t cell, uint8_t ads_channel) {
  float v = readADSVoltage(cell, ads_channel);
  if (v <= 0.01 || v >= 3.29) return -999.0;   // 비정상 방어
  float r_ntc = NTC_R_FIXED * v / (3.3 - v);
  float tempK = 1.0 / ((1.0 / T0_K) + (log(r_ntc / NTC_R0) / NTC_BETA));
  return tempK - 273.15;
}

// Q_PWM 모스펫 온도 (A2) → JSON "mt"
float readPwmMosTemp(uint8_t cell) {
  return readNtcTemp(cell, ADS_CH_NTC_PWM);
}

// Q_LINEAR(DAC) 모스펫 온도 (A3) → JSON "dt"
float readDacMosTemp(uint8_t cell) {
  return readNtcTemp(cell, ADS_CH_NTC_DAC);
}

// ============================================================
// DAC 출력 설정 (MCP4725, 0~4095 → Op-Amp → Q_LINEAR)
// ============================================================
void setCellDAC(uint8_t cell, int value) {
  value = constrain(value, 0, 4095);
  if (!selectCell(cell)) return;
  dac.setVoltage(value, false);
}

// ============================================================
// PWM 출력 설정 (LEDC, 0~255 → Q_PWM 게이트)
// ============================================================
void setCellPWM(uint8_t cell, int duty) {
  if (cell >= 4) return;
  duty = constrain(duty, 0, 255);
  ledcWrite(PWM_CHANNELS[cell], duty);
}

// ============================================================
// 안전 상태 검사 (두 모스펫 온도 모두 확인)
//   mt = Q_PWM 온도, dt = Q_LINEAR 온도
//   하나라도 위험이면 false → forceShutdown()
// ============================================================
bool isSafe(float v[4], float i[4], float mt[4], float dt[4]) {
  for (int cell = 0; cell < 4; cell++) {
    // 센서 읽기 실패
    if (v[cell] < 0) return false;
    if (i[cell] < 0) return false;
    if (mt[cell] == -999.0) return false;
    if (dt[cell] == -999.0) return false;

    // 셀 과전압 / 저전압
    if (v[cell] > CELL_OV_LIMIT) return false;
    if (v[cell] < CELL_UV_LIMIT) return false;

    // 두 모스펫 과온 검사
    if (mt[cell] > MOS_TEMP_LIMIT) return false;   // Q_PWM
    if (dt[cell] > MOS_TEMP_LIMIT) return false;   // Q_LINEAR
  }
  return true;
}

// ============================================================
// Raspberry Pi 명령 처리
//   {"mode":"PWM","dac":[..],"pwm":[..]}
//   STOP=차단 / PWM=PWM만 / DAC=DAC만 / HYBRID=동시
// ============================================================
void processCommand(String json) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, json);
  if (err) { forceShutdown(); return; }

  const char* mode = doc["mode"];
  if (mode == nullptr) { forceShutdown(); return; }

  if (strcmp(mode, "STOP") == 0) { forceShutdown(); return; }

  JsonArray dac_arr = doc["dac"];
  JsonArray pwm_arr = doc["pwm"];
  if (dac_arr.size() < 4 || pwm_arr.size() < 4) { forceShutdown(); return; }

  if (strcmp(mode, "PWM") == 0) {
    for (int cell = 0; cell < 4; cell++) {
      setCellDAC(cell, 0);                      // DAC 끄고
      setCellPWM(cell, pwm_arr[cell].as<int>()); // PWM만
    }
  }
  else if (strcmp(mode, "DAC") == 0) {
    for (int cell = 0; cell < 4; cell++) {
      setCellPWM(cell, 0);                       // PWM 끄고
      setCellDAC(cell, dac_arr[cell].as<int>()); // DAC만
    }
  }
  else if (strcmp(mode, "HYBRID") == 0) {
    for (int cell = 0; cell < 4; cell++) {
      setCellDAC(cell, dac_arr[cell].as<int>());
      setCellPWM(cell, pwm_arr[cell].as<int>());
    }
  }
  else {
    forceShutdown();   // 알 수 없는 모드
  }
}

// ============================================================
// 센서 데이터 송신 (듀얼 MOSFET)
//   {"v":[..],"i":[..],"mt":[..],"dt":[..],"safe":..,"shutdown":..}
//     mt = Q_PWM 온도, dt = Q_LINEAR 온도
// ============================================================
void sendSensorData() {
  StaticJsonDocument<768> doc;

  float v[4], i[4], mt[4], dt[4];

  JsonArray v_arr  = doc.createNestedArray("v");
  JsonArray i_arr  = doc.createNestedArray("i");
  JsonArray mt_arr = doc.createNestedArray("mt");  // Q_PWM 온도
  JsonArray dt_arr = doc.createNestedArray("dt");  // Q_LINEAR 온도

  for (int cell = 0; cell < 4; cell++) {
    v[cell]  = readCellVoltage(cell);
    i[cell]  = readBalanceCurrent(cell);
    mt[cell] = readPwmMosTemp(cell);   // A2
    dt[cell] = readDacMosTemp(cell);   // A3

    v_arr.add(v[cell]);
    i_arr.add(i[cell]);
    mt_arr.add(mt[cell]);
    dt_arr.add(dt[cell]);
  }

  bool safe = isSafe(v, i, mt, dt);
  doc["safe"] = safe;

  if (!safe) {
    forceShutdown();
    doc["shutdown"] = true;
  } else {
    doc["shutdown"] = false;
  }

  serializeJson(doc, Serial);
  Serial.println();
}

// ============================================================
// setup
// ============================================================
void setup() {
  Serial.begin(115200);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);   // 100kHz

  bool allInitOk = true;
  for (int cell = 0; cell < 4; cell++) {
    if (!initCellDevices(cell)) {
      allInitOk = false;
      Serial.print("{\"cell_init_error\":");
      Serial.print(cell + 1);
      Serial.println("}");
    }
  }

  // PWM 초기화
  for (int cell = 0; cell < 4; cell++) {
    ledcSetup(PWM_CHANNELS[cell], PWM_FREQ, PWM_RES);
    ledcAttachPin(PWM_PINS[cell], PWM_CHANNELS[cell]);
    ledcWrite(PWM_CHANNELS[cell], 0);
  }

  forceShutdown();   // 부팅 직후 전체 차단 (안전)

  lastCmdTime = millis();
  lastSendTime = millis();

  if (allInitOk) {
    Serial.println("{\"status\":\"ESP32_READY_DUAL_MOSFET\"}");
  } else {
    Serial.println("{\"status\":\"ESP32_READY_WITH_INIT_ERROR\"}");
  }
}

// ============================================================
// loop
// ============================================================
void loop() {
  // 라즈베리파이 명령 수신
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    processCommand(line);
    lastCmdTime = millis();   // 통신 들어왔으니 시간 갱신
  }

  // 명령 끊기면 전체 차단 (Fail-Safe)
  if (millis() - lastCmdTime > CMD_TIMEOUT_MS) {
    forceShutdown();
  }

  // 1초마다 센서 송신
  if (millis() - lastSendTime > SEND_INTERVAL_MS) {
    sendSensorData();
    lastSendTime = millis();
  }
}
