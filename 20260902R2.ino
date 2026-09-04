#include <DHT.h>
#include <math.h>


// ============================================================
// 1. 하드웨어 및 측정 설정
// ============================================================

const int DHT_PIN = 3;
const int SOUND_ANALOG_PIN = A0;
const int DHT_TYPE = DHT11;

const unsigned long NOISE_MEASUREMENT_MS = 1000;
const unsigned int SAMPLE_INTERVAL_US = 1000;

// UNO R4 Minima에서 12비트 ADC를 사용합니다.
const int ADC_BITS = 12;
const double ADC_MAX_VALUE = 4095.0;
const double ADC_REFERENCE_VOLTAGE = 5.0;

// 반드시 실제 소음계와 비교하여 사용자 환경에 맞게 보정하십시오.
// 예: 기준 소음계가 60 dB일 때 측정된 RMS 전압이 0.010 V라면 아래 값 사용.
const double CALIBRATION_DB = 60.0;
const double CALIBRATION_RMS_VOLTAGE = 0.010;

DHT dht(DHT_PIN, DHT_TYPE);


// ============================================================
// 2. 측정 결과 구조체
// ============================================================

struct NoiseMeasurement {
  unsigned long sampleCount;
  int minimumRaw;
  int maximumRaw;
  int peakToPeakRaw;
  double rmsCounts;
  double rmsVoltage;
  double estimatedDb;
};


struct ClimateMeasurement {
  float temperature;
  float humidity;
  bool valid;
};


// ============================================================
// 3. 센서 초기화
// ============================================================

void initializeSensors() {
  analogReadResolution(ADC_BITS);
  pinMode(SOUND_ANALOG_PIN, INPUT);
  dht.begin();
}


// ============================================================
// 4. 소음 측정 함수
// ============================================================

NoiseMeasurement measureNoiseForFiveSeconds() {
  NoiseMeasurement result;
  result.sampleCount = 0;
  result.minimumRaw = (int)ADC_MAX_VALUE;
  result.maximumRaw = 0;

  // Welford 알고리즘으로 DC 중심값을 제거한 표준편차(RMS)를 계산합니다.
  double runningMean = 0.0;
  double runningM2 = 0.0;

  unsigned long startTime = millis();

  while (millis() - startTime < NOISE_MEASUREMENT_MS) {
    int rawValue = analogRead(SOUND_ANALOG_PIN);
    result.sampleCount++;

    double delta = rawValue - runningMean;
    runningMean += delta / result.sampleCount;
    double delta2 = rawValue - runningMean;
    runningM2 += delta * delta2;

    if (rawValue < result.minimumRaw) {
      result.minimumRaw = rawValue;
    }
    if (rawValue > result.maximumRaw) {
      result.maximumRaw = rawValue;
    }

    delayMicroseconds(SAMPLE_INTERVAL_US);
  }

  result.peakToPeakRaw = result.maximumRaw - result.minimumRaw;

  if (result.sampleCount > 1) {
    result.rmsCounts = sqrt(runningM2 / (result.sampleCount - 1));
  } else {
    result.rmsCounts = 0.0;
  }

  result.rmsVoltage =
      result.rmsCounts * ADC_REFERENCE_VOLTAGE / ADC_MAX_VALUE;

  if (result.rmsVoltage > 0.000001 && CALIBRATION_RMS_VOLTAGE > 0.0) {
    result.estimatedDb =
        CALIBRATION_DB
        + 20.0 * log10(result.rmsVoltage / CALIBRATION_RMS_VOLTAGE);
  } else {
    result.estimatedDb = 0.0;
  }

  return result;
}


// ============================================================
// 5. 온도·습도 측정 함수
// ============================================================

ClimateMeasurement readClimateSensor() {
  ClimateMeasurement result;
  result.temperature = dht.readTemperature();
  result.humidity = dht.readHumidity();
  result.valid = !isnan(result.temperature) && !isnan(result.humidity);
  return result;
}


// ============================================================
// 6. 시리얼 출력 함수
// ============================================================

void printHumanReadable(
    const ClimateMeasurement &climate,
    const NoiseMeasurement &noise) {
  Serial.println(F("----------------------------------------"));
  Serial.print(F("Measurement window : 1 seconds\n"));

  if (climate.valid) {
    Serial.print(F("Temperature        : "));
    Serial.print(climate.temperature, 1);
    Serial.println(F(" C"));
    Serial.print(F("Humidity           : "));
    Serial.print(climate.humidity, 1);
    Serial.println(F(" %"));
  } else {
    Serial.println(F("Temperature/Humidity: DHT11 read error"));
  }

  Serial.print(F("Noise raw P-P       : "));
  Serial.println(noise.peakToPeakRaw);
  Serial.print(F("Noise RMS voltage   : "));
  Serial.print(noise.rmsVoltage, 6);
  Serial.println(F(" V"));
  Serial.print(F("Estimated noise     : "));
  Serial.print(noise.estimatedDb, 1);
  Serial.println(F(" dB"));
}


void printMachineReadable(
    const ClimateMeasurement &climate,
    const NoiseMeasurement &noise) {
  // Python 수집기가 읽는 형식:
  // DATA,temperature,humidity,noise_raw,rms_voltage,estimated_db
  Serial.print(F("DATA,"));

  if (climate.valid) {
    Serial.print(climate.temperature, 1);
    Serial.print(',');
    Serial.print(climate.humidity, 1);
  } else {
    Serial.print(F("nan,nan"));
  }

  Serial.print(',');
  Serial.print(noise.peakToPeakRaw);
  Serial.print(',');
  Serial.print(noise.rmsVoltage, 6);
  Serial.print(',');
  Serial.println(noise.estimatedDb, 1);
}


// ============================================================
// 7. Arduino 기본 실행 함수
// ============================================================

void setup() {
  Serial.begin(115200);
  initializeSensors();

  Serial.println(F("Arduino Process & Safety Sensor Monitor"));
  Serial.println(F("Noise is measured for 5 seconds per record."));
  Serial.println(F("Estimated dB requires calibration."));
}


void loop() {
  NoiseMeasurement noise = measureNoiseForFiveSeconds();
  ClimateMeasurement climate = readClimateSensor();

  printHumanReadable(climate, noise);
  printMachineReadable(climate, noise);
}
