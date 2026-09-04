# 온도·습도 공정관리 및 소음 안전관리

## 파일 구성

- `arduino_sensor_monitor.ino`: Arduino UNO R4용 5초 측정 프로그램
- `serial_collector.py`: 시리얼 데이터를 SQLite에 저장하는 Python 프로그램
- `dashboard.py`: 별도로 제공되는 공정관리·안전관리 Streamlit 프로그램
- `wiring_diagram.svg` / `wiring_diagram.png`: 센서 배선도
- `requirements.txt`: Python 설치 패키지 목록

## 배선

| Arduino | 연결 대상 |
|---|---|
| 5V | 브레드보드 적색 `+` 레일 |
| GND | 브레드보드 청색 `-` 레일 |
| D3 | DHT11 `S/DATA` |
| A0 | 소리센서 `AO` |

- DHT11의 `+`는 5V 레일, `-`는 GND 레일에 연결합니다.
- 소리센서의 `+`는 5V 레일, `G`는 GND 레일에 연결합니다.
- 소리센서 `DO`는 사용하지 않습니다.
- 실제 센서의 핀 배열은 PCB에 인쇄된 글자를 최종 기준으로 확인합니다.

## 실행 순서

1. Arduino IDE에서 DHT sensor library를 설치합니다.
2. `arduino_sensor_monitor.ino`를 UNO R4 Minima에 업로드합니다.
3. 명령 프롬프트에서 다음 패키지를 설치합니다.

   ```bash
   pip install -r requirements.txt
   ```

4. 기존 `sensor_data.db`가 있는 폴더에서 수집기를 실행합니다.

   ```bash
   python serial_collector.py
   ```

5. 별도 명령 프롬프트에서 대시보드를 실행합니다.

   ```bash
   streamlit run dashboard.py
   ```

## 소음 보정

사진의 아날로그 소리센서는 공인 소음계가 아니므로 dB가 자동으로 정확해지지 않습니다.
기준 소음계를 옆에 놓고 일정한 소리를 발생시킨 후 Arduino 코드의 다음 두 값을 조정합니다.

- `CALIBRATION_DB`: 기준 소음계의 dB값
- `CALIBRATION_RMS_VOLTAGE`: 같은 순간 Arduino가 출력한 RMS 전압

대시보드의 80 dB와 85 dB는 수정 가능한 예시값이며 법적 작업환경측정을 대신하지 않습니다.
