import argparse
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime

import serial
from serial.tools import list_ports


# ============================================================
# 1. 프로그램 설정
# ============================================================

DB_FILE = "sensor_data.db"
SERIAL_PORT = "COM7"  # --port를 생략했을 때 사용할 기본 포트
BAUD_RATE = 115200
RECONNECT_DELAY_SECONDS = 3


# ============================================================
# 2. 센서 데이터 구조체
# ============================================================

@dataclass
class SensorRecord:
    temperature: float
    humidity: float
    noise_raw: int
    noise_rms_voltage: float
    noise_db: float
    created_at: str


# ============================================================
# 3. 직렬 포트 처리
# ============================================================

def find_arduino_port(configured_port=None):
    """연결된 Arduino 포트를 찾거나 설정된 포트를 반환합니다."""
    if configured_port:
        return configured_port

    if SERIAL_PORT:
        return SERIAL_PORT

    for port in list_ports.comports():
        description = f"{port.description} {port.manufacturer or ''}".lower()
        if "arduino" in description or "uno r4" in description:
            return port.device

    raise RuntimeError(
        "Arduino 포트를 찾지 못했습니다. --port COM7처럼 COM 번호를 지정하세요."
    )


def open_serial_connection(port):
    """Arduino 직렬 연결을 열고 보드 재시작 시간을 기다립니다."""
    connection = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)
    connection.reset_input_buffer()
    return connection


def is_port_busy_error(error):
    """Windows의 COM 포트 점유(액세스 거부) 오류인지 판단합니다."""
    message = str(error).lower()
    return (
        isinstance(error, PermissionError)
        or "access is denied" in message
        or "액세스가 거부" in message
    )


def print_port_busy_guidance(port):
    """COM 포트를 다른 프로그램이 사용 중일 때의 해결 방법을 출력합니다."""
    print("\n" + "=" * 60)
    print(f"{port} 포트가 다른 프로그램에서 사용 중입니다.")
    print("Arduino IDE의 시리얼 모니터와 시리얼 플로터를 모두 닫으세요.")
    print("이미 실행한 다른 Python 수집기도 Ctrl+C로 종료하세요.")
    print("닫은 뒤 이 창은 그대로 두면 자동으로 다시 연결합니다.")
    print("=" * 60 + "\n")


def parse_sensor_line(line):
    """Arduino의 DATA CSV 한 줄을 SensorRecord로 변환합니다."""
    if not line.startswith("DATA,"):
        return None

    fields = line.split(",")
    if len(fields) != 6:
        raise ValueError(f"DATA 필드 수가 올바르지 않습니다: {line}")

    return SensorRecord(
        temperature=float(fields[1]),
        humidity=float(fields[2]),
        noise_raw=int(fields[3]),
        noise_rms_voltage=float(fields[4]),
        noise_db=float(fields[5]),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


# ============================================================
# 4. SQLite 데이터베이스 처리
# ============================================================

def initialize_database(db_file):
    """온습도 및 안전관리 테이블이 없으면 생성합니다."""
    with sqlite3.connect(db_file) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                noise_raw INTEGER NOT NULL,
                noise_rms_voltage REAL NOT NULL,
                noise_db REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sensor_created_at
            ON sensor_data(created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_safety_created_at
            ON safety_data(created_at)
            """
        )


def save_record(db_file, record):
    """한 번의 측정 결과를 각 업무영역 테이블에 저장합니다."""
    with sqlite3.connect(db_file) as connection:
        if math.isfinite(record.temperature) and math.isfinite(record.humidity):
            connection.execute(
                """
                INSERT INTO sensor_data (temperature, humidity, created_at)
                VALUES (?, ?, ?)
                """,
                (record.temperature, record.humidity, record.created_at),
            )

        if math.isfinite(record.noise_db):
            connection.execute(
                """
                INSERT INTO safety_data (
                    noise_raw,
                    noise_rms_voltage,
                    noise_db,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    record.noise_raw,
                    record.noise_rms_voltage,
                    record.noise_db,
                    record.created_at,
                ),
            )


# ============================================================
# 5. 수집 실행 함수
# ============================================================

def collect_sensor_data(port=None):
    """Arduino 데이터를 계속 읽어 SQLite에 저장합니다."""
    initialize_database(DB_FILE)

    while True:
        try:
            serial_port = find_arduino_port(port)
            print(f"Arduino 연결 시도: {serial_port} / {BAUD_RATE} baud")

            with open_serial_connection(serial_port) as connection:
                print("Arduino 연결 완료. 데이터 수집을 시작합니다.")

                while True:
                    raw_line = connection.readline()
                    line = raw_line.decode("utf-8", errors="replace").strip()

                    if not line:
                        continue

                    print(f"Arduino > {line}")

                    try:
                        record = parse_sensor_line(line)
                    except (ValueError, TypeError) as error:
                        print(f"데이터 해석 오류: {error}")
                        continue

                    if record is None:
                        continue

                    save_record(DB_FILE, record)
                    print(
                        "DB 저장 완료: "
                        f"온도={record.temperature:.1f}℃, "
                        f"습도={record.humidity:.1f}%, "
                        f"소음={record.noise_db:.1f}dB"
                    )

        except (serial.SerialException, OSError, RuntimeError) as error:
            if is_port_busy_error(error):
                print_port_busy_guidance(serial_port)
            else:
                print(f"연결 오류: {error}")
            print(f"{RECONNECT_DELAY_SECONDS}초 후 다시 연결합니다.")
            time.sleep(RECONNECT_DELAY_SECONDS)


def main():
    parser = argparse.ArgumentParser(
        description="Arduino 센서 데이터를 SQLite에 저장합니다."
    )
    parser.add_argument(
        "--port",
        help='사용할 시리얼 포트(예: "COM7"). 생략하면 자동 검색합니다.',
    )
    args = parser.parse_args()
    try:
        collect_sensor_data(args.port)
    except KeyboardInterrupt:
        print("\n데이터 수집을 종료했습니다.")


if __name__ == "__main__":
    main()
