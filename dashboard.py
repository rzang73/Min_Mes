import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. 프로그램 설정값
# ============================================================

DB_FILE = "sensor_data.db"
UCL_TEMP = 28.0
LCL_TEMP = 23.0
TEMP_LINE_WIDTH = 2
LIMIT_LABEL_FONT_SIZE = 15

UCL_HUMIDITY = 60.0
LCL_HUMIDITY = 30.0
HUMIDITY_LINE_WIDTH = 2
HUMIDITY_LABEL_FONT_SIZE = 18

VIOLATION_INTERVAL = "10min"

CAPABILITY_GOOD_LIMIT = 1.33
CAPABILITY_MINIMUM_LIMIT = 1.00
CAPABILITY_LINE_WIDTH = 2
CAPABILITY_LABEL_FONT_SIZE = 15

# 소음 기준은 사용 환경에 맞게 변경할 수 있는 대시보드 예시값입니다.
# 법적 작업환경측정이나 공인 소음측정값을 대신하지 않습니다.
NOISE_WARNING_DB = 80.0
NOISE_DANGER_DB = 85.0
NOISE_LINE_WIDTH = 2


# ============================================================
# 2. 데이터 처리 함수
# ============================================================

@st.cache_data(ttl=10)
def load_data(db_file):
    """SQLite에서 센서 데이터를 읽어 시간순으로 반환합니다."""
    with sqlite3.connect(db_file) as conn:
        query = """
            SELECT id, temperature, humidity, created_at
            FROM sensor_data
            ORDER BY created_at
        """
        data = pd.read_sql_query(query, conn)

    data["created_at"] = pd.to_datetime(data["created_at"])
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")
    data["humidity"] = pd.to_numeric(data["humidity"], errors="coerce")
    return data.dropna(subset=["created_at", "temperature", "humidity"])


@st.cache_data(ttl=10)
def load_safety_data(db_file):
    """SQLite에서 소음 안전관리 데이터를 읽어 시간순으로 반환합니다."""
    with sqlite3.connect(db_file) as conn:
        query = """
            SELECT id, noise_raw, noise_rms_voltage, noise_db, created_at
            FROM safety_data
            ORDER BY created_at
        """
        data = pd.read_sql_query(query, conn)

    data["created_at"] = pd.to_datetime(data["created_at"])
    numeric_columns = ["noise_raw", "noise_rms_voltage", "noise_db"]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(subset=["created_at", "noise_db"])


def select_recent_data(data, data_count):
    """최근에 측정된 지정 개수의 데이터를 반환합니다."""
    return data.tail(data_count).copy()


def add_violation_columns(data):
    """온도와 습도의 UCL 초과 및 LCL 미만 여부를 추가합니다."""
    result = data.copy()
    result["temp_ucl_violation"] = result["temperature"] > UCL_TEMP
    result["temp_lcl_violation"] = result["temperature"] < LCL_TEMP
    result["temp_limit_violation"] = (
        result["temp_ucl_violation"] | result["temp_lcl_violation"]
    )
    result["humidity_ucl_violation"] = result["humidity"] > UCL_HUMIDITY
    result["humidity_lcl_violation"] = result["humidity"] < LCL_HUMIDITY
    result["humidity_limit_violation"] = (
        result["humidity_ucl_violation"] | result["humidity_lcl_violation"]
    )
    return result


def calculate_10min_violation_counts(data, violation_column):
    """선택된 이탈 판정 열을 10분 단위로 묶어 건수를 계산합니다."""
    indexed = data.set_index("created_at").sort_index()
    counts = (
        indexed[violation_column]
        .resample(VIOLATION_INTERVAL)
        .sum()
        .astype(int)
    )
    result = counts.rename("violation_count").reset_index()
    result["time_label"] = result["created_at"].dt.strftime("%m-%d %H:%M")
    return result


def calculate_sensor_statistics(data):
    """KPI 표시에 필요한 센서 통계를 계산합니다."""
    return {
        "avg_temp": data["temperature"].mean(),
        "max_temp": data["temperature"].max(),
        "min_temp": data["temperature"].min(),
        "std_temp": data["temperature"].std(),
        "avg_humi": data["humidity"].mean(),
        "max_humi": data["humidity"].max(),
        "min_humi": data["humidity"].min(),
        "std_humi": data["humidity"].std(),
    }


def calculate_process_capability(data, value_column, lower_limit, upper_limit):
    """선택된 데이터의 평균, 표준편차, Cp 및 Cpk를 계산합니다."""
    values = data[value_column].dropna()
    sample_count = len(values)

    if sample_count == 0:
        return {
            "sample_count": 0,
            "mean": None,
            "std": None,
            "cp": None,
            "cpk": None,
        }

    mean_value = values.mean()
    std_value = values.std(ddof=1)

    # Cp/Cpk는 표본이 2개 이상이고 표준편차가 0보다 클 때 계산합니다.
    if sample_count < 2 or pd.isna(std_value) or std_value <= 0:
        cp_value = None
        cpk_value = None
    else:
        cp_value = (upper_limit - lower_limit) / (6 * std_value)
        cpu_value = (upper_limit - mean_value) / (3 * std_value)
        cpl_value = (mean_value - lower_limit) / (3 * std_value)
        cpk_value = min(cpu_value, cpl_value)

    return {
        "sample_count": sample_count,
        "mean": mean_value,
        "std": std_value,
        "cp": cp_value,
        "cpk": cpk_value,
    }


def format_capability_value(value):
    """Cp/Cpk 계산 불가 상태를 안전하게 표시합니다."""
    if value is None or pd.isna(value):
        return "계산 불가"
    return f"{value:.3f}"


def get_capability_status(cpk_value):
    """Cpk 값을 기준으로 공정능력 상태와 표시 색상을 반환합니다."""
    if cpk_value is None or pd.isna(cpk_value):
        return "표본 수 또는 산포가 부족하여 판정할 수 없습니다.", "info"
    if cpk_value >= 1.33:
        return "공정능력 양호 (Cpk ≥ 1.33)", "success"
    if cpk_value >= 1.00:
        return "공정능력 주의 (1.00 ≤ Cpk < 1.33)", "warning"
    return "공정능력 부족 (Cpk < 1.00)", "error"


def calculate_10min_process_capability(
    data,
    value_column,
    lower_limit,
    upper_limit,
):
    """10분 구간별 Cp와 Cpk를 계산하여 시간순으로 반환합니다."""
    indexed_values = (
        data.set_index("created_at")[value_column]
        .sort_index()
        .resample(VIOLATION_INTERVAL)
    )

    capability_records = []

    for interval_start, values in indexed_values:
        interval_data = values.dropna().to_frame(name=value_column)
        capability = calculate_process_capability(
            data=interval_data,
            value_column=value_column,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

        capability_records.append(
            {
                "created_at": interval_start,
                "sample_count": capability["sample_count"],
                "cp": capability["cp"],
                "cpk": capability["cpk"],
            }
        )

    return pd.DataFrame(capability_records)


# ============================================================
# 3. 그래프 생성 함수
# ============================================================

def create_temperature_chart(data):
    """온도 측정선과 UCL/LCL 기준선을 표시합니다."""
    figure = px.line(
        data,
        x="created_at",
        y="temperature",
        markers=True,
        labels={"created_at": "시간", "temperature": "온도 (°C)"},
    )
    figure.update_traces(line=dict(width=TEMP_LINE_WIDTH))

    x_start = data["created_at"].iloc[0]
    x_end = data["created_at"].iloc[-1]

    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[UCL_TEMP, UCL_TEMP],
            mode="lines",
            name=f"UCL {UCL_TEMP:.0f} °C",
            line=dict(color="red", width=TEMP_LINE_WIDTH, dash="dash"),
            hovertemplate="UCL: %{y:.1f} °C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[LCL_TEMP, LCL_TEMP],
            mode="lines",
            name=f"LCL {LCL_TEMP:.0f} °C",
            line=dict(color="green", width=TEMP_LINE_WIDTH, dash="dash"),
            hovertemplate="LCL: %{y:.1f} °C<extra></extra>",
        )
    )

    # UCL 바로 위와 LCL 바로 아래에 15px 크기의 라벨을 표시합니다.
    figure.add_annotation(
        x=x_end,
        y=UCL_TEMP,
        text="<b>UCL</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=5,
        font=dict(color="red", size=LIMIT_LABEL_FONT_SIZE),
        bgcolor="rgba(255,255,255,0.75)",
    )
    figure.add_annotation(
        x=x_end,
        y=LCL_TEMP,
        text="<b>LCL</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        yshift=-5,
        font=dict(color="green", size=LIMIT_LABEL_FONT_SIZE),
        bgcolor="rgba(255,255,255,0.75)",
    )

    y_min = min(data["temperature"].min(), LCL_TEMP) - 0.5
    y_max = max(data["temperature"].max(), UCL_TEMP) + 0.5
    figure.update_layout(
        yaxis_range=[y_min, y_max],
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return figure


def create_violation_bar_chart(violation_counts, y_axis_title):
    """10분 구간별 UCL/LCL 이탈 건수를 막대그래프로 표시합니다."""
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=violation_counts["time_label"],
            y=violation_counts["violation_count"],
            text=violation_counts["violation_count"],
            textposition="outside",
            textfont=dict(size=14, color="black"),
            cliponaxis=False,
            marker=dict(
                color=violation_counts["violation_count"],
                colorscale=[[0.0, "#B7E4C7"], [1.0, "#D00000"]],
                line=dict(color="#555555", width=1),
            ),
            hovertemplate=(
                "10분 시작: %{x}<br>이탈 건수: %{y}건<extra></extra>"
            ),
        )
    )
    highest_count = int(violation_counts["violation_count"].max())
    figure.update_layout(
        xaxis_title="10분 구간 시작 시각",
        yaxis_title=y_axis_title,
        yaxis=dict(
            rangemode="tozero",
            dtick=1,
            range=[0, max(1, highest_count * 1.25 + 0.5)],
        ),
        showlegend=False,
        margin=dict(t=40),
        height=360,
    )
    return figure


def create_humidity_chart(data):
    """습도 측정선과 UCL/LCL 기준선을 표시합니다."""
    figure = px.line(
        data,
        x="created_at",
        y="humidity",
        markers=True,
        labels={"created_at": "시간", "humidity": "습도 (%)"},
    )
    figure.update_traces(line=dict(width=HUMIDITY_LINE_WIDTH))

    x_start = data["created_at"].iloc[0]
    x_end = data["created_at"].iloc[-1]

    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[UCL_HUMIDITY, UCL_HUMIDITY],
            mode="lines",
            name=f"UCL {UCL_HUMIDITY:.0f} %",
            line=dict(
                color="red",
                width=HUMIDITY_LINE_WIDTH,
                dash="dash",
            ),
            hovertemplate="UCL: %{y:.1f} %<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[LCL_HUMIDITY, LCL_HUMIDITY],
            mode="lines",
            name=f"LCL {LCL_HUMIDITY:.0f} %",
            line=dict(
                color="green",
                width=HUMIDITY_LINE_WIDTH,
                dash="dash",
            ),
            hovertemplate="LCL: %{y:.1f} %<extra></extra>",
        )
    )

    # UCL 바로 위와 LCL 바로 아래에 18px 굵은 라벨을 표시합니다.
    figure.add_annotation(
        x=x_end,
        y=UCL_HUMIDITY,
        text="<b>UCL</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=5,
        font=dict(color="red", size=HUMIDITY_LABEL_FONT_SIZE),
        bgcolor="rgba(255,255,255,0.75)",
    )
    figure.add_annotation(
        x=x_end,
        y=LCL_HUMIDITY,
        text="<b>LCL</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        yshift=-5,
        font=dict(color="green", size=HUMIDITY_LABEL_FONT_SIZE),
        bgcolor="rgba(255,255,255,0.75)",
    )

    y_min = min(data["humidity"].min(), LCL_HUMIDITY) - 3
    y_max = max(data["humidity"].max(), UCL_HUMIDITY) + 3
    figure.update_layout(
        yaxis_range=[y_min, y_max],
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return figure


def create_capability_trend_chart(capability_data):
    """10분 단위 Cp/Cpk 변화와 권장 관리기준선을 표시합니다."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=capability_data["created_at"],
            y=capability_data["cp"],
            customdata=capability_data["sample_count"],
            mode="lines+markers",
            name="Cp",
            line=dict(color="#0068C9", width=CAPABILITY_LINE_WIDTH),
            marker=dict(size=6),
            connectgaps=False,
            hovertemplate=(
                "시간: %{x|%m-%d %H:%M}<br>"
                "Cp: %{y:.3f}<br>"
                "표본: %{customdata}건<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=capability_data["created_at"],
            y=capability_data["cpk"],
            customdata=capability_data["sample_count"],
            mode="lines+markers",
            name="Cpk",
            line=dict(color="#FF8C00", width=CAPABILITY_LINE_WIDTH),
            marker=dict(size=6),
            connectgaps=False,
            hovertemplate=(
                "시간: %{x|%m-%d %H:%M}<br>"
                "Cpk: %{y:.3f}<br>"
                "표본: %{customdata}건<extra></extra>"
            ),
        )
    )

    x_start = capability_data["created_at"].iloc[0]
    x_end = capability_data["created_at"].iloc[-1]

    # 한 개의 10분 구간만 있어도 기준선이 보이도록 길이를 확보합니다.
    if x_start == x_end:
        x_end = x_start + pd.Timedelta(VIOLATION_INTERVAL)

    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[CAPABILITY_GOOD_LIMIT, CAPABILITY_GOOD_LIMIT],
            mode="lines",
            name=f"양호 기준 {CAPABILITY_GOOD_LIMIT:.2f}",
            line=dict(
                color="green",
                width=CAPABILITY_LINE_WIDTH,
                dash="dash",
            ),
            hovertemplate=(
                f"양호 기준: {CAPABILITY_GOOD_LIMIT:.2f}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[CAPABILITY_MINIMUM_LIMIT, CAPABILITY_MINIMUM_LIMIT],
            mode="lines",
            name=f"최소 기준 {CAPABILITY_MINIMUM_LIMIT:.2f}",
            line=dict(
                color="red",
                width=CAPABILITY_LINE_WIDTH,
                dash="dash",
            ),
            hovertemplate=(
                f"최소 기준: {CAPABILITY_MINIMUM_LIMIT:.2f}<extra></extra>"
            ),
        )
    )

    figure.add_annotation(
        x=x_end,
        y=CAPABILITY_GOOD_LIMIT,
        text=f"<b>양호 {CAPABILITY_GOOD_LIMIT:.2f}</b>",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=5,
        font=dict(color="green", size=CAPABILITY_LABEL_FONT_SIZE),
        bgcolor="rgba(255,255,255,0.80)",
    )

    figure.add_annotation(
        x=x_end,
        y=CAPABILITY_MINIMUM_LIMIT,
        text=f"<b>최소 {CAPABILITY_MINIMUM_LIMIT:.2f}</b>",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        yshift=-5,
        font=dict(color="red", size=CAPABILITY_LABEL_FONT_SIZE),
        bgcolor="rgba(255,255,255,0.80)",
    )

    valid_values = pd.concat(
        [capability_data["cp"], capability_data["cpk"]]
    ).dropna()

    if valid_values.empty:
        y_min = 0
        y_max = CAPABILITY_GOOD_LIMIT + 0.5
    else:
        y_min = min(0, valid_values.min(), CAPABILITY_MINIMUM_LIMIT) - 0.1
        y_max = max(valid_values.max(), CAPABILITY_GOOD_LIMIT) + 0.3

    figure.update_layout(
        xaxis_title="10분 구간 시작 시각",
        yaxis_title="공정능력지수",
        yaxis_range=[y_min, y_max],
        height=430,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return figure


def create_noise_chart(data):
    """소음 변화와 주의·위험 기준선을 표시합니다."""
    figure = px.line(
        data,
        x="created_at",
        y="noise_db",
        markers=True,
        labels={"created_at": "시간", "noise_db": "추정 소음 (dB)"},
    )
    figure.update_traces(
        line=dict(color="#0068C9", width=NOISE_LINE_WIDTH),
        marker=dict(size=6),
    )

    x_start = data["created_at"].iloc[0]
    x_end = data["created_at"].iloc[-1]

    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[NOISE_WARNING_DB, NOISE_WARNING_DB],
            mode="lines",
            name=f"주의 {NOISE_WARNING_DB:.0f} dB",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            hovertemplate=(
                f"주의 기준: {NOISE_WARNING_DB:.0f} dB<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[x_start, x_end],
            y=[NOISE_DANGER_DB, NOISE_DANGER_DB],
            mode="lines",
            name=f"위험 {NOISE_DANGER_DB:.0f} dB",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate=(
                f"위험 기준: {NOISE_DANGER_DB:.0f} dB<extra></extra>"
            ),
        )
    )

    y_min = min(data["noise_db"].min(), NOISE_WARNING_DB) - 5
    y_max = max(data["noise_db"].max(), NOISE_DANGER_DB) + 5
    figure.update_layout(
        yaxis_range=[max(0, y_min), y_max],
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return figure


def calculate_10min_noise_counts(data):
    """10분 단위 주의·위험 소음 발생 건수를 계산합니다."""
    indexed = data.set_index("created_at").sort_index()
    warning_mask = indexed["noise_db"] >= NOISE_WARNING_DB
    danger_mask = indexed["noise_db"] >= NOISE_DANGER_DB

    warning_counts = warning_mask.resample(VIOLATION_INTERVAL).sum().astype(int)
    danger_counts = danger_mask.resample(VIOLATION_INTERVAL).sum().astype(int)

    result = pd.DataFrame(
        {
            "warning_count": warning_counts,
            "danger_count": danger_counts,
        }
    ).reset_index()
    result["time_label"] = result["created_at"].dt.strftime("%m-%d %H:%M")
    return result


def create_noise_count_chart(noise_counts):
    """10분 단위 소음 주의·위험 건수를 막대그래프로 표시합니다."""
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=noise_counts["time_label"],
            y=noise_counts["warning_count"],
            name=f"{NOISE_WARNING_DB:.0f} dB 이상",
            marker_color="#F59E0B",
            text=noise_counts["warning_count"],
            textposition="outside",
        )
    )
    figure.add_trace(
        go.Bar(
            x=noise_counts["time_label"],
            y=noise_counts["danger_count"],
            name=f"{NOISE_DANGER_DB:.0f} dB 이상",
            marker_color="#D00000",
            text=noise_counts["danger_count"],
            textposition="outside",
        )
    )
    figure.update_layout(
        barmode="group",
        xaxis_title="10분 구간 시작 시각",
        yaxis_title="발생 건수",
        yaxis=dict(dtick=1, rangemode="tozero"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return figure


def create_temperature_humidity_scatter(data):
    """온도와 습도의 관계를 나타내는 산점도를 생성합니다."""
    return px.scatter(
        data,
        x="temperature",
        y="humidity",
        hover_data=["created_at"],
        labels={"temperature": "온도 (°C)", "humidity": "습도 (%)"},
    )


# ============================================================
# 4. 화면 출력 함수
# ============================================================

def render_navigation():
    """회색 사이드바에 대시보드 관리 메뉴를 표시합니다."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #E5E7EB;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #1F2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("⚙️ Dashboard 설정")
    return st.sidebar.radio(
        "관리 메뉴",
        [
            "1) 온도·습도 관리",
            "2) 안전관리(데시벨 외)",
            "3) 추가예정",
        ],
        index=0,
    )

def render_sidebar(total_count):
    """사이드바를 출력하고 화면에 표시할 데이터 수를 반환합니다."""
    st.sidebar.header("Dashboard 설정")
    data_count = st.sidebar.slider(
        "표시할 데이터 개수",
        min_value=10,
        max_value=1000,
        value=1000,
        step=10,
    )
    st.sidebar.caption(f"현재 저장된 데이터: {total_count:,}건")
    return data_count


def render_current_status(data):
    """현재 센서값과 직전 측정값 대비 변화를 표시합니다."""
    latest = data.iloc[-1]
    if len(data) >= 2:
        previous = data.iloc[-2]
        temp_delta = latest["temperature"] - previous["temperature"]
        humi_delta = latest["humidity"] - previous["humidity"]
    else:
        temp_delta = 0
        humi_delta = 0

    st.subheader("현재 센서 상태")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 온도", f"{latest['temperature']:.1f} °C", f"{temp_delta:+.1f} °C")
    col2.metric("현재 습도", f"{latest['humidity']:.1f} %", f"{humi_delta:+.1f} %")
    col3.metric("수집 데이터", f"{len(data):,} 건")
    col4.metric("최근 측정 시간", latest["created_at"].strftime("%H:%M:%S"))


def render_sensor_statistics(data):
    """온도와 습도의 요약 통계를 표시합니다."""
    statistics = calculate_sensor_statistics(data)
    st.subheader("📊 센서 통계")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균 온도", f"{statistics['avg_temp']:.1f} °C")
    col2.metric("최고 온도", f"{statistics['max_temp']:.1f} °C")
    col3.metric("평균 습도", f"{statistics['avg_humi']:.1f} %")
    col4.metric("최고 습도", f"{statistics['max_humi']:.1f} %")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("최저 온도", f"{statistics['min_temp']:.1f} °C")
    col2.metric("온도 표준편차", f"{statistics['std_temp']:.2f}")
    col3.metric("최저 습도", f"{statistics['min_humi']:.1f} %")
    col4.metric("습도 표준편차", f"{statistics['std_humi']:.2f}")


def render_violation_panel(
    data,
    title,
    caption,
    limit_column,
    ucl_column,
    lcl_column,
    y_axis_title,
):
    """온도 또는 습도의 이탈 현황판을 공통 형식으로 표시합니다."""
    total_count = int(data[limit_column].sum())
    ucl_count = int(data[ucl_column].sum())
    lcl_count = int(data[lcl_column].sum())
    normal_count = len(data) - total_count
    st.subheader(title)
    st.caption(caption)
    col1, col2 = st.columns(2)
    col1.metric("전체 이탈", f"{total_count:,} 건")
    col2.metric("UCL 초과", f"{ucl_count:,} 건")
    col1, col2 = st.columns(2)
    col1.metric("LCL 미만", f"{lcl_count:,} 건")
    col2.metric("정상 범위", f"{normal_count:,} 건")
    violation_counts = calculate_10min_violation_counts(data, limit_column)
    st.plotly_chart(
        create_violation_bar_chart(violation_counts, y_axis_title),
        use_container_width=True,
    )


def render_temperature_monitoring_section(data):
    """온도 그래프와 온도 이탈 현황판을 좌우로 표시합니다."""
    chart_column, status_column = st.columns(2, gap="large")
    with chart_column:
        st.subheader("🌡️ 온도 변화")
        st.plotly_chart(create_temperature_chart(data), use_container_width=True)
    with status_column:
        render_violation_panel(
            data=data,
            title="🚨 온도 관리기준 이탈 현황",
            caption=(
                f"판정 기준: UCL {UCL_TEMP:.0f}℃ 초과 또는 "
                f"LCL {LCL_TEMP:.0f}℃ 미만"
            ),
            limit_column="temp_limit_violation",
            ucl_column="temp_ucl_violation",
            lcl_column="temp_lcl_violation",
            y_axis_title="온도 이탈 건수",
        )


def render_humidity_monitoring_section(data):
    """습도 그래프와 습도 이탈 현황판을 좌우로 표시합니다."""
    chart_column, status_column = st.columns(2, gap="large")
    with chart_column:
        st.subheader("💧 습도 변화")
        st.plotly_chart(create_humidity_chart(data), use_container_width=True)
    with status_column:
        render_violation_panel(
            data=data,
            title="🚨 습도 관리기준 이탈 현황",
            caption=(
                f"판정 기준: UCL {UCL_HUMIDITY:.0f}% 초과 또는 "
                f"LCL {LCL_HUMIDITY:.0f}% 미만"
            ),
            limit_column="humidity_limit_violation",
            ucl_column="humidity_ucl_violation",
            lcl_column="humidity_lcl_violation",
            y_axis_title="습도 이탈 건수",
        )


def render_upper_limit_alerts(data, ucl_column, value_column, unit):
    """최근 상한 초과 발생 시각과 측정값을 크고 선명하게 표시합니다."""
    exceeded_data = (
        data.loc[data[ucl_column], ["created_at", value_column]]
        .sort_values("created_at", ascending=False)
    )

    st.markdown("#### 🔴 상한 초과 발생 시각")

    if exceeded_data.empty:
        st.success("현재 선택된 데이터에는 상한 초과가 없습니다.")
        return

    recent_exceeded = exceeded_data.head(5)
    alert_lines = []

    for _, row in recent_exceeded.iterrows():
        measured_time = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        measured_value = row[value_column]
        alert_lines.append(
            f'<div style="font-size:22px; font-weight:800; color:#D00000; '
            f'line-height:1.55;">{measured_time} &nbsp; '
            f'({measured_value:.1f}{unit})</div>'
        )

    st.markdown("".join(alert_lines), unsafe_allow_html=True)

    if len(exceeded_data) > 5:
        st.caption(
            f"상한 초과 총 {len(exceeded_data):,}건 중 최근 5건을 표시합니다."
        )


def render_process_capability_panel(
    data,
    title,
    value_column,
    lower_limit,
    upper_limit,
    unit,
    ucl_column,
):
    """온도 또는 습도의 Cp/Cpk와 상한 초과 시각을 표시합니다."""
    capability = calculate_process_capability(
        data=data,
        value_column=value_column,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )

    st.subheader(title)
    st.caption(
        f"계산 규격: LSL {lower_limit:.0f}{unit} / "
        f"USL {upper_limit:.0f}{unit} · 표본 {capability['sample_count']:,}건"
    )

    col1, col2 = st.columns(2)
    col1.metric("Cp", format_capability_value(capability["cp"]))
    col2.metric("Cpk", format_capability_value(capability["cpk"]))

    mean_text = (
        "계산 불가"
        if capability["mean"] is None
        else f"{capability['mean']:.2f}{unit}"
    )
    std_text = (
        "계산 불가"
        if capability["std"] is None or pd.isna(capability["std"])
        else f"{capability['std']:.3f}"
    )

    col1, col2 = st.columns(2)
    col1.metric("평균", mean_text)
    col2.metric("표준편차", std_text)

    status_text, status_type = get_capability_status(capability["cpk"])
    getattr(st, status_type)(status_text)

    render_upper_limit_alerts(
        data=data,
        ucl_column=ucl_column,
        value_column=value_column,
        unit=unit,
    )

    st.markdown("#### 📈 10분 단위 Cp·Cpk 변화")
    capability_trend = calculate_10min_process_capability(
        data=data,
        value_column=value_column,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    st.plotly_chart(
        create_capability_trend_chart(capability_trend),
        use_container_width=True,
    )


def render_process_capability_section(data):
    """온도와 습도의 공정능력 현황을 좌우 50%로 표시합니다."""
    st.subheader("📐 공정능력 관리")
    st.caption(
        "현재 선택된 데이터와 설정된 관리기준을 규격한계로 사용하여 "
        "Cp와 Cpk를 계산합니다. 10분 추세의 양호 기준은 1.33, "
        "최소 기준은 1.00입니다."
    )

    temperature_column, humidity_column = st.columns(2, gap="large")

    with temperature_column:
        render_process_capability_panel(
            data=data,
            title="🌡️ 온도 공정능력",
            value_column="temperature",
            lower_limit=LCL_TEMP,
            upper_limit=UCL_TEMP,
            unit="℃",
            ucl_column="temp_ucl_violation",
        )

    with humidity_column:
        render_process_capability_panel(
            data=data,
            title="💧 습도 공정능력",
            value_column="humidity",
            lower_limit=LCL_HUMIDITY,
            upper_limit=UCL_HUMIDITY,
            unit="%",
            ucl_column="humidity_ucl_violation",
        )


def render_comparison_section(data):
    st.subheader("📈 온도 / 습도 변화 비교")
    chart_data = data.set_index("created_at")[["temperature", "humidity"]]
    st.line_chart(chart_data)


def render_relationship_section(data):
    st.subheader("🔬 온도와 습도의 관계")
    st.plotly_chart(
        create_temperature_humidity_scatter(data),
        use_container_width=True,
    )


def render_distribution_section(data):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("온도 분포")
        figure = px.histogram(
            data,
            x="temperature",
            nbins=20,
            labels={"temperature": "온도 (°C)"},
        )
        st.plotly_chart(figure, use_container_width=True)

    with col2:
        st.subheader("습도 분포")
        figure = px.histogram(
            data,
            x="humidity",
            nbins=20,
            labels={"humidity": "습도 (%)"},
        )
        st.plotly_chart(figure, use_container_width=True)


def render_data_table(data):
    """센서 상세 데이터와 CSV 다운로드 버튼을 표시합니다."""
    st.divider()
    st.subheader("📋 센서 데이터")
    display_data = data.sort_values("created_at", ascending=False).copy()
    display_data["온도 판정"] = display_data["temp_limit_violation"].map(
        {True: "이탈", False: "정상"}
    )
    display_data["습도 판정"] = display_data["humidity_limit_violation"].map(
        {True: "이탈", False: "정상"}
    )
    table_columns = [
        "id", "temperature", "humidity", "created_at", "온도 판정", "습도 판정"
    ]
    st.dataframe(
        display_data[table_columns],
        use_container_width=True,
        hide_index=True,
    )
    csv = display_data[table_columns].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSV 다운로드",
        data=csv,
        file_name="sensor_data.csv",
        mime="text/csv",
    )


def render_noise_alert(data):
    """최근 위험 소음 발생 시각을 큰 적색 글씨로 표시합니다."""
    danger_data = data[data["noise_db"] >= NOISE_DANGER_DB].sort_values(
        "created_at", ascending=False
    )

    st.subheader("🔴 위험 소음 발생 시각")

    if danger_data.empty:
        st.success("선택된 데이터에는 위험 기준 이상 소음이 없습니다.")
        return

    alert_lines = []
    for _, row in danger_data.head(5).iterrows():
        measured_time = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        alert_lines.append(
            f'<div style="font-size:22px; font-weight:800; color:#D00000; '
            f'line-height:1.55;">{measured_time} &nbsp; '
            f'({row["noise_db"]:.1f} dB)</div>'
        )

    st.markdown("".join(alert_lines), unsafe_allow_html=True)

    if len(danger_data) > 5:
        st.caption(
            f"위험 소음 총 {len(danger_data):,}건 중 최근 5건을 표시합니다."
        )


def render_safety_management_page(data):
    """소음 데이터를 이용한 안전관리 대시보드를 표시합니다."""
    st.header("🦺 안전관리 · 소음 모니터링")
    st.warning(
        "표시되는 dB는 아날로그 소리센서의 보정값을 이용한 추정치입니다. "
        "법적 작업환경측정이나 공인 소음계 결과를 대신하지 않습니다."
    )

    data_count = render_sidebar(len(data))
    view_data = select_recent_data(data, data_count)
    latest = view_data.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 소음", f"{latest['noise_db']:.1f} dB")
    col2.metric("평균 소음", f"{view_data['noise_db'].mean():.1f} dB")
    col3.metric("최고 소음", f"{view_data['noise_db'].max():.1f} dB")
    col4.metric("최저 소음", f"{view_data['noise_db'].min():.1f} dB")

    chart_column, status_column = st.columns(2, gap="large")

    with chart_column:
        st.subheader("🔊 소음 변화")
        st.plotly_chart(
            create_noise_chart(view_data),
            use_container_width=True,
        )

    with status_column:
        warning_count = int((view_data["noise_db"] >= NOISE_WARNING_DB).sum())
        danger_count = int((view_data["noise_db"] >= NOISE_DANGER_DB).sum())
        normal_count = int((view_data["noise_db"] < NOISE_WARNING_DB).sum())

        st.subheader("🚨 소음 안전기준 현황")
        st.caption(
            f"대시보드 예시 기준: 주의 {NOISE_WARNING_DB:.0f} dB / "
            f"위험 {NOISE_DANGER_DB:.0f} dB"
        )
        col1, col2 = st.columns(2)
        col1.metric("주의 이상", f"{warning_count:,} 건")
        col2.metric("위험 이상", f"{danger_count:,} 건")
        col1, col2 = st.columns(2)
        col1.metric("정상", f"{normal_count:,} 건")
        col2.metric("표준편차", f"{view_data['noise_db'].std():.2f}")
        render_noise_alert(view_data)

    st.subheader("📊 10분 단위 소음 발생 현황")
    st.plotly_chart(
        create_noise_count_chart(calculate_10min_noise_counts(view_data)),
        use_container_width=True,
    )

    st.subheader("📋 소음 측정 데이터")
    display_data = view_data.sort_values("created_at", ascending=False).copy()
    display_data["상태"] = "정상"
    display_data.loc[
        display_data["noise_db"] >= NOISE_WARNING_DB, "상태"
    ] = "주의"
    display_data.loc[
        display_data["noise_db"] >= NOISE_DANGER_DB, "상태"
    ] = "위험"

    st.dataframe(display_data, use_container_width=True, hide_index=True)
    csv = display_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="소음 데이터 CSV 다운로드",
        data=csv,
        file_name="safety_noise_data.csv",
        mime="text/csv",
    )


def render_planned_page():
    """향후 센서를 확장할 메뉴의 안내 화면입니다."""
    st.header("🧩 추가예정")
    st.info(
        "가스, 진동, 조도, 화재감지 등 새로운 안전·공정 센서를 "
        "추가할 수 있도록 준비된 메뉴입니다."
    )


def render_process_management_page(data):
    """기존 온도·습도 공정관리 화면을 표시합니다."""
    st.header("🏭 온도·습도 공정관리")
    data_count = render_sidebar(len(data))
    view_data = select_recent_data(data, data_count)
    view_data = add_violation_columns(view_data)

    render_current_status(data)
    st.divider()
    render_sensor_statistics(view_data)
    st.divider()
    render_temperature_monitoring_section(view_data)
    st.divider()
    render_humidity_monitoring_section(view_data)
    st.divider()
    render_process_capability_section(view_data)
    render_comparison_section(view_data)
    render_relationship_section(view_data)
    render_distribution_section(view_data)
    render_data_table(view_data)


# ============================================================
# 5. 메인 실행 함수
# ============================================================

def main():
    """대시보드 프로그램의 전체 실행 순서를 관리합니다."""
    st.set_page_config(
        page_title="Temperature & Humidity Dashboard",
        page_icon="🌡️",
        layout="wide",
    )
    st.title("🏭 Smart Factory Sensor Dashboard")
    st.caption("Arduino Sensor Monitoring · Process & Safety")

    selected_menu = render_navigation()

    if selected_menu == "1) 온도·습도 관리":
        try:
            process_data = load_data(DB_FILE)
        except (sqlite3.Error, pd.errors.DatabaseError) as error:
            st.error(f"온도·습도 데이터를 읽는 중 오류가 발생했습니다: {error}")
            st.stop()

        if process_data.empty:
            st.warning("수집된 온도·습도 데이터가 없습니다.")
            st.stop()

        render_process_management_page(process_data)

    elif selected_menu == "2) 안전관리(데시벨 외)":
        try:
            safety_data = load_safety_data(DB_FILE)
        except (sqlite3.Error, pd.errors.DatabaseError) as error:
            st.warning(
                "소음 데이터 테이블이 아직 준비되지 않았습니다. "
                "serial_collector.py를 먼저 실행해 주세요."
            )
            st.caption(f"데이터베이스 메시지: {error}")
            st.stop()

        if safety_data.empty:
            st.warning("수집된 소음 데이터가 없습니다.")
            st.stop()

        render_safety_management_page(safety_data)

    else:
        render_planned_page()


if __name__ == "__main__":
    main()
