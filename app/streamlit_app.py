import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

API_URL = "https://driveguard-mq2z.onrender.com/predict"

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DriveGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* 전체 배경 */
.stApp { background-color: #ffffff; }

/* 메인 컨텐츠 텍스트 — 다크 테마 흰색 글씨 덮어쓰기 */
.stMain, .block-container { color: #1a1a2e; }
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown strong, .stMarkdown em { color: #1a1a2e !important; }
.stMain .stCaption, .stMain small { color: #555 !important; }

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #1a1a2e;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #aaaacc !important;
    font-size: 0.85rem;
}
section[data-testid="stSidebar"] hr {
    border-color: #333355;
}

/* 결과 카드 */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 16px;
}

/* 위험도 뱃지 */
.severity-badge {
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
}
.severity-badge .grade { font-size: 3rem; font-weight: 900; color: white; line-height: 1.1; }
.severity-badge .label { font-size: 1.1rem; color: rgba(255,255,255,0.9); margin-top: 4px; }

/* 신뢰도 바 */
.conf-bar-wrap {
    background: #e8e8ee;
    border-radius: 8px;
    height: 12px;
    margin: 8px 0 4px;
    overflow: hidden;
}
.conf-bar-inner {
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
}

/* SHAP 방향 아이콘 */
.up   { color: #e74c3c; font-weight: 700; }
.down { color: #3498db; font-weight: 700; }

/* 헤더 */
.page-title {
    font-size: 2rem;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -0.5px;
}
.page-sub {
    color: #666;
    margin-top: -8px;
    margin-bottom: 24px;
    font-size: 0.95rem;
}

/* placeholder */
.placeholder {
    background: white;
    border-radius: 16px;
    padding: 60px;
    text-align: center;
    color: #aaa;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── 상수 ──────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {1: "#27ae60", 2: "#f39c12", 3: "#e67e22", 4: "#c0392b"}

# 미국 주별 사고 다발 지역 평균 좌표 (상위 20개 주)
STATE_COORDS = {
    "캘리포니아 (California)":        (34.05, -118.24),
    "텍사스 (Texas)":                  (32.78,  -96.80),
    "플로리다 (Florida)":              (28.54,  -81.38),
    "뉴욕 (New York)":                 (40.71,  -74.01),
    "펜실베이니아 (Pennsylvania)":     (40.00,  -75.13),
    "일리노이 (Illinois)":             (41.84,  -87.68),
    "오하이오 (Ohio)":                 (41.10,  -81.52),
    "조지아 (Georgia)":                (33.75,  -84.39),
    "노스캐롤라이나 (North Carolina)": (35.23,  -80.84),
    "미시간 (Michigan)":               (42.33,  -83.05),
    "버지니아 (Virginia)":             (38.65,  -77.29),
    "워싱턴 (Washington)":             (47.56, -122.33),
    "테네시 (Tennessee)":              (36.17,  -86.78),
    "미네소타 (Minnesota)":            (44.97,  -93.27),
    "콜로라도 (Colorado)":             (39.74, -104.99),
    "애리조나 (Arizona)":              (33.45, -112.07),
    "인디애나 (Indiana)":              (39.79,  -86.15),
    "위스콘신 (Wisconsin)":            (43.07,  -89.40),
    "미주리 (Missouri)":               (38.63,  -90.24),
    "메릴랜드 (Maryland)":             (39.30,  -76.61),
}
SEVERITY_LABELS = {1: "낮음 (Minor)", 2: "보통 (Moderate)", 3: "높음 (Serious)", 4: "매우 높음 (Severe)"}
SEVERITY_EMOJI  = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}

WEATHER_PRESETS = {
    "맑음 ☀️":  {"code": 16,  "precip": 0.00, "humidity": 0.25, "wind_add": 0.00},
    "흐림 ☁️":  {"code": 7,   "precip": 0.00, "humidity": 0.55, "wind_add": 0.05},
    "비 🌧️":   {"code": 62,  "precip": 0.18, "humidity": 0.82, "wind_add": 0.10},
    "눈 🌨️":   {"code": 70,  "precip": 0.14, "humidity": 0.75, "wind_add": 0.08},
    "안개 🌫️": {"code": 18,  "precip": 0.00, "humidity": 0.92, "wind_add": 0.02},
}

# MinMax 근사 범위 (US Accidents 데이터 기준)
SCALE = {
    "temp_f":   (-30, 200),
    "chill_f":  (-70, 200),
    "vis_mi":   (0,    10),
    "wind_mph": (0,   100),
    "precip":   (0,     5),
    "dist_mi":  (0,    50),
}

def minmax(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


# ── 사이드바 입력 ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DriveGuard")
    st.markdown("---")

    st.markdown("### 📍 지역 선택")
    state_sel = st.selectbox("미국 주 (State)", list(STATE_COORDS.keys()))
    st.markdown("---")

    st.markdown("### 🌤 날씨 / 시야")
    temp_c      = st.slider("기온 (°C)", -20, 40, 15)
    wind_mph_input = st.slider("풍속 (mph)", 0, 60, 10, step=1)
    weather_key = st.selectbox("날씨 상태", list(WEATHER_PRESETS.keys()))
    vis_km      = st.slider("가시거리 (km)", 0.0, 10.0, 8.0, step=0.5)
    time_sel    = st.radio("시간대", ["낮 ☀️", "밤 🌙"], horizontal=True)
    st.markdown("---")

    st.markdown("### 🛣 도로 환경")
    c1, c2 = st.columns(2)
    with c1:
        crossing     = st.checkbox("횡단보도")
        junction     = st.checkbox("교차로")
        roundabout   = st.checkbox("회전교차로")
        give_way     = st.checkbox("양보 표지판")
        railway      = st.checkbox("철도 건널목")
        station      = st.checkbox("정류장")
    with c2:
        traffic_sig  = st.checkbox("신호등")
        bump         = st.checkbox("과속방지턱")
        stop_sign    = st.checkbox("정지 표지판")
        no_exit      = st.checkbox("막힌 도로")
        t_calming    = st.checkbox("교통 진정 구역")
        amenity      = st.checkbox("편의시설")
    st.markdown("---")

    predict_btn = st.button("🔍 위험도 예측", use_container_width=True, type="primary")


# ── 페이로드 빌더 ─────────────────────────────────────────────────────────────
def build_payload():
    wp = WEATHER_PRESETS[weather_key]
    is_night = 1 if "밤" in time_sel else 0

    temp_f    = temp_c * 9 / 5 + 32
    wind_mph  = float(wind_mph_input)

    # Wind Chill 공식 (미국 NWS 기준, °F / mph)
    if wind_mph >= 3 and temp_f <= 50:
        chill_f = (35.74 + 0.6215 * temp_f
                   - 35.75 * wind_mph ** 0.16
                   + 0.4275 * temp_f * wind_mph ** 0.16)
    else:
        chill_f = temp_f

    temp_sc   = minmax(temp_f,  *SCALE["temp_f"])
    chill_sc  = minmax(chill_f, *SCALE["chill_f"])
    vis_sc    = minmax(vis_km * 0.621371, *SCALE["vis_mi"])
    wind_sc   = minmax(wind_mph, *SCALE["wind_mph"])
    precip_sc = minmax(wp["precip"] * 25, *SCALE["precip"])

    lat, lng = STATE_COORDS[state_sel]

    return {
        "Start_Lat": lat, "Start_Lng": lng,
        "End_Lat":   lat, "End_Lng":   lng,
        "Temperature(F)":   round(temp_sc,  4),
        "Wind_Chill(F)":    round(chill_sc, 4),
        "Humidity(%)":      round(wp["humidity"], 4),
        "Pressure(in)":     0.5,
        "Visibility(mi)":   round(vis_sc,   4),
        "Wind_Speed(mph)":  round(wind_sc,  4),
        "Precipitation(in)":round(precip_sc,4),
        "Weather_Condition": wp["code"],
        "Amenity": int(amenity), "Bump": int(bump), "Crossing": int(crossing),
        "Give_Way": int(give_way), "Junction": int(junction), "No_Exit": int(no_exit),
        "Railway": int(railway), "Roundabout": int(roundabout), "Station": int(station),
        "Stop": int(stop_sign), "Traffic_Calming": int(t_calming),
        "Traffic_Signal": int(traffic_sig),
        "Sunrise_Sunset": is_night, "Civil_Twilight": is_night,
        "Nautical_Twilight": is_night, "Astronomical_Twilight": is_night,
    }


# ── 결과 렌더러 ───────────────────────────────────────────────────────────────
def render_gauge(severity: int, confidence: float):
    # 1~4 → 0~100 (각 등급 중앙에 바늘 위치)
    needle_val = (severity - 1) * 25 + 12.5

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=needle_val,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "shape": "angular",
            "axis": {
                "range": [0, 100],
                "tickvals": [12.5, 37.5, 62.5, 87.5],
                "ticktext": ["1등급", "2등급", "3등급", "4등급"],
                "tickfont": {"size": 11, "color": "#555"},
            },
            "bar": {"color": SEVERITY_COLORS[severity], "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   25],  "color": "#d5f5e3"},
                {"range": [25,  50],  "color": "#fef9e7"},
                {"range": [50,  75],  "color": "#fde8d8"},
                {"range": [75, 100],  "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": SEVERITY_COLORS[severity], "width": 5},
                "thickness": 0.85,
                "value": needle_val,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor="white",
        font={"family": "sans-serif"},
    )
    return fig


def render_shap_chart(top_features: list):
    df = pd.DataFrame(top_features)
    df = df.sort_values("shap_value")

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["feature_ko"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="outside",
        textfont=dict(size=11, color="#111"),
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#aaa")
    fig.update_layout(
        height=280,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111", family="sans-serif"),
        xaxis=dict(
            showgrid=True, gridcolor="#f0f0f0", zeroline=False,
            title=dict(text="SHAP 값", font=dict(color="#111")),
            tickfont=dict(size=11, color="#111"),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#111"),
        ),
        showlegend=False,
    )
    return fig


def render_prob_chart(probs: dict):
    labels = ["1등급", "2등급", "3등급", "4등급"]
    values = [probs[f"severity_{i+1}"] for i in range(4)]
    colors = ["#27ae60", "#f39c12", "#e67e22", "#c0392b"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#111"),
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=10, b=0, l=0, r=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111", family="sans-serif"),
        yaxis=dict(
            range=[0, max(values) * 1.25],
            showgrid=True, gridcolor="#f0f0f0",
            tickformat=".0%",
            title=dict(text="확률", font=dict(color="#111")),
            tickfont=dict(color="#111"),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(color="#111")),
        showlegend=False,
    )
    return fig


# ── 메인 ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">🛡️ DriveGuard</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">AI 기반 교통사고 위험도 예측 시스템 — 주행 조건을 입력하고 위험도를 확인하세요.</p>', unsafe_allow_html=True)

if predict_btn:
    payload = build_payload()
    try:
        with st.spinner("AI 분석 중..."):
            resp = requests.post(API_URL, json=payload, timeout=15)

        if resp.status_code != 200:
            st.error(f"API 오류 ({resp.status_code}): {resp.text}")
            st.stop()

        data        = resp.json()
        severity    = data["severity"]
        confidence  = data["confidence"]
        probs       = data["probabilities"]
        # Distance(mi)는 자동 채움 변수이므로 SHAP 표시에서 제외
        top_feats   = [f for f in data["top_features"] if f["feature"] != "Distance(mi)"][:5]
        color       = SEVERITY_COLORS[severity]
        label       = SEVERITY_LABELS[severity]

        # ── 상단 요약 배너 ──────────────────────────────────────────────────
        st.markdown(f"""
        <div class="severity-badge" style="background: linear-gradient(135deg, {color}dd, {color});">
            <div class="grade">{SEVERITY_EMOJI[severity]} {severity}등급</div>
            <div class="label">{label} &nbsp;|&nbsp; 신뢰도 {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 두 열 레이아웃 ──────────────────────────────────────────────────
        left, right = st.columns([1, 1], gap="large")

        with left:
            # 게이지
            st.markdown("**위험도 게이지**")
            st.plotly_chart(render_gauge(severity, confidence), use_container_width=True)

            # 신뢰도 바
            st.markdown(f"""
            <div style="margin-top:4px;">
                <span style="font-size:0.85rem;color:#666;">신뢰도</span>
                <span style="float:right;font-weight:700;color:{color};">{confidence*100:.1f}%</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-inner" style="width:{confidence*100:.1f}%;background:{color};"></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # 등급별 확률
            st.markdown("**등급별 예측 확률**")
            st.plotly_chart(render_prob_chart(probs), use_container_width=True)

        with right:
            # SHAP 차트
            st.markdown("**주요 영향 요인 (SHAP Top 5)**")
            st.caption("🔴 빨강: 위험 증가 요인 &nbsp; 🔵 파랑: 위험 감소 요인")
            st.plotly_chart(render_shap_chart(top_feats), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # SHAP 테이블
            st.markdown("**요인별 상세 분석**")
            rows_html = ""
            for f in sorted(top_feats, key=lambda x: abs(x["shap_value"]), reverse=True):
                is_up = f["shap_value"] > 0
                dir_html = (
                    '<span style="color:#e74c3c;font-weight:700;">▲ 위험 증가</span>'
                    if is_up else
                    '<span style="color:#3498db;font-weight:700;">▼ 위험 감소</span>'
                )
                shap_color = "#e74c3c" if is_up else "#3498db"
                rows_html += f"""
                <tr>
                    <td style="padding:8px 12px;color:#111;border-bottom:1px solid #f0f0f0;">{f['feature_ko']}</td>
                    <td style="padding:8px 12px;color:{shap_color};font-weight:600;font-family:monospace;border-bottom:1px solid #f0f0f0;">{f['shap_value']:+.4f}</td>
                    <td style="padding:8px 12px;border-bottom:1px solid #f0f0f0;">{dir_html}</td>
                </tr>"""
            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;">
                <thead>
                    <tr style="background:#f8f9fb;">
                        <th style="padding:10px 12px;text-align:left;color:#555;font-size:0.82rem;font-weight:600;border-bottom:2px solid #e8e8ee;">요인</th>
                        <th style="padding:10px 12px;text-align:left;color:#555;font-size:0.82rem;font-weight:600;border-bottom:2px solid #e8e8ee;">SHAP 값</th>
                        <th style="padding:10px 12px;text-align:left;color:#555;font-size:0.82rem;font-weight:600;border-bottom:2px solid #e8e8ee;">방향</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

    except requests.exceptions.ConnectionError:
        st.error("FastAPI 서버에 연결할 수 없습니다. 터미널에서 다음 명령어로 서버를 먼저 실행하세요:\n\n```\nuvicorn api.main:app --host 0.0.0.0 --port 8000\n```")
    except Exception as e:
        st.error(f"오류 발생: {e}")

else:
    # 초기 화면 placeholder
    st.markdown("""
    <div class="placeholder">
        <div style="font-size:3rem;">🛡️</div>
        <div style="font-size:1.2rem;font-weight:600;margin-top:12px;color:#555;">
            주행 조건을 입력하고 위험도를 예측해보세요
        </div>
        <div style="margin-top:8px;color:#aaa;font-size:0.9rem;">
            좌측 사이드바 → 조건 입력 → <b>위험도 예측</b> 클릭
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 사용법 카드
    col1, col2, col3 = st.columns(3)
    tips = [
        ("🚗", "주행 속도", "속도와 기온을 설정하여\n주행 환경을 시뮬레이션합니다."),
        ("🌤", "날씨 / 시야", "날씨와 가시거리, 시간대로\n기상 조건을 입력합니다."),
        ("🛣", "도로 환경", "교차로, 신호등 등 도로 특성을\n체크박스로 선택합니다."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], tips):
        col.markdown(f"""
        <div class="result-card" style="text-align:center;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:700;margin:8px 0 4px;">{title}</div>
            <div style="color:#888;font-size:0.88rem;white-space:pre-line;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
