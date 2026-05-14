"""
DriveGuard FastAPI — /predict, /health
모델 입력은 preprocess.py 와 동일한 전처리 포맷을 따릅니다.
  - 수치형(Temperature 등): Min-Max 스케일된 0-1 값
  - Weather_Condition: Label-Encoded 정수 (0-143)
  - 불리언 도로특성: 0 또는 1
  - 시간대(Sunrise_Sunset 등): 0=Day, 1=Night
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── 경로 설정 (api/ 내부에서도 프로젝트 루트 기준으로 탐색) ───────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT, "models", "xgboost_model.pkl")

# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DriveGuard API",
    description="교통사고 위험도 예측 서비스",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 모델 로드 (서버 시작 시 1회) ──────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
except FileNotFoundError:
    print(f"[ERROR] 모델 파일 없음: {MODEL_PATH}", file=sys.stderr)
    model = None
    explainer = None

FEATURE_NAMES = [
    "Start_Lat", "Start_Lng", "End_Lat", "End_Lng",
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Weather_Condition",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction",
    "No_Exit", "Railway", "Roundabout", "Station", "Stop",
    "Traffic_Calming", "Traffic_Signal",
    "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]

SEVERITY_LABEL = {
    1: "낮음 (Minor)",
    2: "보통 (Moderate)",
    3: "높음 (Serious)",
    4: "매우 높음 (Severe)",
}

FEATURE_KO = {
    "Start_Lat":             "출발지 위도",
    "Start_Lng":             "출발지 경도",
    "End_Lat":               "도착지 위도",
    "End_Lng":               "도착지 경도",
    "Temperature(F)":        "기온",
    "Wind_Chill(F)":         "체감온도",
    "Humidity(%)":           "습도",
    "Pressure(in)":          "기압",
    "Visibility(mi)":        "가시거리",
    "Wind_Speed(mph)":       "풍속",
    "Precipitation(in)":     "강수량",
    "Weather_Condition":     "날씨 상태",
    "Amenity":               "편의시설",
    "Bump":                  "과속방지턱",
    "Crossing":              "횡단보도",
    "Give_Way":              "양보 표지판",
    "Junction":              "교차로",
    "No_Exit":               "막힌 도로",
    "Railway":               "철도 건널목",
    "Roundabout":            "회전교차로",
    "Station":               "정류장",
    "Stop":                  "정지 표지판",
    "Traffic_Calming":       "교통 진정 구역",
    "Traffic_Signal":        "신호등",
    "Sunrise_Sunset":        "낮/밤",
    "Civil_Twilight":        "일출/일몰 시간대",
    "Nautical_Twilight":     "어두운 시간대",
    "Astronomical_Twilight": "야간 여부",
}


# ── 입력 스키마 ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    # 좌표 (원본 위경도 값)
    Start_Lat: float = Field(..., ge=-90,  le=90,  description="출발지 위도")
    Start_Lng: float = Field(..., ge=-180, le=180, description="출발지 경도")
    End_Lat:   float = Field(..., ge=-90,  le=90,  description="도착지 위도")
    End_Lng:   float = Field(..., ge=-180, le=180, description="도착지 경도")

    # 수치형 — Min-Max 스케일된 값 (0.0 ~ 1.0)
    Temperature_F: float = Field(0.5, ge=0.0, le=1.0, description="기온 (스케일값 0~1)", alias="Temperature(F)")
    Wind_Chill_F: float = Field(0.5, ge=0.0, le=1.0, description="체감온도 (스케일값 0~1)", alias="Wind_Chill(F)")
    Humidity_pct: float = Field(0.5, ge=0.0, le=1.0, description="습도 (스케일값 0~1)", alias="Humidity(%)")
    Pressure_in: float = Field(0.5, ge=0.0, le=1.0, description="기압 (스케일값 0~1)", alias="Pressure(in)")
    Visibility_mi: float = Field(0.8, ge=0.0, le=1.0, description="가시거리 (스케일값 0~1)", alias="Visibility(mi)")
    Wind_Speed_mph: float = Field(0.1, ge=0.0, le=1.0, description="풍속 (스케일값 0~1)", alias="Wind_Speed(mph)")
    Precipitation_in: float = Field(0.0, ge=0.0, le=1.0, description="강수량 (스케일값 0~1)", alias="Precipitation(in)")

    # 날씨 (Label-Encoded 정수, 0~143)
    Weather_Condition: int = Field(16, ge=0, le=200, description="날씨 코드 (LabelEncoded, 16=Fair)")

    # 도로 특성 (0 또는 1)
    Amenity: int = Field(0, ge=0, le=1)
    Bump: int = Field(0, ge=0, le=1)
    Crossing: int = Field(0, ge=0, le=1)
    Give_Way: int = Field(0, ge=0, le=1)
    Junction: int = Field(0, ge=0, le=1)
    No_Exit: int = Field(0, ge=0, le=1)
    Railway: int = Field(0, ge=0, le=1)
    Roundabout: int = Field(0, ge=0, le=1)
    Station: int = Field(0, ge=0, le=1)
    Stop: int = Field(0, ge=0, le=1)
    Traffic_Calming: int = Field(0, ge=0, le=1)
    Traffic_Signal: int = Field(0, ge=0, le=1)

    # 시간대 (0=Day, 1=Night)
    Sunrise_Sunset: int = Field(0, ge=0, le=1, description="0=낮, 1=밤")
    Civil_Twilight: int = Field(0, ge=0, le=1, description="0=낮, 1=밤")
    Nautical_Twilight: int = Field(0, ge=0, le=1, description="0=낮, 1=밤")
    Astronomical_Twilight: int = Field(0, ge=0, le=1, description="0=낮, 1=밤")

    model_config = {"populate_by_name": True}


# ── 응답 스키마 ───────────────────────────────────────────────────────────────
class ShapFeature(BaseModel):
    feature: str
    feature_ko: str
    shap_value: float
    direction: str  # "위험 증가" / "위험 감소"

class PredictResponse(BaseModel):
    severity: int
    severity_label: str
    confidence: float
    probabilities: dict
    top_features: list[ShapFeature]


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def request_to_df(req: PredictRequest) -> pd.DataFrame:
    raw = req.model_dump(by_alias=True)
    row = {
        "Start_Lat": raw["Start_Lat"],
        "Start_Lng": raw["Start_Lng"],
        "End_Lat":   raw["End_Lat"],
        "End_Lng":   raw["End_Lng"],
        "Temperature(F)": raw["Temperature(F)"],
        "Wind_Chill(F)": raw["Wind_Chill(F)"],
        "Humidity(%)": raw["Humidity(%)"],
        "Pressure(in)": raw["Pressure(in)"],
        "Visibility(mi)": raw["Visibility(mi)"],
        "Wind_Speed(mph)": raw["Wind_Speed(mph)"],
        "Precipitation(in)": raw["Precipitation(in)"],
        "Weather_Condition": raw["Weather_Condition"],
        "Amenity": raw["Amenity"],
        "Bump": raw["Bump"],
        "Crossing": raw["Crossing"],
        "Give_Way": raw["Give_Way"],
        "Junction": raw["Junction"],
        "No_Exit": raw["No_Exit"],
        "Railway": raw["Railway"],
        "Roundabout": raw["Roundabout"],
        "Station": raw["Station"],
        "Stop": raw["Stop"],
        "Traffic_Calming": raw["Traffic_Calming"],
        "Traffic_Signal": raw["Traffic_Signal"],
        "Sunrise_Sunset": raw["Sunrise_Sunset"],
        "Civil_Twilight": raw["Civil_Twilight"],
        "Nautical_Twilight": raw["Nautical_Twilight"],
        "Astronomical_Twilight": raw["Astronomical_Twilight"],
    }
    return pd.DataFrame([row], columns=FEATURE_NAMES)


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    df = request_to_df(req)

    # 예측
    pred_class = int(model.predict(df)[0])          # 0~3
    proba = model.predict_proba(df)[0]               # shape (4,)
    severity = pred_class + 1                        # 1~4

    # SHAP — 구버전: list[n_class] of (n, 24) / 신버전: (n, 24, n_class)
    shap_vals = explainer.shap_values(df)
    if isinstance(shap_vals, list):
        pred_sv = shap_vals[pred_class][0]           # (24,)
    else:
        pred_sv = shap_vals[0, :, pred_class]        # (24,)

    top5_idx = np.argsort(np.abs(pred_sv))[::-1][:5]
    top_features = [
        ShapFeature(
            feature=FEATURE_NAMES[i],
            feature_ko=FEATURE_KO.get(FEATURE_NAMES[i], FEATURE_NAMES[i]),
            shap_value=round(float(pred_sv[i]), 6),
            direction="위험 증가" if pred_sv[i] > 0 else "위험 감소",
        )
        for i in top5_idx
    ]

    return PredictResponse(
        severity=severity,
        severity_label=SEVERITY_LABEL[severity],
        confidence=round(float(proba[pred_class]), 4),
        probabilities={f"severity_{i+1}": round(float(p), 4) for i, p in enumerate(proba)},
        top_features=top_features,
    )
