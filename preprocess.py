import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

DATA_PATH = "data/US_Accidents_March23.csv"
OUTPUT_PATH = "data/processed.csv"

DROP_COLS = [
    "ID", "Source", "Description", "Street", "City", "County",
    "State", "Zipcode", "Country", "Timezone", "Airport_Code",
    "Weather_Timestamp", "Wind_Direction", "Turning_Loop",
    "Start_Time", "End_Time",
    # 사고 후 측정 변수 제외 (좌표는 학습에 포함)
    "Distance(mi)",
]

CATEGORICAL_COLS = [
    "Weather_Condition", "Sunrise_Sunset",
    "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
]

SCALE_COLS = [
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
    "Precipitation(in)",
]

def load_data(path):
    print(f"[1/6] 데이터 로딩: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"      원본 shape: {df.shape}")
    return df

def drop_columns(df):
    print("[2/6] 불필요한 컬럼 제거")
    existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing)
    print(f"      제거 컬럼 수: {len(existing)} | 남은 shape: {df.shape}")
    return df

def fill_missing(df):
    print("[3/6] 결측치 처리")
    num_cols = df.select_dtypes(include=np.number).columns.difference(["Severity"])
    cat_cols = df.select_dtypes(include=["str", "bool"]).columns

    for col in num_cols:
        median = df[col].median()
        missing = df[col].isna().sum()
        if missing:
            df[col] = df[col].fillna(median)

    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty and df[col].isna().sum():
            df[col] = df[col].fillna(mode[0])

    remaining = df.isna().sum().sum()
    print(f"      처리 후 잔여 결측치: {remaining}")
    return df

def encode_categoricals(df):
    print("[4/6] 범주형 Label Encoding")
    le = LabelEncoder()
    existing = [c for c in CATEGORICAL_COLS if c in df.columns]

    # bool 컬럼 → int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    for col in existing:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"      {col}: {le.classes_.tolist()}")
    return df

def scale_numerics(df):
    print("[5/6] 수치형 Min-Max Scaling")
    existing = [c for c in SCALE_COLS if c in df.columns]
    scaler = MinMaxScaler()
    df[existing] = scaler.fit_transform(df[existing])
    print(f"      스케일링 컬럼: {existing}")
    return df

def plot_class_distribution(df):
    print("[6/6] 클래스 분포 출력")
    counts = df["Severity"].value_counts().sort_index()
    total = len(df)

    print("\n  Severity 분포:")
    print(f"  {'등급':<8} {'건수':>10} {'비율':>8}")
    print("  " + "-" * 30)
    for sev, cnt in counts.items():
        print(f"  Severity {sev}  {cnt:>10,}  {cnt/total*100:>7.2f}%")
    print(f"  {'합계':<8} {total:>10,}  100.00%\n")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [f"Severity {i}" for i in counts.index],
        counts.values,
        color=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, cnt in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.003,
            f"{cnt:,}\n({cnt/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("DriveGuard — Severity Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("data/class_distribution.png", dpi=150)
    print("  그래프 저장: data/class_distribution.png")
    plt.show()

def main():
    os.makedirs("data", exist_ok=True)

    df = load_data(DATA_PATH)
    df = drop_columns(df)
    df = fill_missing(df)
    df = encode_categoricals(df)
    df = scale_numerics(df)

    plot_class_distribution(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n전처리 완료 → {OUTPUT_PATH}  (shape: {df.shape})")

if __name__ == "__main__":
    main()
