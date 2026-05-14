import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

matplotlib.rcParams["font.family"] = "DejaVu Sans"

DATA_PATH = "data/processed.csv"
MODEL_PATH = "models/xgboost_model.pkl"
SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

CLASS_LABELS = ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]


# ── 데이터 로드 & 샘플링 ──────────────────────────────────────────────────────

def load_sample(path, n):
    print(f"[1/5] 데이터 로드 & 클래스별 샘플링")
    df = pd.read_csv(path, low_memory=False)
    print(f"      전체 행 수: {len(df):,}")

    # Severity 1: 전체 사용 / 나머지: 각 250,000건
    per_class = {1: None, 2: 250_000, 3: 250_000, 4: 250_000}
    chunks = []
    for sev, g in df.groupby("Severity"):
        limit = per_class.get(sev)
        chunks.append(g if limit is None else g.sample(min(len(g), limit), random_state=RANDOM_STATE))
    df = pd.concat(chunks).reset_index(drop=True)
    print(f"      샘플 후 행 수: {len(df):,}")
    print(f"      클래스 분포:\n{df['Severity'].value_counts().sort_index().to_string()}\n")
    return df


# ── 학습/테스트 분리 ─────────────────────────────────────────────────────────

def split(df):
    print("[2/5] 학습/테스트 분리 (8:2)")
    # Severity 1~4 → 0~3 (XGBoost 0-indexed 요구)
    X = df.drop(columns=["Severity"])
    y = df["Severity"] - 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      학습: {len(X_tr):,}  테스트: {len(X_te):,}\n")
    return X_tr, X_te, y_tr, y_te


# ── 모델 정의 ────────────────────────────────────────────────────────────────

def build_models():
    xgb = XGBClassifier(
        objective="multi:softmax",
        num_class=4,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        random_state=RANDOM_STATE,
    )
    return {"XGBoost": xgb, "RandomForest": rf, "LogisticRegression": lr}


# ── 학습 & 평가 ──────────────────────────────────────────────────────────────

def train_evaluate(models, X_tr, X_te, y_tr, y_te):
    print("[3/5] 모델 학습 & 평가")
    results = {}
    sample_weights = compute_sample_weight("balanced", y_tr)

    for name, model in models.items():
        t0 = time.time()
        if name == "XGBoost":
            model.fit(X_tr, y_tr, sample_weight=sample_weights)
        else:
            model.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average="macro")
        cm = confusion_matrix(y_te, y_pred)

        results[name] = {"model": model, "acc": acc, "f1": f1, "cm": cm}
        print(f"  {name:<20}  Accuracy: {acc:.4f}  F1-macro: {f1:.4f}  ({elapsed:.1f}s)")

    print()
    return results


# ── Confusion Matrix 시각화 ───────────────────────────────────────────────────

def plot_confusion_matrices(results):
    print("[4/5] Confusion Matrix 저장")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DriveGuard — Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        cm_norm = res["cm"].astype(float) / res["cm"].sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            linewidths=0.5,
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{name}\nAcc={res['acc']:.4f}  F1={res['f1']:.4f}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = "data/confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"      저장 완료: {out}\n")
    plt.show()


# ── 성능 비교 & 모델 저장 ────────────────────────────────────────────────────

def save_best(results):
    print("[5/5] 최고 성능 모델 저장")
    print("\n  ── 최종 성능 비교 ──────────────────────────────")
    print(f"  {'모델':<22} {'Accuracy':>9} {'F1-macro':>9}")
    print("  " + "─" * 44)
    for name, res in results.items():
        mark = " ★" if name == "XGBoost" else ""
        print(f"  {name:<22} {res['acc']:>9.4f} {res['f1']:>9.4f}{mark}")
    print()

    os.makedirs("models", exist_ok=True)
    joblib.dump(results["XGBoost"]["model"], MODEL_PATH)
    print(f"  XGBoost 모델 저장 완료: {MODEL_PATH}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    df = load_sample(DATA_PATH, SAMPLE_SIZE)
    X_tr, X_te, y_tr, y_te = split(df)
    models = build_models()
    results = train_evaluate(models, X_tr, X_te, y_tr, y_te)
    plot_confusion_matrices(results)
    save_best(results)
    print("\n완료.")


if __name__ == "__main__":
    main()
