"""분류·회귀 평가 지표 (A5-1). 베이스라인, 회귀, 분류 모듈이 같은 계산을 쓰도록 모았다."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """연체(1)를 양성으로 본 분류 지표.

    y_pred 는 0/1 예측, y_score 는 AUC 계산용 점수(확률, 또는 규칙 발동 수)다.
    불균형 데이터(연체 12%)에서는 전부 '정상'이라고 해도 Accuracy 가 0.88 이므로, Accuracy 만으로
    모델을 고르지 않고 Recall·F1·AUC 를 함께 본다.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """회귀 지표 3종. RMSE 는 큰 오차에 민감하고, MAE 는 평균 절대 오차, R² 는 설명력이다."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def to_markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """DataFrame 을 GitHub 마크다운 표로 바꾼다 (tabulate 의존성 없이). 첫 열은 인덱스 이름."""
    header = [df.index.name or "", *df.columns]
    lines = [
        "| " + " | ".join(str(h) for h in header) + " |",
        "| --- | " + " | ".join("---:" for _ in df.columns) + " |",
    ]
    for i, idx in enumerate(df.index):
        # 열 단위로 값을 꺼내야 정수 열이 실수로 바뀌지 않는다 (iterrows 는 행을 공통 dtype 으로 바꾼다)
        cells = []
        for col in df.columns:
            v = df[col].iloc[i]
            cells.append(f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
