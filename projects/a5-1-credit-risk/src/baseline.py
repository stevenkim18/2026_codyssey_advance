"""규칙 기반 베이스라인 (A5-1): if-else 규칙 5개의 투표로 연체를 예측한다.

머신러닝 모델이 규칙보다 얼마나 나은지 재려면 기준선이 필요하다. 사람이 경험으로 만드는 규칙을
최대한 공정하게 만들기 위해 규칙의 임계값은 **Train 데이터의 구간별 연체율**을 보고 정했고,
Test 는 최종 평가에서 한 번만 쓴다.

규칙 5개 (한 고객에 대해 걸리는 규칙의 수를 센다)
    1. 최근 6개월 연체 2회 이상          Train 연체율 38.7%
    2. 부채 비율 0.8 초과                Train 연체율 23.8%
    3. 연 소득 3,000만원 미만            Train 연체율 40.8%
    4. 연 소득 4,000만원 미만 이면서 부채 비율 0.6 초과
    5. 최근 6개월 연체 1회 이상 이면서 부채 비율 0.5 초과

예측: 걸린 규칙이 `min_hits` 개 이상이면 연체(1). `min_hits` 는 Train F1 이 가장 높은 값으로 고른다.
규칙 1개만 걸려도 연체로 보면(min_hits=1) 거의 모두를 연체라고 해서 Precision 이 무너진다.

과제 예시에 있는 "신용카드 8장 이상" 규칙은 뺐다. Train 에서 카드 8장 이상의 연체율이 12.3% 로
전체 평균(12.0%)과 같아서, 가짜 양성만 늘리기 때문이다 (이 변수는 데이터 생성식에 들어 있지 않다).

AUC 는 걸린 규칙의 수(0~5)를 점수로 삼아 계산한다. 규칙 모델은 확률을 내지 않기 때문이다.

실행: `uv run python -m src.baseline` (프로젝트 폴더에서)
"""

import pandas as pd
from sklearn.metrics import f1_score

from src.metrics import classification_metrics, to_markdown_table
from src.preprocess import Split, load_data, split_data

# 규칙 임계값 (Train 구간별 연체율을 보고 정함)
OVERDUE_MIN = 2
DEBT_HIGH = 0.8
INCOME_LOW = 3000
INCOME_MID = 4000
DEBT_MID_HIGH = 0.6
DEBT_MID = 0.5

RULE_NAMES = [
    f"연체 {OVERDUE_MIN}회 이상",
    f"부채 비율 > {DEBT_HIGH}",
    f"연 소득 < {INCOME_LOW}",
    f"연 소득 < {INCOME_MID} 이면서 부채 비율 > {DEBT_MID_HIGH}",
    f"연체 1회 이상 이면서 부채 비율 > {DEBT_MID}",
]


def rule_flags(row: pd.Series) -> list[int]:
    """고객 한 명에게 규칙 5개가 각각 걸리는지(1) 아닌지(0). 규칙의 유일한 정의이며 if 문으로 적었다."""
    flags = [0] * len(RULE_NAMES)
    if row["overdue_count_6m"] >= OVERDUE_MIN:  # 규칙 1: 연체 이력이 잦다
        flags[0] = 1
    if row["debt_ratio"] > DEBT_HIGH:  # 규칙 2: 빚이 소득에 비해 많다
        flags[1] = 1
    if row["annual_income"] < INCOME_LOW:  # 규칙 3: 소득이 낮다
        flags[2] = 1
    if row["annual_income"] < INCOME_MID and row["debt_ratio"] > DEBT_MID_HIGH:  # 규칙 4
        flags[3] = 1
    if row["overdue_count_6m"] >= 1 and row["debt_ratio"] > DEBT_MID:  # 규칙 5
        flags[4] = 1
    return flags


def count_rule_hits(row: pd.Series) -> int:
    """고객 한 명에게 걸리는 규칙의 수 (0~5)."""
    return sum(rule_flags(row))


def rule_based_predict(row: pd.Series, min_hits: int) -> int:
    """걸린 규칙이 min_hits 개 이상이면 연체(1), 아니면 정상(0)."""
    if count_rule_hits(row) >= min_hits:
        return 1
    return 0


def select_min_hits(X_train: pd.DataFrame, y_train: pd.Series) -> int:
    """Train 의 F1 이 가장 높은 min_hits 를 고른다 (1~5)."""
    hits = X_train.apply(count_rule_hits, axis=1)
    scores = {k: f1_score(y_train, (hits >= k).astype(int), zero_division=0) for k in range(1, 6)}
    return max(scores, key=scores.get)


def run_baseline(split: Split) -> dict:
    """Train 로 min_hits 를 고르고 Test 에서 평가한다. 결과를 딕셔너리로 돌려준다."""
    min_hits = select_min_hits(split.X_train, split.y_clf_train)
    hits_test = split.X_test.apply(count_rule_hits, axis=1)
    y_pred = split.X_test.apply(rule_based_predict, axis=1, min_hits=min_hits)
    return {
        "min_hits": min_hits,
        "y_pred": y_pred,
        "y_score": hits_test,
        "metrics": classification_metrics(split.y_clf_test, y_pred, hits_test),
    }


def main() -> None:
    split = split_data(load_data())

    # Train 에서 규칙 하나하나가 얼마나 맞는지 (규칙을 정한 근거)
    y = split.y_clf_train
    print("[규칙별 Train 성능] 규칙 단독으로 '걸리면 연체'라고 했을 때")
    flags = split.X_train.apply(rule_flags, axis=1, result_type="expand")
    rows = {}
    for i, name in enumerate(RULE_NAMES):
        fired = flags[i] == 1
        tp = int((fired & (y == 1)).sum())
        rows[f"규칙 {i + 1}. {name}"] = {
            "발동 수": int(fired.sum()),
            "연체율": tp / max(int(fired.sum()), 1),
            "연체 포착률": tp / int(y.sum()),
        }
    print(to_markdown_table(pd.DataFrame(rows).T.astype({"발동 수": int}).rename_axis("규칙")))

    hits_train = split.X_train.apply(count_rule_hits, axis=1)
    print("\n[min_hits 선택] Train F1")
    for k in range(1, 6):
        f1 = f1_score(y, (hits_train >= k).astype(int), zero_division=0)
        print(f"  규칙 {k}개 이상 걸리면 연체 → F1 {f1:.3f}")

    result = run_baseline(split)
    print(f"\n선택된 min_hits = {result['min_hits']}")
    print("\n[Test 성능] 규칙 기반 베이스라인")
    table = pd.DataFrame([result["metrics"]], index=["규칙 기반 베이스라인"]).rename_axis("모델")
    print(to_markdown_table(table))


if __name__ == "__main__":
    main()
