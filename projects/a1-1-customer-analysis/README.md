# A1-1. 쇼핑몰 고객 분석과 RFM 세분화

- 과제 원문: [subjects/subject_a1-1.md](../../subjects/subject_a1-1.md)
- 진행 상태: 필수 요구사항 완료 (보너스 과제 미수행)

## 프로젝트 개요

H&M 거래 데이터로 고객의 구매 패턴을 분석해 **자주 오고 많이 사는 단골을 찾고**, 세그먼트별 전략을 제안합니다.
수치, 범주, 날짜, 텍스트, 이미지 배열을 하나의 파이프라인(`DataAnalyzer`)으로 처리하고, IQR 이상치 탐지, 그룹별 결측치 대치, 6종 시각화, RFM 세분화까지 수행합니다.

| 결과물 | 위치 |
|---|---|
| 분석 파이프라인 (OOP) | [`src/pipeline.py`](src/pipeline.py) |
| 분석 리포트 노트북 | [`notebooks/analysis_report.ipynb`](notebooks/analysis_report.ipynb) |
| 파이프라인 테스트 | [`tests/test_pipeline.py`](tests/test_pipeline.py) |

### 평가를 위한 문서

| 문서 | 내용 |
|---|---|
| [요구사항 대응표](reports/requirements_checklist.md) | 미션 요구사항(2, 4.1~4.8, 6~7장)별 구현 위치와 확인 방법, 충족 상태 |
| [개념 설명 Q&A](reports/concepts_qa.md) | 미션 3장의 학습 목표 5가지와 설계 선택에 대한 질문을 코드와 함께 설명 |
| [의사결정과 한계](reports/decisions_and_limitations.md) | 데이터 선택, 이미지 확보 경위, 분석 방법의 선택 이유, 한계, 확인하지 못한 것 |

## 데이터 설명

- **출처**: Kaggle 대회 [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) (거래 `transactions_train.csv`, 상품 `articles.csv`, 고객 `customers.csv`, 상품 이미지).
- **이용 조건**: 대회 약관에 동의한 계정으로 내려받았습니다. 약관상 재배포가 제한될 수 있어 **원본, 샘플, 이미지는 저장소에 포함하지 않았습니다** (`.gitignore`). 아래 실행 방법으로 재현합니다.
- **샘플링**: 전체 고객 중 5,000명을 무작위 추출(`seed=42`)하고 거래가 없는 고객을 제외해 **고객 4,962명, 거래 113,723행, 상품 34,543개**(2018-09-20 ~ 2020-09-22)를 사용했습니다.
- **컬럼**: 거래, 상품, 고객을 합친 41개 컬럼 (수치 `price`·`age`, 범주 `product_group_name` 등, 날짜 `t_dat`, 텍스트 `prod_name`) + 파생 변수(`amount`, 상품명 단어 수·길이, `img_mean`, `img_std`, `has_image`).
- **`price`는 정규화된 상대 금액**입니다. 분석에는 1,000을 곱한 `amount`(금액 지수)를 사용하며 실제 통화 금액이 아닙니다.
- **이미지**: 판매량 상위 상품 중 **765장**만 확보했습니다 (Kaggle API 요청 제한으로 목표 1,000장에 못 미침). 이미지가 있는 거래는 전체의 16.7%이며, 나머지 이미지 피처(83.3%)는 상품군 평균으로 대치했습니다. 이미지 분석은 이 상품들에만 해당합니다.

## 분석 방법 요약

| 단계 | 내용 |
|---|---|
| 피처 | 이미지 `(N, H, W, 3)` 배열의 평균·표준편차를 `axis=(1, 2, 3)` 집계로 계산 (반복문 없음), 상품명 단어 수·길이 |
| 결측치 | 연령은 고객 단위 그룹 중앙값, 이미지 피처는 상품 단위 그룹 평균으로 대치 (`has_image` 플래그 유지) |
| 이상치 | IQR(1.5)로 `amount` 상한 초과 5,144건(4.5%) 탐지, 클리핑 전후 비교. 이상치는 고가 상품의 정상 구매라 RFM에는 원본 금액 사용 |
| RFM | Recency(마지막 거래일 다음 날 기준), Frequency(서로 다른 구매일 수), Monetary(금액 합계)를 1~5점으로 변환하고 6개 세그먼트로 분류 |

### RFM 세그먼트 결과

| 세그먼트 | 고객 수 | 고객 비중 | 평균 Monetary | 평균 구매 횟수 | 평균 Recency(일) | 매출 비중 |
|---|---:|---:|---:|---:|---:|---:|
| VIP | 1,190 | 24.0% | 1,696.7 | 17.6 | 37 | 64.1% |
| Loyal | 956 | 19.3% | 593.7 | 6.3 | 106 | 18.0% |
| At Risk | 520 | 10.5% | 530.5 | 5.6 | 374 | 8.8% |
| Regular | 448 | 9.0% | 139.0 | 1.4 | 158 | 2.0% |
| New | 386 | 7.8% | 127.3 | 1.4 | 49 | 1.6% |
| Churned | 1,462 | 29.5% | 119.7 | 1.2 | 499 | 5.6% |

## 비즈니스 인사이트

### 인사이트 1. 소수의 VIP가 매출 대부분을 만든다

- **(근거)** VIP는 고객 1,190명(24.0%)인데 매출의 64.1%를 차지한다. 평균 구매 17.6회, 마지막 구매는 평균 37일 전이다. 고객 전체로 봐도 상위 10%가 매출의 50.2%를 만든다 (7장 RFM 표, 막대그래프).
- **(실행)** VIP 대상 → 신상품 사전 공개와 전용 혜택 등 멤버십 혜택 강화 → 이탈 방지. VIP 1명의 평균 구매 금액이 New 고객의 13배이므로 혜택 비용을 감안해도 우선순위가 높다.
- **(검증)** 혜택의 비용 대비 효과를 계산하려면 마진과 쿠폰 사용 데이터가 필요하다. 또한 VIP 매출이 소수의 고가 구매에 몰린 것이라면 반증이 되므로, 구매 단가 분포를 함께 확인해야 한다.

### 인사이트 2. 과거 가치가 큰 At Risk를 Churned보다 먼저 공략한다

- **(근거)** At Risk는 고객 520명(10.5%)으로, 평균 5.6회 구매했고 평균 구매 금액이 530.5이지만 마지막 구매가 평균 374일 전이다. Churned(고객 1,462명, 29.5%)는 평균 1.2회, 119.7으로, At Risk의 과거 구매 가치가 4.4배 높다.
- **(실행)** At Risk 대상 → 과거 구매 카테고리 기반의 개인화 추천 메일과 복귀 쿠폰 발송 → 소수 인원으로 큰 매출 회복을 기대할 수 있다. 쿠폰 비용이 한정돼 있다면 Churned보다 At Risk에 먼저 배정한다.
- **(검증)** 관측 기간이 2년이라 '오래 구매 안 함'이 곧 이탈은 아닐 수 있고(반증 가능성), 이탈 사유(가격, 품질, 경쟁사)를 알려면 설문 또는 방문 로그 데이터가 추가로 필요하다.

### 인사이트 3. 첫 구매 이후 재구매로 이어지지 못한 고객이 많다

- **(근거)** 전체 고객의 32.7%(1,622명)가 한 번만 구매했고, 이 중 69.1%가 Churned다. Churned의 평균 구매 횟수는 1.2회로, 이탈 고객 대부분이 1~2회 구매에서 끝났다. 반면 구매 횟수와 총 구매 금액의 상관계수는 0.83(강한 양의 관계)이며, 구매 횟수가 총 구매 금액 차이의 약 68%를 설명한다.
- **(실행)** New·Regular 대상 → 첫 구매 후 일정 기간 안에 2차 구매를 유도하는 웰컴 메시지와 2회차 구매 쿠폰 → 재구매율 상승과 장기적으로 VIP·Loyal 비중 확대를 기대한다.
- **(검증)** 첫 구매 후 2차 구매까지의 간격 분포와 가입 월별 재구매율(코호트 분석)이 추가로 필요하다. 한 번만 산 고객이 선물 구매처럼 원래 재구매 의사가 없는 경우라면 반증이 된다.

### 인사이트 4. 6월 성수기에 맞춘 시즌 캠페인

- **(근거)** 월 매출이 2019년(2019-06, 174,838)과 2020년(2020-06, 157,781) 모두 6월에 해당 연도 최고였고, 월 평균 130,188을 크게 웃돈다. 6월의 구매 고객 수도 1,091명, 1,122명으로 월 평균 931명보다 많다 (6장 라인차트, 일부만 포함된 첫 달과 마지막 달은 제외).
- **(실행)** 활동 중인 Loyal·VIP 대상 → 5월 말부터 시즌 상품 사전 안내와 한정 혜택 → 성수기 매출 상승폭 확대. 재고와 프로모션도 6월에 맞춰 선행 준비한다.
- **(검증)** 관측 기간이 2년뿐이어서 계절성이라 단정하기 어렵다(반증 가능성). 상품군별 월별 매출과 프로모션 일정 데이터가 있으면 원인을 가를 수 있다.

## 한계

- 이미지 피처는 판매량 상위 765개 상품에만 실제 값이 있어 결과를 전체 상품으로 일반화하기 어렵습니다.
- 관측 기간이 약 2년이라 Churned는 실제 이탈이 아닐 수 있습니다.
- 금액이 정규화된 상대 값이라 절대 금액과 마진은 알 수 없습니다.
- 보너스 과제(이미지 고급 피처, 코호트 분석, Plotly)는 수행하지 않았습니다.

## 실행 방법

모든 명령은 저장소 루트에서 실행합니다 (`uv` 환경, Python 3.13).

```bash
uv sync

# 1. 데이터 내려받기 (Kaggle 계정, 대회 약관 동의, API 토큰 필요)
export KAGGLE_API_TOKEN=<토큰>
for f in articles.csv customers.csv transactions_train.csv; do
  uv run kaggle competitions download h-and-m-personalized-fashion-recommendations \
    -f $f -p projects/a1-1-customer-analysis/data/raw
done
(cd projects/a1-1-customer-analysis/data/raw && unzip -o '*.zip')

# 2. 고객 5,000명 샘플링 -> data/sample/
uv run python projects/a1-1-customer-analysis/src/prepare_sample.py --n-customers 5000 --seed 42

# 3. 상품 이미지 내려받기 -> data/images/ (요청 제한 때문에 오래 걸리거나 일부만 받아질 수 있음)
uv run python projects/a1-1-customer-analysis/src/download_images.py --top-n 1000 --workers 2

# 4. 이미지를 NumPy 배열로 변환 -> data/processed/
uv run python projects/a1-1-customer-analysis/src/build_image_array.py --stride 10

# 5. 테스트와 노트북
uv run pytest projects/a1-1-customer-analysis/tests
uv run jupyter lab projects/a1-1-customer-analysis/notebooks/analysis_report.ipynb
```

`download_images.py`는 이미 받은 파일을 건너뛰므로 중단 후 다시 실행해도 됩니다. 받아지는 이미지 수에 따라 노트북의 이미지 관련 수치는 달라질 수 있습니다.

## 폴더 구조

```
a1-1-customer-analysis/
├── data/            # raw(원본), sample(샘플), images, processed(배열) - 저장소 제외
├── notebooks/       # analysis_report.ipynb (제출용 분석 리포트)
├── src/             # pipeline.py(DataAnalyzer), prepare_sample.py, download_images.py, build_image_array.py
├── tests/           # pipeline 단위 테스트
├── reports/         # 평가용 문서 (요구사항 대응표, 개념 Q&A, 의사결정과 한계)
└── studys/          # 개인 공부 노트북 (제출 대상 아님)
```

## 공부 기록

개인 공부 노트와 시도한 방법, 실패 원인, 배운 점은 [`studys/`](studys/)에 기록합니다. `notebooks/`는 미션에서 요구하는 실험·분석 노트북에 사용합니다.
