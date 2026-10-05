# 요구사항 대응표

미션 원문([subject_a1-1.md](../../../subjects/subject_a1-1.md))의 요구사항별로 **어디에 구현했는지**와 **확인 방법**을 정리했습니다.
상태는 ✅ 충족, ⚠️ 충족했지만 주의할 점 있음, ➖ 선택 과제로 수행하지 않음입니다.

노트북 장 번호: 1 로드·탐색, 2 피처, 3 결측치, 4 이상치, 5 통계, 6 시각화, 7 RFM, 8 인사이트, 9 한계.

## 2. 최종 결과물

| 요구 | 상태 | 위치 | 비고 |
|---|---|---|---|
| `src/pipeline.py` (OOP 클래스) | ✅ | [`src/pipeline.py`](../src/pipeline.py) | `DataAnalyzer` 클래스. 절차형 스크립트가 아님 |
| `notebooks/analysis_report.ipynb` | ✅ | [`notebooks/analysis_report.ipynb`](../notebooks/analysis_report.ipynb) | 단계마다 마크다운 해석. 실행 결과(차트 11개)가 저장돼 있음 |
| `README.md` (개요, 데이터 설명, 인사이트 3개 이상, 실행 방법) | ✅ | [`README.md`](../README.md) | 인사이트 4개 |

## 4.1 데이터 로드와 기본 탐색

| 요구 | 상태 | 위치 | 확인 |
|---|---|---|---|
| 1,000건 이상, 8컬럼 이상 | ✅ | 노트북 1장 | 113,723행, 36컬럼 |
| 수치, 범주, 날짜, 이미지 중 3유형 이상 | ✅ | 노트북 개요 표 | 수치, 범주, 날짜, 텍스트, 이미지(5유형) |
| `info()`, `describe()`, `head()` + 마크다운 해석 | ✅ | 노트북 1장 | 해석에 수치 근거와 시사점 포함 |
| RFM용 `customer_id`, 날짜, 금액 컬럼 | ✅ | `load_data()` | 날짜 컬럼명은 `t_dat`, 금액은 `amount = price × 1000` |
| Recency 기준일 = 최종 거래일(또는 분석 기준일) | ✅ | `calculate_rfm()` | 마지막 거래일의 다음 날(2020-09-23)을 기준일로 사용. 노트북 7장에 명시 |
| 해석에 수치 근거 1개 이상 + 시사점 1문장 이상 | ✅ | 노트북 전 장 | 각 해석 셀에서 확인 가능 |

⚠️ `price`는 H&M이 정규화한 상대 금액입니다. 실제 통화가 아니라는 점을 노트북과 README에 명시했습니다.

## 4.2 분석 파이프라인 모듈화(OOP)

| 요구 | 상태 | 위치 | 확인 |
|---|---|---|---|
| `DataAnalyzer` 클래스 | ✅ | `src/pipeline.py` | |
| 필수 메서드 4개 | ✅ | `load_data`, `handle_missing_values`, `detect_outliers`, `calculate_rfm` | `tests/test_pipeline.py`에서 각각 검증 |
| 주요 파라미터를 메서드 인자로 | ✅ | 아래 표 | |

| 메서드 | 인자로 받는 파라미터 |
|---|---|
| `handle_missing_values` | `column`, `strategy`(mean, median, group_mean, group_median, drop), `group_col`, `unit_col` |
| `detect_outliers` / `iqr_bounds` / `treat_outliers` | `column`, `threshold`(기본 1.5), `method`(clip, remove) |
| `calculate_rfm` | `customer_col`, `date_col`, `amount_col`, `n_bins`, `snapshot_date` |

⚠️ 미션 예시의 `__init__(data_path)`와 달리 `__init__(data_dir)`로 폴더를 받습니다. 거래, 상품, 고객 세 테이블을 합쳐서 읽기 때문입니다. 예시는 참고용이라고 원문에 적혀 있습니다.

## 4.3 멀티 모달 피처 엔지니어링

| 요구 | 상태 | 위치 | 확인 |
|---|---|---|---|
| 이미지 평균/표준편차를 새 컬럼으로, 반복문 없이 | ✅ | `add_image_features()` | `images.mean(axis=(1, 2, 3))`로 한 번에 계산. 노트북 2장에서 직접 계산값과 일치 확인, 테스트로도 검증 |
| 텍스트 단어 수 또는 길이 | ✅ | `add_text_features()` | `prod_name_word_count`, `prod_name_len` |

⚠️ 이미지는 765장만 확보해서 이미지 피처에 실제 값이 있는 거래는 16.7%입니다. 나머지는 상품군 평균으로 대치했고(`has_image` 컬럼으로 구분), 이미지 관련 분석은 이미지가 있는 행으로 한정했습니다. 자세한 경위는 [decisions_and_limitations.md](decisions_and_limitations.md)에 있습니다.

## 4.4 결측치와 이상치

| 요구 | 상태 | 위치 | 확인 |
|---|---|---|---|
| 결측 현황 파악 | ✅ | `missing_report()`, 노트북 3장 | |
| GroupBy 통계 대치 1가지 이상 | ✅ | `handle_missing_values("age", "group_median", "club_member_status", ...)`, 이미지 피처는 `product_group_name` 그룹 평균 | 그룹이 결측이면 전체 통계로 대체하는 규칙을 테스트로 검증 |
| IQR 이상치 탐지를 직접 구현 | ✅ | `iqr_bounds()`, `detect_outliers()` | 라이브러리 함수 없이 사분위수로 계산 |
| 처리 전후 분포 시각화 | ✅ | 노트북 4장 | 히스토그램 전후, 박스플롯 전후 |

## 4.5 통계 분석

| 요구 | 상태 | 위치 | 확인 |
|---|---|---|---|
| 평균, 중앙값, 표준편차, 사분위수 + 해석 | ✅ | `summary_statistics()`, 노트북 5장 | |
| 2쌍 이상 상관계수 + 해석 | ✅ | `correlation_pairs()`, 노트북 5장, 7장 | 5쌍(전체 거래/이미지 있는 행) + 고객 단위 2쌍 |

## 4.6 시각화 (6종, 제목과 축 레이블 필수)

| 차트 | 상태 | 위치 |
|---|---|---|
| 히스토그램 | ✅ | 4장(금액 분포 전후), 6.1(연령, 이미지 밝기) |
| 박스플롯 (이상치 처리 전후) | ✅ | 4장 |
| 막대그래프 | ✅ | 6.2(상품군별 거래 수), 7장(세그먼트별 고객 수, 매출 비중) |
| 히트맵 | ✅ | 6.3(상관행렬), 7장(세그먼트별 평균 RFM 점수) |
| 산점도 | ✅ | 6.4(이미지 밝기와 금액), 7장(구매 횟수와 금액) |
| 라인차트 | ✅ | 6.5(월별 매출, 구매 고객 수) |

차트는 `label()` 헬퍼로 제목, x축, y축 레이블을 모두 지정했습니다. 2장의 상품 이미지 예시는 6종에 포함되지 않는 보조 그림이라 축 없이 제목만 있습니다.

## 4.7 RFM 기반 고객 세분화

| 요구 | 상태 | 위치 | 확인 |
|---|---|---|---|
| 고객별 R, F, M 점수 | ✅ | `calculate_rfm()` | 1~5점. 동점은 같은 점수(테스트로 검증) |
| 4개 이상 그룹 + 특징 분석 | ✅ | `assign_segments()`, 노트북 7장 | 6개 세그먼트(VIP, Loyal, New, At Risk, Churned, Regular) |
| 비즈니스 인사이트를 README에 | ✅ | README | |

## 4.8 인사이트 작성 기준

| 요구 | 상태 | 위치 |
|---|---|---|
| 각 인사이트에 (근거), (실행), (검증) | ✅ | README, 노트북 8장 (4개 모두 3요소 포함) |

## 5. 보너스 과제

| 과제 | 상태 |
|---|---|
| 이미지 고급 피처(히스토그램, 엣지) | ➖ |
| 코호트 분석 | ➖ (인사이트 3의 검증 항목으로 다음 단계 후보) |
| Plotly 인터랙티브 시각화 | ➖ |

## 6~7. 개발 환경과 제약

| 항목 | 상태 | 확인 |
|---|---|---|
| 허용 라이브러리(NumPy, Pandas, Matplotlib, Seaborn) | ⚠️ | 분석 코드(`pipeline.py`)와 노트북은 이 네 개와 표준 라이브러리만 import. **데이터 내려받기 스크립트(`download_images.py`)만 `kaggle`, `requests` 사용** (분석이 아닌 수집 단계) |
| OpenCV, Pillow 금지 | ✅ | `src/`, `tests/`, 노트북에서 import 없음 (검색으로 확인). 이미지는 `matplotlib.image.imread`와 NumPy 슬라이싱으로 처리 |
| scikit-learn, NLTK 금지 | ✅ | 위와 같이 import 없음 |
| 자동 EDA 라이브러리 금지 | ✅ | 사용하지 않음 |
| 반복문 대신 배열 연산 | ✅ | 이미지 피처 계산에 반복문 없음. `build_image_array.py`의 반복은 파일 읽기(I/O)이고 통계 계산이 아님 |
| Python 3.8 이상 | ✅ | 프로젝트는 3.13(`pyproject.toml`) |
| 출처 명시, 공개 라이선스 우선 | ⚠️ | 출처는 README에 명시. 다만 H&M 데이터는 CC-BY/MIT 같은 공개 라이선스가 아니라 **Kaggle 대회 약관**을 따르며, 약관 원문은 확인하지 못함. 그래서 데이터는 저장소에 포함하지 않음 |

⚠️ 저장소의 `pyproject.toml`에는 다른 미션용으로 OpenCV, scikit-learn, NLTK 등이 설치돼 있습니다. 이 프로젝트의 코드는 사용하지 않습니다.

## 직접 확인하는 방법

```bash
# 테스트 (17개)
uv run pytest projects/a1-1-customer-analysis/tests

# 금지 라이브러리 import 검색 (결과가 없어야 함)
grep -rnE "^\s*(import|from)\s+(cv2|PIL|sklearn|nltk|sweetviz|pandas_profiling|ydata_profiling|plotly)" \
  projects/a1-1-customer-analysis/src projects/a1-1-customer-analysis/tests
```
