# 2026 Codyssey Advance

과제별 공부 과정과 제출 결과물을 모으는 저장소입니다.

## 프로젝트 목록

| 과제 | 프로젝트 | 상태 | 과제 원문 |
| --- | --- | --- | --- |
| [A1-1](projects/a1-1-customer-analysis/README.md) | 쇼핑몰 고객 분석과 RFM 세분화 | 시작 전 | [과제](subjects/subject_a1-1.md) |
| [A2-1](projects/a2-1-ai-math/README.md) | AI 학습 원리와 수학 구현 | 시작 전 | [과제](subjects/subject_a2-1.md) |
| [A3-1](projects/a3-1-document-scanner/README.md) | 문서 스캐너 | 시작 전 | [과제](subjects/subject_a3-1.md) |
| [A3-2](projects/a3-2-object-tracker/README.md) | 동영상 객체 추적 | 시작 전 | [과제](subjects/subject_a3-2.md) |
| [A4-1](projects/a4-1-document-search/README.md) | TF-IDF 문서 검색 | 시작 전 | [과제](subjects/subject_a4-1.md) |
| [A4-2](projects/a4-2-information-sentiment/README.md) | 정보 추출과 감성 분석 | 시작 전 | [과제](subjects/subject_a4-2.md) |
| [A5-1](projects/a5-1-credit-risk/README.md) | 신용 점수와 연체 위험 예측 | 시작 전 | [과제](subjects/subject_a5-1.md) |
| [A5-2](projects/a5-2-explainable-ai/README.md) | 고객 군집과 SHAP 설명 | 시작 전 | [과제](subjects/subject_a5-2.md) |
| [A6-1](projects/a6-1-autograd/README.md) | NumPy 자동 미분 엔진 | 시작 전 | [과제](subjects/subject_a6-1.md) |
| [A6-2](projects/a6-2-error-analysis/README.md) | 전이 학습과 시계열 오차 분석 | 시작 전 | [과제](subjects/subject_a6-2.md) |
| [A7-1](projects/a7-1-team-project/README.md) | CV·NLP 팀 종합 프로젝트 | 시작 전 | [과제](subjects/subject_a7-1.md) |

## 폴더 구성

- `subjects/`: 과제 원문
- `projects/<과제>/src/`: 재실행 가능한 구현 코드
- `projects/<과제>/notebooks/`: 미션에서 요구하는 실험·분석 노트북
- `projects/<과제>/studys/`: 개인 공부 노트와 학습 기록
- `projects/<과제>/tests/`: 검증 코드
- `projects/<과제>/reports/`: 분석 보고서와 기획 문서
- `projects/<과제>/outputs/`: 제출할 그래프와 예시 결과
- `projects/<과제>/data/`: 과제에 필요한 데이터

각 프로젝트의 `README.md`에 목표, 실행 방법, 결과와 제출 링크를 기록합니다. 빈 폴더의 `.gitkeep`은 첫 파일을 추가할 때 삭제해도 됩니다.

## 개발 환경

Python 3.13과 루트의 단일 uv 환경을 사용합니다.

```bash
uv sync
uv run python projects/<과제>/src/<실행파일>.py
```

의존성은 루트에서 `uv add` 또는 `uv remove`로 관리하고, `pyproject.toml`과 `uv.lock`을 함께 반영합니다. `requirements.txt`가 필수인 과제는 제출 시 해당 프로젝트 폴더에 별도로 준비합니다.

## 제출

과제별 제출 조건은 `subjects/`의 원문을 확인합니다. 독립적인 GitHub 저장소 URL이 필요한 과제는 해당 프로젝트를 별도 저장소로 제출하고, 링크를 프로젝트 `README.md`에 기록합니다.
