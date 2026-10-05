# 미션: 원하는 내용이 들어있는 문서를 똑똑하게 찾아주는 검색기 만들기

> 개인 필수 미션 · AI/SW 심화 · 자연어 처리 · 70시간

## 1. 미션 소개

뉴스나 리뷰 데이터로 토큰화, TF-IDF, 코사인 유사도 기반 문서 검색·분류 시스템을 구축한다. 핵심 TF-IDF 계산을 NumPy로 직접 구현하고 BoW가 문맥과 어순을 놓치는 한계를 분석한다.

## 2. 최종 결과물

- 전처리·토큰화·불용어 제거 파이프라인
- NumPy TF-IDF 직접 구현과 scikit-learn 검증 로그
- 코사인 유사도 검색 및 TF-IDF 분류 모델
- `README.md`: 구현 요약, 검증/실험, 한계·개선, 실행 방법
- 실행 인터페이스: CLI `python main.py --query "..." --topk 5` 또는 `search(query)` 함수

## 3. 기능 요구 사항

- 문서 500개 이상, 카테고리 2개 이상
- 특수문자 제거, 소문자화, 토큰화, 불용어 제거
- 한국어 형태소 분석 또는 공백 토큰화 대안과 성능 차이 분석
- 전처리 기준의 설계 근거 기록
- TF·IDF·TF-IDF 행렬을 NumPy로 계산
- `TfidfVectorizer`와 오차 `1e-6` 이내 검증
- `sublinear_tf`, `smooth_idf`, `norm` 등 검증 설정을 README에 명시
- 쿼리 벡터화와 코사인 유사도를 NumPy로 직접 계산
- 상위 5개 문서의 점수, ID, 스니펫 반환
- Logistic Regression 또는 Linear SVM, 8:2 분할, 고정 `random_state`
- Accuracy, F1, 혼동 행렬 및 오분류 5개 이상 분석

## 4. 보너스 과제

- Raw Count, Log, Double Normalization TF 비교
- BM25 직접 구현 및 TF-IDF와 결과 비교
- 역색인과 전체 스캔의 시간복잡도 비교

## 5. 데이터 가이드

- 한국어: NSMC, AI Hub 한국어 뉴스
- 영어: 20 Newsgroups, AG News, IMDB
- 결측치·개인정보·라이선스 확인 필수

## 6. 개발 환경과 제약

- Python 3.10+, `requirements.txt` 필수
- TF-IDF 핵심 로직과 코사인 유사도는 NumPy로 직접 구현
- Pandas, NumPy, Matplotlib/Seaborn, KoNLPy, NLTK 허용
- scikit-learn은 분류 및 검증용
- 식별 가능한 개인정보·연락처·건강정보 포함 데이터 금지

