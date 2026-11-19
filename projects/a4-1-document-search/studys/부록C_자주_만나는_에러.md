# 부록 C. 자주 만나는 에러

> 이 과제를 하면서 자주 만나는 에러와 이상한 결과를 **증상 → 원인 → 해결 → 관련 장** 순서로 정리했습니다. 에러 메시지의 마지막 줄을 아래 제목에서 찾아보세요.

## 목차
- **환경과 설치**
  - [C1. KoNLPy: JVM을 찾을 수 없음](#c1-konlpy-jvm을-찾을-수-없음)
  - [C2. NLTK: LookupError](#c2-nltk-lookuperror)
  - [C3. ModuleNotFoundError: No module named 'src'](#c3-modulenotfounderror-no-module-named-src)
  - [C4. 노트북 실행 문제 (NameError, 커널, nbconvert)](#c4-노트북-실행-문제)
- **데이터**
  - [C5. 20 Newsgroups 첫 다운로드](#c5-20-newsgroups-첫-다운로드)
  - [C6. KLUE-YNAT 다운로드 실패](#c6-klue-ynat-다운로드-실패)
  - [C7. read_csv 구분자와 인코딩](#c7-read_csv-구분자와-인코딩)
  - [C8. groupby().sample에서 n이 그룹보다 큼](#c8-groupbysample에서-n이-그룹보다-큼)
  - [C9. Okt 캐시가 갱신되지 않음](#c9-okt-캐시가-갱신되지-않음)
- **NumPy 계산**
  - [C10. 브로드캐스팅 ValueError](#c10-브로드캐스팅-valueerror)
  - [C11. 0으로 나누기와 nan, log(0) 경고](#c11-0으로-나누기와-nan-log0-경고)
  - [C12. X[rows, cols] += 1로 센 개수가 모자람](#c12-xrows-cols--1로-센-개수가-모자람)
  - [C13. 메모리 부족 (커널이 죽음)](#c13-메모리-부족)
- **검증과 모델**
  - [C14. sklearn과 결과가 다름 (어휘·오차 불일치)](#c14-sklearn과-결과가-다름)
  - [C15. np.allclose는 통과했는데 기준이 다름](#c15-npallclose는-통과했는데-기준이-다름)
  - [C16. LogisticRegression ConvergenceWarning](#c16-logisticregression-convergencewarning)
- **그림**
  - [C17. 한글 글꼴 깨짐과 마이너스 기호](#c17-한글-글꼴-깨짐과-마이너스-기호)

---

## 환경과 설치

### C1. KoNLPy: JVM을 찾을 수 없음

**증상**
```
jpype._jvmfinder.JVMNotFoundException: No JVM shared library file (libjvm.dylib) found.
Try setting up the JAVA_HOME environment variable properly.
```
또는 Apple Silicon Mac에서 아키텍처가 맞지 않는다는 오류(`mach-o file, but is an incompatible architecture`)가 납니다.

**원인**
- KoNLPy는 파이썬 래퍼일 뿐이고, Okt 분석 엔진은 Java로 만들어져 있습니다. JPype가 파이썬 안에서 JVM을 띄워야 하는데, JDK가 없거나 위치를 못 찾았습니다.
- `JAVA_HOME`이 JDK가 아닌 엉뚱한 폴더(JRE나 지워진 경로)를 가리키고 있을 수 있습니다.
- Apple Silicon(arm64)에서 x86용 JDK를 설치하면 파이썬과 JVM의 아키텍처가 달라 실패합니다.

**해결**
1. JDK를 설치합니다. macOS는 `brew install openjdk@17`, Ubuntu는 `sudo apt install openjdk-17-jdk`, Windows는 Temurin JDK 17을 설치합니다.
2. `JAVA_HOME`을 설정합니다. macOS는 `export JAVA_HOME=$(/usr/libexec/java_home)`이고, Windows는 환경 변수에서 JDK 폴더로 지정합니다.
3. 터미널과 노트북 커널을 다시 시작합니다. 환경 변수는 실행 중인 커널에 반영되지 않습니다.
4. `java -version`과 1장의 환경 점검 셀로 확인합니다.

**관련 장:** [1장 1.4](01_개발_환경과_프로젝트_구조.ipynb)

### C2. NLTK: LookupError

**증상**
```
LookupError:
  Resource stopwords not found.
  Please use the NLTK Downloader to obtain the resource:
  >>> import nltk
  >>> nltk.download('stopwords')
```
`word_tokenize`를 쓰면 `Resource punkt_tab not found`가 나옵니다.

**원인:** NLTK는 라이브러리와 데이터(불용어 목록, 토크나이저 모델)를 따로 배포합니다. `uv add nltk`만으로는 데이터가 설치되지 않습니다.

**해결**
```python
import nltk
nltk.download("stopwords")   # 10장 불용어
nltk.download("punkt_tab")   # 9장 word_tokenize (예전 버전은 "punkt")
```
데이터는 홈 폴더의 `nltk_data/`에 저장되어 한 번만 받으면 됩니다. 1장의 `ensure_nltk` 함수처럼 "없을 때만 받기"를 코드에 넣어 두면 다른 컴퓨터에서도 바로 실행됩니다.

**관련 장:** [1장 1.5](01_개발_환경과_프로젝트_구조.ipynb), [9장](09_토큰화.ipynb), [10장](10_불용어와_전처리_파이프라인.ipynb)

### C3. ModuleNotFoundError: No module named 'src'

**증상**
```
$ python src/verify.py
ModuleNotFoundError: No module named 'src'
```

**원인:** 파일 경로로 실행하면 `sys.path`의 첫 위치가 그 파일이 있는 폴더(`src/`)가 됩니다. `src/` 안에서 `src`라는 패키지를 찾으니 실패합니다. 노트북에서도 `studys/`에서 실행하면 `src/`가 보이지 않습니다.

**해결**
- 터미널에서는 **프로젝트 폴더**에서 모듈 이름으로 실행합니다: `uv run python -m src.verify`
- `main.py`는 프로젝트 루트에 있으므로 `uv run python main.py ...`로 실행하면 됩니다.
- 노트북에서는 첫 셀에 `import sys; sys.path.insert(0, "..")`를 넣어 프로젝트 루트를 경로에 추가합니다.
- `src/` 폴더에 `__init__.py`가 있는지 확인합니다.

**관련 장:** [1장 1.3](01_개발_환경과_프로젝트_구조.ipynb), [22장](22_통합과_리포트_작성.ipynb)

### C4. 노트북 실행 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `NameError: name 'X' is not defined` | 앞 셀을 실행하지 않고 중간 셀만 실행했다 | 메뉴 **Kernel → Restart Kernel and Run All Cells**로 위에서부터 다시 실행 |
| 결과가 코드와 맞지 않는다 | 셀을 순서 없이 여러 번 실행해서 변수 상태가 꼬였다 | 커널 재시작 후 처음부터 실행 |
| `ModuleNotFoundError: numpy` 등 | 노트북 커널이 가상환경(`.venv`)이 아니다 | 오른쪽 위 커널 선택에서 `.venv` 선택. JupyterLab은 저장소 루트에서 `uv run jupyter lab`으로 실행 |
| `Kernel died` / 커널이 갑자기 재시작 | 대부분 메모리 부족 | [C13](#c13-메모리-부족) 참고. 다른 노트북의 커널을 종료한다 |
| `nbconvert` 실행 중 `CellTimeoutError` | 셀 하나가 기본 시간 제한을 넘었다 (형태소 분석, 큰 행렬) | `--ExecutePreprocessor.timeout=900`처럼 시간 제한을 늘린다 |
| 환경 변수(`JAVA_HOME`)를 바꿨는데 반영이 안 된다 | 이미 실행 중인 커널은 예전 환경을 쓴다 | JupyterLab을 종료하고 터미널에서 다시 실행 |

**관련 장:** [0장 0.1](00_자료_사용법과_과제_개요.ipynb), [1장](01_개발_환경과_프로젝트_구조.ipynb)

---

## 데이터

### C5. 20 Newsgroups 첫 다운로드

**증상:** `fetch_20newsgroups`를 처음 실행할 때 오래 걸리거나, 인터넷이 없는 환경에서 `URLError`/`HTTPError`가 납니다.

**원인:** 처음 한 번은 인터넷에서 데이터(약 14MB)를 받아 홈 폴더의 `scikit_learn_data/`에 저장합니다. 다음부터는 저장된 파일을 씁니다.

**해결**
- 인터넷이 되는 곳에서 한 번 실행해 캐시를 만들어 둡니다.
- 다운로드가 중간에 끊겨 파일이 깨졌다면 `~/scikit_learn_data/` 안의 20 Newsgroups 파일을 지우고 다시 실행합니다.
- 회사·학교 네트워크에서 막힌다면 프록시 설정을 확인합니다.

**관련 장:** [5장 5.2](05_Pandas_기초.ipynb), [7장 7.3](07_NLP_개요와_데이터셋.ipynb)

### C6. KLUE-YNAT 다운로드 실패

**증상:** `urllib.error.URLError` 또는 `HTTPError: 404`가 납니다. 파일은 생겼는데 `pd.read_json`에서 `ValueError`가 나기도 합니다.

**원인**
- 네트워크 문제로 GitHub에서 파일(약 44MB)을 받지 못했습니다.
- 다운로드가 중간에 끊겨 **불완전한 파일**이 남았습니다. 코드는 "파일이 있으면 다시 받지 않는다"이므로 깨진 파일을 계속 읽습니다.

**해결**
1. `data/raw/ynat-v1.1_train.json`을 지우고 다시 실행합니다.
2. 계속 실패하면 브라우저로 KLUE GitHub 저장소(`klue_benchmark/ynat-v1.1/ynat-v1.1_train.json`)에서 직접 받아 같은 위치에 둡니다.
3. 파일 크기가 약 44MB인지 확인합니다.

**관련 장:** [7장 7.4](07_NLP_개요와_데이터셋.ipynb)

### C7. read_csv 구분자와 인코딩

| 증상 | 원인 | 해결 |
|---|---|---|
| 열이 하나로 뭉쳐 나온다 | 탭으로 구분된 파일(TSV)을 기본 구분자(쉼표)로 읽었다 | `pd.read_csv(path, sep="\t")`. 과제 문서 예시의 `sep='    '`(공백)는 탭이 잘못 옮겨진 것 |
| `ParserError: Error tokenizing data` | 구분자가 틀렸거나 본문에 구분자 문자가 섞였다 | 구분자 확인, 필요하면 `quoting` 옵션 조정 |
| `UnicodeDecodeError` | 파일 인코딩과 읽을 때 인코딩이 다르다 (Windows의 cp949 등) | `encoding="utf-8"`을 명시. 엑셀용 CSV는 `utf-8-sig`로 저장 |
| 빈 문자열이 `NaN`이 되어 있다 | 파일에서는 빈칸과 `None`이 구별되지 않는다 | 결측(`isna`)과 공백뿐인 문서(`str.strip() == ""`)를 따로 검사 |

**관련 장:** [5장 5.2, 5.5](05_Pandas_기초.ipynb)

### C8. groupby().sample에서 n이 그룹보다 큼

**증상**
```
ValueError: Cannot take a larger sample than population when 'replace=False'
```

**원인:** `df.groupby("label").sample(n=1000)`에서 어떤 카테고리의 문서가 1,000건보다 적습니다.

**해결:** 먼저 `value_counts()`로 가장 작은 카테고리 크기를 확인하고, `n=min(n, 최소 크기)`로 맞춥니다. KLUE-YNAT 원본은 가장 작은 주제(사회)도 5,133건이라 1,000건 샘플링은 문제없습니다.

**관련 장:** [5장 5.4](05_Pandas_기초.ipynb), [7장 7.4](07_NLP_개요와_데이터셋.ipynb)

### C9. Okt 캐시가 갱신되지 않음

**증상:** 품사 목록이나 정제 규칙을 바꿨는데 토큰 결과가 그대로입니다.

**원인:** 형태소 분석 결과를 파일로 캐시해 두고 재사용하는데, 캐시 파일 이름이 설정과 상관없이 같아서 예전 결과를 읽고 있습니다.

**해결**
- 캐시 파일 이름에 **설정의 지문(해시)**을 넣습니다. 품사 목록, 정제 정규식, Okt 옵션을 JSON으로 묶어 해시하면 설정이 바뀔 때 자동으로 다른 파일을 쓰게 됩니다.
- 급하면 `data/cache/` 폴더를 지우고 다시 실행합니다.
- 불용어 제거는 캐시 **뒤에** 적용하면, 불용어 목록을 바꿀 때 캐시를 다시 만들 필요가 없습니다.

**관련 장:** [9장 9.8](09_토큰화.ipynb)

---

## NumPy 계산

### C10. 브로드캐스팅 ValueError

**증상**
```
ValueError: operands could not be broadcast together with shapes (5,15) (5,)
```

**원인:** `(문서 수, 어휘 수)` 행렬을 `(문서 수,)` 벡터로 나누려 했습니다. 브로드캐스팅은 **오른쪽 축부터** 맞추므로 15와 5를 비교하게 되어 실패합니다.

**해결:** 행마다 나누려면 벡터를 `(문서 수, 1)`로 만듭니다.
```python
X / X.sum(axis=1, keepdims=True)   # 또는 X / X.sum(axis=1)[:, None]
```
열마다 곱하는 `X * idf`는 `(n, V) × (V,)`라서 그대로 됩니다.

**관련 장:** [4장 4.5](04_NumPy_기초.ipynb)

### C11. 0으로 나누기와 nan, log(0) 경고

**증상**
```
RuntimeWarning: invalid value encountered in divide
RuntimeWarning: divide by zero encountered in log
```
또는 검색 점수가 모두 `nan`으로 나옵니다.

**원인**
- 전처리 후 단어가 하나도 남지 않은 **빈 문서**나, 불용어·OOV만 있는 **쿼리**는 노름이 0이라 정규화에서 `0/0 = nan`이 생깁니다. "the and of"를 공식 그대로 계산하면 4,646개 문서 점수가 모두 nan이 됩니다.
- `np.where(X > 0, 1 + np.log(X), 0)`은 `np.where`가 **양쪽 값을 모두 먼저 계산**해서, 0인 칸에서도 `log(0)`을 계산합니다.

**해결**
```python
norms = np.linalg.norm(X, axis=1, keepdims=True)
X = X / np.where(norms == 0, 1, norms)       # 노름이 0이면 1로 나눠 0 벡터로 둔다

tf = np.zeros_like(X, dtype=float)
mask = X > 0
tf[mask] = 1 + np.log(X[mask])               # 0이 아닌 칸만 계산한다
```
검색에서는 쿼리 벡터가 0이면 "일치하는 문서 없음"을 돌려줍니다.

**관련 장:** [4장 4.6~4.7](04_NumPy_기초.ipynb), [13장](13_NumPy로_TF-IDF_구현.ipynb), [15장](15_코사인_유사도.ipynb)

### C12. X[rows, cols] += 1로 센 개수가 모자람

**증상:** 에러는 없는데, 같은 단어가 여러 번 나온 문서의 개수가 1로 나옵니다.

**원인:** 팬시 인덱싱의 `+=`는 같은 위치가 여러 번 나와도 **한 번만** 더합니다.

**해결:** 중복까지 누적하는 `np.add.at`을 씁니다.
```python
np.add.at(X, (rows, cols), 1)
```

**관련 장:** [4장 4.2](04_NumPy_기초.ipynb), [11장](11_Bag_of_Words와_문서-단어_행렬.ipynb)

### C13. 메모리 부족

**증상:** `MemoryError`, 또는 노트북이 `Kernel died`로 재시작되고 컴퓨터가 느려집니다.

**원인:** 문서-단어 행렬을 NumPy 밀집 행렬로 만들면 0까지 모두 8바이트씩 저장합니다. 크기는 `문서 수 × 어휘 수 × 8`바이트입니다.

| 영어 min_df | 어휘 수 | 밀집 행렬 |
|---|---|---|
| 1 | 35,004 | 1,241MB |
| 2 | 18,628 | 660MB |
| 3 | 13,387 | 475MB |

**해결**
- `min_df`를 올려 드문 단어를 어휘에서 뺍니다. 이 과제는 영어 3, 한국어 2입니다.
- 큰 행렬을 만드는 노트북을 **동시에 여러 개 실행하지 않습니다.** 다 쓴 노트북의 커널은 종료합니다.
- 중간 결과를 복사하지 않도록 `x *= idf`처럼 제자리 연산을 씁니다.
- `float32`로 바꾸면 메모리는 절반이 되지만 정밀도가 약 1e-7이라 1e-6 검증에는 `float64`를 유지합니다.

**관련 장:** [4장 4.10](04_NumPy_기초.ipynb), [11장](11_Bag_of_Words와_문서-단어_행렬.ipynb)

---

## 검증과 모델

### C14. sklearn과 결과가 다름

**증상:** 검증 로그가 FAIL이거나, shape(어휘 수)부터 다릅니다.

먼저 **어휘 → DF → IDF → 정규화** 순서로 하나씩 비교해 어느 단계에서 어긋나는지 찾습니다. 가장 크게 어긋난 칸은 `np.unravel_index(err.argmax(), err.shape)`로 찾아 어느 문서, 어느 단어인지 확인합니다.

| 증상 | 원인 | 해결 |
|---|---|---|
| 어휘에 `the`, `in` 같은 단어가 더 있다 | sklearn이 자체 토큰화(`token_pattern`)와 소문자 변환을 했다 | `analyzer=`에 "전처리된 토큰을 그대로 돌려주는 함수"를 주고 `lowercase=False` |
| `token_pattern will not be used` 경고 | `tokenizer=`를 줄 때 `token_pattern`이 남아 있다 | `token_pattern=None`을 함께 주거나 `analyzer=`를 쓴다 |
| 어휘 수가 다르다 | `min_df`, `max_df`가 서로 다르다 | 두 구현에 같은 값을 준다 (영어 3, 한국어 2) |
| 어휘는 같은데 오차가 크다 (0.9 수준) | 내 어휘가 정렬되지 않아 열 순서가 다르다 | 어휘를 `sorted()`로 정렬하거나 `get_feature_names_out()`으로 열을 맞춘다 |
| IDF 오차가 정확히 1.0 | IDF 끝의 `+1`을 빠뜨렸다 | `np.log((1 + n) / (1 + df)) + 1` |
| `smooth_idf=False`에서만 다르다 | non-smooth IDF 식이 다르다 | `np.log(n / df) + 1` |
| `norm="l2"`에서만 다르다 | 정규화를 L1 노름이나 다른 값으로 했다 | 행마다 L2 노름 `sqrt(sum(x**2))`로 나눈다 |
| 오차가 1e-7~1e-8 수준 | 한쪽이 float32로 계산했다 | 두 쪽 모두 `float64` (`dtype=np.float64`) |
| `AttributeError: toarray` / 비교가 안 된다 | sklearn 결과는 희소 행렬(`csr_matrix`)이다 | `.toarray()`로 바꿔 비교한다 |

14장의 디버깅 실습에서 실제로 만든 버그 결과입니다.
- `+1`을 빠뜨리면 IDF 오차가 1.0이 나오고, 가장 크게 어긋난 칸은 흔한 단어 `time`입니다.
- 어휘를 정렬하지 않고 열 순서대로 비교하면 최대 오차가 0.901입니다.
- L1 노름으로 나누는 버그는 `norm='l2'`에서만 FAIL이 납니다.

**관련 장:** [14장](14_sklearn으로_검증하기.ipynb)

### C15. np.allclose는 통과했는데 기준이 다름

**증상:** 과제는 "오차 1e-6 이내"를 요구하는데, `np.allclose(A, B)`가 True라고 해서 그 기준을 만족한다고 보고했습니다.

**원인:** `np.allclose`의 기본값은 `rtol=1e-05, atol=1e-08`이고, 판정식은 $\lvert a - b \rvert \le \text{atol} + \text{rtol} \times \lvert b \rvert$입니다. 상대 오차가 섞인 **다른 기준**입니다.

**해결:** 최대 절대 오차를 직접 계산해 기록합니다.
```python
max_err = np.abs(A - B).max()
passed = max_err <= 1e-6
# 또는 np.allclose(A, B, rtol=0, atol=1e-6)
```
같은 이유로 실수는 `==`로 비교하지 않습니다 (`0.1 + 0.2 == 0.3`은 False).

**관련 장:** [2장 2.6](02_과제에_필요한_파이썬.ipynb), [4장 4.9](04_NumPy_기초.ipynb), [14장](14_sklearn으로_검증하기.ipynb)

### C16. LogisticRegression ConvergenceWarning

**증상**
```
ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
```

**원인:** 최적화가 최대 반복 횟수(`max_iter`, 기본 100) 안에 끝나지 않았습니다. 특성이 많거나 규제가 약하면(C가 크면) 더 많은 반복이 필요합니다.

**해결**
- `LogisticRegression(max_iter=2000)`처럼 반복 횟수를 늘립니다. 이 과제의 영어 데이터는 실제로 45번 만에 수렴했으므로 2000이면 충분합니다.
- 경고를 무시하면 덜 학습된 모델로 평가하게 됩니다. 18장에서 `max_iter=5`로 일부러 줄이면 정확도가 0.873으로, 충분히 학습한 0.900보다 낮게 나옵니다.
- TF-IDF처럼 정규화된 입력은 스케일 문제가 거의 없습니다. 원시 개수를 그대로 넣었다면 정규화를 먼저 확인합니다.

**관련 장:** [18장](18_선형_분류기.ipynb)

---

## 그림

### C17. 한글 글꼴 깨짐과 마이너스 기호

**증상:** 그래프의 한글이 네모(□)로 나오고 경고가 쏟아집니다.
```
UserWarning: Glyph 51221 (\N{HANGUL SYLLABLE ...}) missing from font(s) DejaVu Sans.
```
한글 글꼴을 설정한 뒤에는 음수의 `-` 기호가 깨집니다.

**원인:** Matplotlib 기본 글꼴(DejaVu Sans)에는 한글이 없습니다. 한글 글꼴에는 유니코드 마이너스 기호(−)가 없는 경우가 많습니다. 6장에서 글꼴 설정 전에 그린 그림 하나에서 경고가 38개 나왔습니다.

**해결**
```python
plt.rcParams["font.family"] = "AppleGothic"   # Windows: "Malgun Gothic", Linux: "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
```
- 운영체제마다 글꼴이 다르므로 6장의 `set_korean_font()`처럼 설치된 글꼴을 찾아 설정합니다.
- 글꼴을 새로 설치했는데도 안 보이면 `matplotlib.get_cachedir()` 폴더를 지우고 커널을 재시작합니다.
- 제출용 그림은 레이블을 영어로 바꾸면(`정치` → `Politics`) 평가자 컴퓨터의 글꼴과 상관없이 깨지지 않습니다.
- seaborn 히트맵의 세로축 글자가 누워 있으면 `ax.tick_params(axis="y", rotation=0)`을 줍니다.

**관련 장:** [6장 6.4](06_Matplotlib_Seaborn_최소한.ipynb), [19장](19_평가_지표와_혼동_행렬.ipynb)
