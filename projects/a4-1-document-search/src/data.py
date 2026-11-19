"""데이터 로드 (A4-1): 영어 20 Newsgroups 5개 카테고리, 한국어 KLUE-YNAT 7개 주제.

- 영어: scikit-learn 내장 20 Newsgroups. 헤더·서명·인용문을 지운다 (`remove=...`).
  헤더에는 작성자 이메일·실명과 뉴스그룹 이름이 들어 있어서, 남겨 두면 개인정보가 섞이고
  뉴스그룹 이름이 곧 정답이라 레이블이 새어 나간다.
- 한국어: KLUE-YNAT (연합뉴스 기사 제목, CC BY-SA 4.0). 학습 파일 45,678건에서
  주제별로 같은 수를 무작위 추출한다. 파일이 없으면 GitHub 공개 저장소에서 내려받는다.

두 데이터 모두 본문에 남은 이메일·URL·전화번호를 로드 단계에서 가리고, 본문이 같은 중복 문서를 지운다.
가린 텍스트가 검색 결과 스니펫으로 화면에 그대로 나오기 때문이다.

실행: `uv run python -m src.data` (프로젝트 폴더에서) → 두 데이터의 문서 수와 분포 출력
"""

import re
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

RANDOM_STATE = 42

EN_CATEGORIES = [
    "comp.graphics",
    "comp.windows.x",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.space",
]

YNAT_URL = (
    "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/"
    "klue_benchmark/ynat-v1.1/ynat-v1.1_train.json"
)
YNAT_PATH = RAW_DIR / "ynat-v1.1_train.json"
KO_PER_CLASS = 1000

# 그래프는 한글 글꼴 문제를 피하려고 영어 이름을 쓴다.
KO_LABEL_EN = {
    "IT과학": "IT/Science",
    "경제": "Economy",
    "사회": "Society",
    "생활문화": "Life/Culture",
    "세계": "World",
    "스포츠": "Sports",
    "정치": "Politics",
}

_EMAIL = re.compile(r"\S+@\S+\.\S+")
_URL = re.compile(r"(https?://|www\.)\S+")
_PHONE = re.compile(r"\b\d{2,4}[-.\s]\d{3,4}[-.\s]\d{4}\b")


def mask_pii(text: str) -> str:
    """이메일·URL·전화번호를 자리표시자로 바꾼다. 자리표시자는 전처리에서 특수문자로 지워진다."""
    text = _EMAIL.sub(" <email> ", text)
    text = _URL.sub(" <url> ", text)
    return _PHONE.sub(" <phone> ", text)


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """본문이 같은 문서(공백 차이 무시)를 하나만 남긴다. 같은 본문에 레이블이 다르면 모두 뺀다.

    20 Newsgroups에는 한 글을 여러 게시판에 동시에 올린 크로스포스팅이 있다. 그대로 두면
    같은 글이 Train과 Test에 함께 들어가(누수) 점수가 부풀고, 레이블이 다른 사본은 정답이 모호하다.
    """
    norm = df["text"].str.split().str.join(" ")
    conflict = df.groupby(norm)["label"].transform("nunique") > 1
    duplicate = norm.duplicated(keep="first") & ~conflict
    stats = {"conflict_removed": int(conflict.sum()), "duplicate_removed": int(duplicate.sum())}
    return df[~conflict & ~duplicate], stats


def _finalize(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """공백만 있는 문서(결측과 같다)와 중복 문서를 지우고, 문서 ID를 0부터 다시 붙인다."""
    df = df.assign(text=df["text"].map(mask_pii).str.strip())
    n_raw = len(df)
    df = df[df["text"].str.len() > 0]
    n_empty = n_raw - len(df)
    df, dup = drop_duplicates(df)
    df = df.reset_index(drop=True)
    df.insert(0, "doc_id", [f"{prefix}-{i:05d}" for i in range(len(df))])
    df.attrs["cleaning"] = {"raw": n_raw, "empty_removed": n_empty, **dup}
    return df


def load_english() -> pd.DataFrame:
    """20 Newsgroups 5개 카테고리 → 열: doc_id, text, label."""
    from sklearn.datasets import fetch_20newsgroups

    bunch = fetch_20newsgroups(
        subset="all",
        categories=EN_CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    labels = [bunch.target_names[t] for t in bunch.target]
    return _finalize(pd.DataFrame({"text": bunch.data, "label": labels}), "en")


def download_ynat(path: Path = YNAT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"KLUE-YNAT 내려받는 중: {YNAT_URL}")
        urllib.request.urlretrieve(YNAT_URL, path)
    return path


def load_korean(per_class: int = KO_PER_CLASS) -> pd.DataFrame:
    """KLUE-YNAT 주제별 per_class건 → 열: doc_id, text, label, guid.

    레이블은 주석자 3명이 붙인 `label`을 쓴다 (언론사가 붙인 `predefined_news_category`와 17% 다르다).
    """
    raw = pd.read_json(download_ynat())
    raw = raw.rename(columns={"title": "text"})[["guid", "text", "label"]].dropna()
    sampled = (
        raw.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=RANDOM_STATE)
        .sample(frac=1.0, random_state=RANDOM_STATE)  # 주제별로 몰려 있지 않게 섞는다
    )
    return _finalize(sampled.reset_index(drop=True), "ko")[["doc_id", "text", "label", "guid"]]


def load_dataset(lang: str) -> pd.DataFrame:
    if lang == "en":
        return load_english()
    if lang == "ko":
        return load_korean()
    raise ValueError(f"lang은 'en' 또는 'ko' 여야 합니다: {lang}")


if __name__ == "__main__":
    for lang in ("en", "ko"):
        df = load_dataset(lang)
        c = df.attrs["cleaning"]
        print(f"=== {lang}: 문서 {len(df):,}건, 카테고리 {df['label'].nunique()}개 ===")
        print(
            f"원본 {c['raw']:,}건 → 빈 문서 {c['empty_removed']}건, 중복 사본 {c['duplicate_removed']}건, "
            f"레이블이 엇갈린 중복 {c['conflict_removed']}건 제거"
        )
        print(df["label"].value_counts().sort_index().to_string())
        print(f"평균 길이(글자): {df['text'].str.len().mean():.1f}\n")
