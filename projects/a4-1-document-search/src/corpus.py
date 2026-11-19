"""데이터 로드 + 전처리를 한 번에: 문서마다 토큰 리스트를 붙이고, 토큰이 너무 적은 문서를 뺀다.

Okt 형태소 분석은 7,000건에 수십 초가 걸려서, 불용어를 거르기 전의 형태소 결과를
`data/cache/`에 JSON으로 저장해 두고 다음 실행부터 재사용한다 (git 제외).
파일 이름에 품사 목록·정제 규칙·Okt 옵션의 지문을 넣어서, 규칙을 바꾸면 새로 분석한다.
"""

import json

import pandas as pd

from src.data import DATA_DIR, load_dataset
from src.preprocess import KoreanPreprocessor, get_preprocessor, okt_cache_signature

CACHE_DIR = DATA_DIR / "cache"
MIN_TOKENS = {"en": 5, "ko": 2}
# 어휘 사전에 넣을 최소 문서 빈도. 한 문서에만 나온 단어는 오타·고유한 잡음이 많고 검색·분류에 쓸모가 없다.
# 영어는 본문이 길어 min_df=2 로도 어휘가 1.9만 개(밀집 행렬 약 700MB)라 3으로 올렸다 (1.3만 개, 약 500MB).
# 한국어는 제목이 짧아 단어가 드물게 나오므로 2로 둔다.
MIN_DF = {"en": 3, "ko": 2}
OUTPUT_ROOT = DATA_DIR.parent / "outputs"


def output_dir(lang: str, tokenizer: str = "okt"):
    """결과 폴더: outputs/en, outputs/ko-okt, outputs/ko-whitespace"""
    path = OUTPUT_ROOT / (lang if lang == "en" else f"{lang}-{tokenizer}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tokenize_all(texts: list[str], prep) -> list[list[str]]:
    if not (isinstance(prep, KoreanPreprocessor) and prep.tokenizer == "okt"):
        return [prep(t) for t in texts]

    path = CACHE_DIR / f"okt_morphs_{okt_cache_signature()}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [t for t in texts if t not in cache]
    if missing:
        print(f"Okt 형태소 분석 {len(missing):,}건 (처음 한 번만, 결과는 {path.name}에 저장)")
        for t in missing:
            cache[t] = prep.tokenize(t)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    # 불용어·숫자 제거는 캐시 뒤에 적용해서, 불용어 목록을 바꿔도 캐시를 다시 만들 필요가 없게 한다.
    return [prep.filter(cache[t]) for t in texts]


def load_corpus(lang: str, tokenizer: str = "okt", use_stopwords: bool = True) -> pd.DataFrame:
    """열: doc_id, text, label, tokens. 전처리 후 토큰이 MIN_TOKENS개 미만인 문서는 뺀다."""
    df = load_dataset(lang)
    prep = get_preprocessor(lang, tokenizer=tokenizer, use_stopwords=use_stopwords)
    df["tokens"] = _tokenize_all(df["text"].tolist(), prep)
    keep = df["tokens"].map(len) >= MIN_TOKENS[lang]
    return df[keep].reset_index(drop=True)
