"""전처리 파이프라인 (A4-1): 정제 → 토큰화 → 불용어 제거.

영어 (EnglishPreprocessor)
    1. 소문자 변환.
    2. 정제: 알파벳과 단어 안의 아포스트로피(don't, nasa's)만 남기고 숫자·특수문자는 공백으로 바꾼다.
       숫자는 연도, 버전, 좌표, 스코어처럼 문서마다 제각각이라 주제 신호가 약하고 어휘만 키운다.
    3. 공백 기반 토큰화 (정규식 `[a-z]+(?:'[a-z]+)*`).
    4. 불용어 제거: NLTK english 198개 + 데이터에서 찾은 도메인 불용어 (DOMAIN_STOPWORDS_EN).
       NLTK 목록에 "don't" 같은 축약형이 있으므로 아포스트로피를 지우기 전에 거른다.
    5. 소유격 's 제거, 남은 아포스트로피 삭제 (nasa's → nasa, can't → cant).
    6. 길이 2 미만(a, x 같은 한 글자)과 20 초과(uuencode로 인코딩된 그림 파일 조각) 토큰 제거.
    표제어 추출·어간 추출은 하지 않는다: 검색 결과와 분류 근거를 사람이 읽을 수 있는 단어로 남기기 위해서다.

한국어 (KoreanPreprocessor)
    1. 정제: 한글, 한자, 영문, 숫자, 공백만 남긴다. 한자를 남기는 이유는 뉴스 제목에서
       美·中·北·與·野·朴 같은 한 글자 한자가 정치·세계 주제의 강한 신호이기 때문이다 (제목 15%에 한자).
    2. 토큰화 (둘 중 선택)
       - "okt": KoNLPy Okt 형태소 분석 (norm=True, stem=True → 동사·형용사를 원형으로).
         명사, 동사, 형용사, 영문(Alpha), 한자(Foreign 중 한자)만 남긴다.
         조사·어미·구두점·숫자·접미사·관형사는 버린다.
       - "whitespace": 띄어쓰기로만 자른다. "삼성전자가", "삼성전자는"이 서로 다른 단어가 된다.
    3. 영문 소문자 변환, 불용어 제거 (STOPWORDS_KO), 숫자만 있는 토큰 제거.

불용어 목록을 고른 근거는 `uv run python -m src.stopword_analysis`의 출력(문서 빈도 + 카테고리 간 분포)이다.
"""

import re
from functools import lru_cache

# 데이터에서 찾은 영어 도메인 불용어: 문서 빈도가 높고 5개 카테고리에 고르게 퍼진 단어 (src/stopword_analysis.py).
# 주제를 가르지 못하면서 모든 문서 벡터에 조금씩 들어가 코사인 유사도를 흐린다.
# 선택 기준: DF 5% 이상, 균등도 0.95 이상이면서 조동사·일반 동사·부사·대명사처럼 내용이 없는 단어.
# 같은 기준을 넘었어도 time, new, people, first, two, good, better, back 은 검색어로 쓰일 수 있는 내용어라 남겼다.
DOMAIN_STOPWORDS_EN = frozenset(
    {
        "would", "could", "might", "may", "also", "like", "get", "got", "go", "going", "know", "think",
        "see", "say", "make", "want", "look", "one", "well", "much", "many", "even", "way", "something",
        "anyone", "since", "still", "really", "right", "another", "cant",
    }
)  # fmt: skip

# 한국어 불용어: 기능어 성격의 용언·의존명사와, 기사 제목 형식어 (src/stopword_analysis.py 근거).
# - 기능 용언: 하다 되다 있다 없다 이다 않다 (Okt stem=True 로 원형이 된 것)
# - 의존명사·나열어: 등 것 수 중 및 때 후 내 듯 제 개
# - Okt 가 조사를 명사로 잘못 자른 조각: 의 로
# - 기사 제목 형식어: 종합 ("…출격종합"처럼 기사 버전 표시)
# '한국'은 균등도 0.97 로 기준을 넘지만 검색어로 쓰이는 내용어라 남겼다.
STOPWORDS_KO = frozenset(
    {
        "하다", "되다", "있다", "없다", "이다", "않다",
        "등", "것", "수", "중", "및", "때", "후", "내", "듯", "제", "개",
        "의", "로",
        "종합",
    }
)  # fmt: skip

_PLACEHOLDER = re.compile(r"<(?:email|url|phone)>")  # src/data.py 의 mask_pii 가 넣은 자리표시자
_EN_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")
_KO_CLEAN = re.compile(r"[^0-9a-zA-Z가-힣一-鿿\s]")
_HANJA = re.compile(r"^[一-鿿]+$")
_DIGITS = re.compile(r"^\d+$")

KO_KEEP_POS = frozenset({"Noun", "Verb", "Adjective", "Alpha"})
OKT_OPTIONS = {"norm": True, "stem": True}


def okt_cache_signature() -> str:
    """Okt 토큰화 결과를 바꾸는 설정의 지문. 설정이 바뀌면 캐시 파일 이름이 바뀌어 다시 분석한다."""
    import hashlib
    import json

    from konlpy import __version__ as konlpy_version

    config = {
        "keep_pos": sorted(KO_KEEP_POS),
        "okt": OKT_OPTIONS,
        "patterns": [_PLACEHOLDER.pattern, _KO_CLEAN.pattern, _HANJA.pattern],
        "konlpy": konlpy_version,
    }
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


@lru_cache(maxsize=1)
def nltk_english_stopwords() -> frozenset[str]:
    import nltk

    try:
        from nltk.corpus import stopwords

        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        words = stopwords.words("english")
    return frozenset(words)


class EnglishPreprocessor:
    name = "en"

    def __init__(self, use_domain_stopwords: bool = True, min_len: int = 2, max_len: int = 20):
        self.stopwords = nltk_english_stopwords() | (DOMAIN_STOPWORDS_EN if use_domain_stopwords else frozenset())
        self.min_len = min_len
        self.max_len = max_len

    @staticmethod
    def clean(text: str) -> str:
        text = _PLACEHOLDER.sub(" ", text.lower()).replace("’", "'")
        return " ".join(_EN_TOKEN.findall(text))

    def __call__(self, text: str) -> list[str]:
        tokens = []
        for tok in self.clean(text).split():
            if tok in self.stopwords:
                continue
            tok = tok.removesuffix("'s").replace("'", "")
            if self.min_len <= len(tok) <= self.max_len and tok not in self.stopwords:
                tokens.append(tok)
        return tokens


@lru_cache(maxsize=1)
def _okt():
    from konlpy.tag import Okt

    return Okt()


class KoreanPreprocessor:
    def __init__(self, tokenizer: str = "okt", use_stopwords: bool = True):
        if tokenizer not in ("okt", "whitespace"):
            raise ValueError(f"tokenizer는 'okt' 또는 'whitespace': {tokenizer}")
        self.tokenizer = tokenizer
        self.name = f"ko-{tokenizer}"
        self.stopwords = STOPWORDS_KO if use_stopwords else frozenset()

    @staticmethod
    def clean(text: str) -> str:
        text = _PLACEHOLDER.sub(" ", text)
        return " ".join(_KO_CLEAN.sub(" ", text).split())

    def tokenize(self, text: str) -> list[str]:
        cleaned = self.clean(text)
        if self.tokenizer == "whitespace":
            return cleaned.split()
        tokens = []
        for word, pos in _okt().pos(cleaned, **OKT_OPTIONS):
            if pos in KO_KEEP_POS or (pos == "Foreign" and _HANJA.match(word)):
                tokens.append(word)
        return tokens

    def filter(self, tokens: list[str]) -> list[str]:
        out = []
        for tok in tokens:
            tok = tok.lower()
            if tok not in self.stopwords and not _DIGITS.match(tok):
                out.append(tok)
        return out

    def __call__(self, text: str) -> list[str]:
        return self.filter(self.tokenize(text))


def get_preprocessor(lang: str, tokenizer: str = "okt", use_stopwords: bool = True):
    if lang == "en":
        return EnglishPreprocessor(use_domain_stopwords=use_stopwords)
    if lang == "ko":
        return KoreanPreprocessor(tokenizer=tokenizer, use_stopwords=use_stopwords)
    raise ValueError(f"lang은 'en' 또는 'ko': {lang}")
