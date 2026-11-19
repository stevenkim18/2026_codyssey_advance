"""전처리 규칙 테스트 (데이터 파일 없이 돈다)."""

import pytest
from src.data import mask_pii
from src.preprocess import EnglishPreprocessor, KoreanPreprocessor


def test_mask_pii_hides_email_url_phone():
    text = "mail me at john.doe@cs.example.edu or see http://x.org/a, call 02-123-4567"
    masked = mask_pii(text)
    assert "john.doe" not in masked and "x.org" not in masked and "4567" not in masked


def test_english_pipeline():
    prep = EnglishPreprocessor()
    tokens = prep("NASA's Shuttle LAUNCHED in 1993!!! I don't think it would fly <email> x")
    assert tokens == ["nasa", "shuttle", "launched", "fly"]  # 소문자, 숫자·특수문자·불용어·한 글자 제거


def test_english_drops_long_garbage_tokens():
    prep = EnglishPreprocessor()
    assert prep("m" * 25 + " graphics") == ["graphics"]


def test_domain_stopwords_toggle():
    assert "would" not in EnglishPreprocessor()("would orbit")
    assert "would" in EnglishPreprocessor(use_domain_stopwords=False).clean("would orbit")


def test_korean_whitespace_keeps_eojeol():
    prep = KoreanPreprocessor("whitespace")
    assert prep("삼성전자가 3분기 실적…발표!") == ["삼성전자가", "3분기", "실적", "발표"]


def test_korean_clean_keeps_hanja_and_alpha():
    assert KoreanPreprocessor.clean("北 미사일·KT 5G!") == "北 미사일 KT 5G"


def test_korean_okt_pos_filter():
    pytest.importorskip("konlpy")
    try:
        tokens = KoreanPreprocessor("okt")("北 미사일을 발사했다")
    except (OSError, RuntimeError, ValueError) as exc:  # Java(JVM)가 없는 환경
        pytest.skip(f"Okt 사용 불가: {exc}")
    assert "北" in tokens and "미사일" in tokens and "발사" in tokens
    assert "을" not in tokens and "하다" not in tokens  # 조사 제거, 불용어 '하다' 제거


def test_drop_duplicates_keeps_one_and_drops_conflicting_labels():
    import pandas as pd
    from src.data import drop_duplicates

    df = pd.DataFrame(
        {
            "text": ["same  text", "same text", "cross post", "cross  post", "unique"],
            "label": ["a", "a", "a", "b", "b"],
        }
    )
    out, stats = drop_duplicates(df)
    assert out["text"].tolist() == ["same  text", "unique"]
    assert stats == {"conflict_removed": 2, "duplicate_removed": 1}


def test_okt_cache_signature_changes_with_pos_filter(monkeypatch):
    pytest.importorskip("konlpy")
    import src.preprocess as pp

    before = pp.okt_cache_signature()
    monkeypatch.setattr(pp, "KO_KEEP_POS", pp.KO_KEEP_POS | {"Number"})
    assert pp.okt_cache_signature() != before
