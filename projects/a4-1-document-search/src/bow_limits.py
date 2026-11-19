"""BoW(TF-IDF) 한계를 실제 벡터로 보여 준다. 말뭉치로 학습한 어휘·IDF를 그대로 쓴다.

    1. 어순 무시: 단어 순서만 다른 두 문장 → 같은 벡터 (코사인 1.0)
    2. 동음이의어: 뜻이 다른 같은 단어 → 서로 비슷하다고 판단
    3. 동의어·표기 차이: 뜻이 같은 다른 단어 → 겹치는 차원이 없어 유사도 0
    4. 부정: "좋다" / "좋지 않다" → 불용어·형태소 처리 후 거의 같은 벡터
    5. 어휘 밖 단어(OOV): 말뭉치에 없던 단어 → 0 벡터

실행: `uv run python -m src.bow_limits`
"""

import numpy as np

from src.search import cosine_similarity, get_engine

CASES = [
    ("어순 무시", "en", "the dog bit the man", "the man bit the dog"),
    ("어순 무시", "ko", "한국이 일본을 이겼다", "일본이 한국을 이겼다"),
    ("동음이의어", "ko", "배가 고프다", "배를 타고 바다로"),
    ("동음이의어", "en", "apple released a new mac", "apple pie recipe"),
    ("동의어·표기 차이", "ko", "북한 미사일", "北 미사일"),
    ("동의어·표기 차이", "ko", "북한", "北"),
    ("동의어·표기 차이", "en", "car", "automobile"),
    ("동의어·표기 차이", "en", "nasa shuttle launch", "space agency orbiter liftoff"),
    ("부정", "ko", "실적이 좋다", "실적이 좋지 않다"),
    ("부정", "en", "the new card works well", "the new card does not work"),
    ("어휘 밖 단어", "ko", "코로나 백신", "백신 접종"),
]


def pair_report(kind: str, lang: str, a: str, b: str) -> str:
    eng = get_engine(lang)
    (va, ta), (vb, tb) = eng.query_vector(a), eng.query_vector(b)
    cos = float(cosine_similarity(va[None, :], vb)[0]) if va.any() and vb.any() else 0.0
    oov = [t for t in ta + tb if t not in eng.vec.vocabulary_]
    lines = [
        f"[{kind}] ({lang})",
        f'  A: "{a}" → {ta}',
        f'  B: "{b}" → {tb}',
        f"  코사인 유사도 = {cos:.3f}" + ("  (벡터 완전히 동일)" if va.any() and np.allclose(va, vb) else ""),
    ]
    if oov:
        lines.append(f"  어휘 사전에 없어 버려진 단어: {sorted(set(oov))}")
    return "\n".join(lines)


def main() -> None:
    print("=== BoW 한계 분석 (말뭉치로 학습한 TF-IDF 어휘·IDF 사용) ===\n")
    for case in CASES:
        print(pair_report(*case), "\n")


if __name__ == "__main__":
    main()
