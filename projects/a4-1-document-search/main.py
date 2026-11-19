"""A4-1 문서 검색·분류 CLI.

예시 (프로젝트 폴더에서):
    uv run python main.py --query "space shuttle launch" --topk 5
    uv run python main.py --lang ko --query "삼성전자 반도체 실적" --explain
    uv run python main.py --mode verify --lang ko
    uv run python main.py --mode classify --lang en
    uv run python main.py --mode bow
    uv run python main.py --mode compare          # 한국어 Okt vs 공백 토큰화
    uv run python main.py --mode errors           # 오분류 원인: 문서 길이, 한국어 레이블 모호성
    uv run python main.py --mode all              # 검증·분류 (en, ko-okt, ko-whitespace) + BoW + 비교 + 오분류
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-IDF 문서 검색·분류 (A4-1)")
    ap.add_argument("--mode", choices=["search", "verify", "classify", "bow", "compare", "errors", "all"], default="search")
    ap.add_argument("--lang", choices=["en", "ko"], default="en", help="en: 20 Newsgroups, ko: KLUE-YNAT")
    ap.add_argument("--tokenizer", choices=["okt", "whitespace"], default="okt", help="한국어 토큰화 방식")
    ap.add_argument("--query", help="검색어 (search 모드)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--explain", action="store_true", help="1위 문서의 코사인 유사도 계산 과정 출력")
    args = ap.parse_args()
    if args.topk < 1:
        ap.error(f"--topk 는 1 이상이어야 합니다: {args.topk}")

    if args.mode == "search":
        from src.search import format_results, get_engine

        if not args.query:
            ap.error("search 모드에는 --query 가 필요합니다.")
        engine = get_engine(args.lang, args.tokenizer)
        labels = dict(zip(engine.corpus["doc_id"], engine.corpus["label"]))
        print(format_results(args.query, engine.search(args.query, args.topk), labels))
        if args.explain:
            print("\n" + engine.explain(args.query))
    elif args.mode == "verify":
        from src.verify import run

        print(run(args.lang, args.tokenizer))
    elif args.mode == "classify":
        from src.classify import run

        run(args.lang, args.tokenizer)
    elif args.mode == "bow":
        from src.bow_limits import main as bow_main

        bow_main()
    elif args.mode == "compare":
        from src.compare_tokenizers import main as compare_main

        compare_main()
    elif args.mode == "errors":
        from src.error_analysis import main as errors_main

        errors_main()
    else:
        from src.bow_limits import main as bow_main
        from src.classify import run as classify_run
        from src.compare_tokenizers import main as compare_main
        from src.error_analysis import main as errors_main
        from src.verify import run as verify_run

        for lang, tok in (("en", "okt"), ("ko", "okt"), ("ko", "whitespace")):
            print(verify_run(lang, tok), "\n")
            classify_run(lang, tok)
            print()
        bow_main()
        compare_main()
        errors_main()


if __name__ == "__main__":
    main()
