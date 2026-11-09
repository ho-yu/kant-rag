"""이미 저장된 검색 비교 결과(JSON)를 HTML 페이지로 바꾼다.

    uv run python evaluation/reporting/retrieval_report.py results/comparison/retrieval_b/basic_development_para-table.json

같은 폴더에 이름만 .html로 바꾼 파일을 만든다. --out으로 경로를 바꿀 수 있다.
--open을 붙이면 만든 뒤 브라우저로 바로 연다.
retrieval_b_compare.py를 실행하면 .html도 자동으로 만들어지므로, 이 스크립트는 예전 JSON용이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_service.retrieval_b.metrics import golden_path, load_golden
from rag_service.retrieval_b.report import page_link, write_html


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="검색 비교 JSON → HTML 페이지")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--detail-k", type=int, default=5, help="질문별 표의 k")
    parser.add_argument(
        "--open", action="store_true", help="만든 뒤 브라우저로 바로 연다"
    )
    args = parser.parse_args(argv)

    report = json.loads(args.json_path.read_text(encoding="utf-8"))
    settings = report.get("settings", {})
    try:
        items = load_golden(
            golden_path(
                settings.get("split", "development"),
                domain=settings.get("domain", "shipping"),
                difficulty=settings.get("difficulty", "basic"),
            )
        )
    except (OSError, ValueError) as e:
        print(f"골든셋을 못 읽어서 질문 문장 없이 만듭니다: {e}")
        items = None
    out = write_html(
        report, args.out or args.json_path.with_suffix(".html"), items, args.detail_k
    )
    print(f"저장 (Ctrl+클릭): {page_link(out, open_browser=args.open)}")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
