"""Search the generated theorem catalog without loading every page into context."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from build_theorem_catalog import DEST

ALIASES = {
    "锁存": ("latch", "hold"), "释放": ("release",), "学习率": ("gain", "learning rate"),
    "竞争": ("compet",), "守恒": ("conserv", "budget"), "分化": ("differentiat",),
    "记忆": ("memory",), "优化器": ("optimizer",), "点燃": ("ignition",),
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("terms", nargs="+", help="case-insensitive terms; all must match an entry")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--scope", choices=("production", "reviews", "all"), default="all")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    terms = [term.casefold() for term in args.terms]
    scopes = ("production", "reviews") if args.scope == "all" else (args.scope,)
    hits = []
    for scope in scopes:
        for page in sorted((DEST / scope).glob("*.md")):
            content = page.read_text(encoding="utf-8")
            for entry in re.split(r"(?m)^## ", content)[1:]:
                name, _, body = entry.partition("\n")
                haystack = (page.stem + " " + name + " " + body).casefold()
                if not all(any(word in haystack for word in (term, *ALIASES.get(term, ()))) for term in terms):
                    continue
                explanation = re.search(r"(?m)^(?:解释|源码说明)：(.+)$", body)
                summary = explanation.group(1) if explanation else "无文字解释；请阅读形式陈述与源码。"
                source = re.search(r"\[源码位置\]\(([^)]+)\)", body)
                # Rank matches in the declaration name and its Chinese summary
                # ahead of incidental mentions in a long formal signature.
                name_text = name.casefold()
                module_text = page.stem.casefold()
                summary_text = summary.casefold()
                score = sum(max(3 * (word in name_text) + 2 * (word in summary_text)
                                + (word in module_text)
                                for word in (term, *ALIASES.get(term, ())))
                            for term in terms)
                hits.append((score, scope, name, summary, page, source.group(1) if source else ""))
    hits.sort(key=lambda hit: (-hit[0], hit[1], hit[2]))
    for _, scope, name, summary, page, source in hits[:args.limit]:
        print(f"[{scope}] {name}\n  {summary}\n  catalog: {page}\n  source: {(page.parent / source.split('#')[0]).resolve()}{'#' + source.split('#', 1)[1] if '#' in source else ''}\n")
    print(f"{len(hits)} matches; showing {min(args.limit, len(hits))}")


if __name__ == "__main__":
    main()
