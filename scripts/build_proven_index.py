"""Build docs/information_boltzmann/CGB_PROVEN_INDEX.md: every Lean theorem with the first
sentence of its doc comment, grouped by module, so existing results are searched before new work.

usage: python scripts/build_proven_index.py
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "CGBAxioms" / "CGBAxioms"
OUT = ROOT / "docs" / "information_boltzmann" / "CGB_PROVEN_INDEX.md"
DECL = re.compile(r"^(?:private\s+)?(theorem|lemma|def|structure|noncomputable def|abbrev)\s+([^\s:({\[]+)")


def first_sentence(doc: str) -> str:
    doc = " ".join(doc.split())
    m = re.match(r"(.+?[.。])(\s|$)", doc)
    return (m.group(1) if m else doc)[:220]


def scan(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    out, doc, in_doc, buf = [], None, False, []
    header = ""
    for ln in lines:
        s = ln.strip()
        if s.startswith("/-!") and not header:
            in_doc, buf = "module", [s[3:]]
            if s.endswith("-/"):
                header, in_doc = first_sentence(" ".join(buf)[:-2]), False
            continue
        if s.startswith("/--"):
            in_doc, buf = "decl", [s[3:]]
            if s.endswith("-/"):
                doc, in_doc = " ".join(buf)[:-2], False
            continue
        if in_doc:
            buf.append(s)
            if s.endswith("-/"):
                text = " ".join(buf)[:-2]
                if in_doc == "module":
                    header = first_sentence(text.lstrip("# "))
                else:
                    doc = text
                in_doc = False
            continue
        m = DECL.match(s)
        if m:
            kind, name = m.group(1), m.group(2)
            if kind in ("theorem", "lemma"):
                out.append((name, first_sentence(doc) if doc else ""))
            doc = None
        elif s and not s.startswith("--") and not s.startswith("@[") and not s.startswith("omit") \
                and not s.startswith("open") and not s.startswith("variable"):
            pass
    return header, out


def main():
    parts = ["# CGB 已证定理索引（自动生成，勿手改）",
             "",
             "由 `scripts/build_proven_index.py` 从 Lean 源码生成。新工作开始前先在这里检索"
             "（定理名与文档注释首句），避免重复推导。",
             ""]
    total = 0
    for path in sorted(LEAN.glob("*.lean")):
        header, thms = scan(path)
        if not thms:
            continue
        total += len(thms)
        parts.append(f"## {path.stem}（{len(thms)}）")
        if header:
            parts.append(f"_{header}_")
        parts.append("")
        for name, doc in thms:
            parts.append(f"- `{name}`" + (f"：{doc}" if doc else ""))
        parts.append("")
    parts.insert(4, f"共 {total} 条定理/引理。")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(OUT, total)


if __name__ == "__main__":
    main()
