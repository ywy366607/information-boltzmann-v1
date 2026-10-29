"""Generate a progressively disclosed theorem catalog from audited Lean sources.

Production signatures come from the kernel audit; review declarations are a
source inventory only and are never represented as covered by that audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import build_proven_index as base

DEST = base.OUT.parent / "CGB_THEOREMS"
REVIEWS = base.LEAN.parent / "reviews"
INTERPRETATIONS = base.OUT.parent / "CGB_THEOREM_INTERPRETATIONS.json"
TOPICS = {
    "公理、守恒和能量": r"medium|foundation|noether|conserv|energy|thermo|material|homology",
    "生命、世界与边界": r"life|world|nest|boundary|port|organism|ecoli|sponge|recovery|maintenance",
    "记忆、点燃和健康锁存": r"memory|latch|ignition|retain|hold|export|forget|cascade",
    "学习、信用与优化器": r"learn|credit|gradient|optim|gain|plastic|signal|kalman|modulat",
    "竞争、资源与分化": r"compet|resource|growth|differenti|budget|alloc|pool|winner|sinkhorn",
    "语言、信息与预测": r"language|information|predict|speech|causal|stream|entropy",
}


def relative_link(page: Path, target: Path, line: int | None = None) -> str:
    link = Path(os.path.relpath(target, page.parent)).as_posix()
    return link + (f"#L{line}" if line is not None else "")


def review_signature(path: Path, line: int) -> str:
    """Read declaration through its first assignment, never its proof body."""
    source = path.read_text(encoding="utf-8")
    clean_lines = base.AUDIT.code_only(source).splitlines()
    lines = source.splitlines()
    start = line - 1
    for i in range(start, len(lines)):
        pos = clean_lines[i].find(":=")
        if pos >= 0:
            return "\n".join(lines[start:i] + [lines[i][:pos]]).strip()
    raise ValueError(f"Missing declaration assignment: {path}:{line}")


def render_page(path: Path, page: Path, header: str, entries: list[dict], audited: bool) -> str:
    scope = ("生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。"
             if audited else "探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。")
    out = [f"# {path.stem}", "", scope, "", f"[源文件]({relative_link(page, path)}) · [返回导航](../README.md)", ""]
    if header:
        out += [f"模块源码说明：{header}", ""]
    for e in entries:
        out += [f"## {e['name']}", "", f"[源码位置]({relative_link(page, path, e['line'])})", ""]
        if e.get("interpretation"):
            out += ["解释：" + e["interpretation"], ""]
        if e["doc"]:
            out += ["源码说明：" + e["doc"], ""]
        else:
            out += ["源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，"
                    "需阅读上下文后才能给出可靠解释。", ""]
        out += ["形式陈述：", "", "```lean", e["signature"], "```", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=base.INVENTORY)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = base.checked_inventory(args.inventory)
    explanations = json.loads(INTERPRETATIONS.read_text(encoding="utf-8")) if INTERPRETATIONS.exists() else {}
    authored = [r for r in inventory["declarations"] if r["kind"] == "theorem" and r["source_declared"]]
    by_module: dict[str, list[dict]] = {}
    for record in authored:
        by_module.setdefault(record["module"], []).append(record)
    pages: dict[Path, str] = {}
    modules = []
    seen = set()
    no_doc = 0
    for path in sorted(base.LEAN.rglob("*.lean")):
        rel = path.relative_to(base.LEAN.parent).as_posix()
        header, thms = base.scan(path, inventory["source_hashes"][rel], full_docs=True)
        if not thms:
            continue
        records = by_module.get("CGBAxioms." + path.stem, [])
        entries = []
        for name, doc, line in thms:
            matches = [r for r in records if r["name"] == name or r["name"].endswith("." + name)]
            if len(matches) != 1 or matches[0]["name"] in seen:
                raise ValueError(f"Unresolved or duplicate: {path}:{line}: {name}")
            record = matches[0]
            seen.add(record["name"])
            no_doc += not bool(doc)
            explanation = explanations.get(record["name"], {})
            if explanation and explanation.get("source_sha256") != inventory["source_hashes"][rel]:
                raise ValueError(f"Interpretation source changed: {record['name']}")
            entries.append(dict(name=record["name"], doc=doc, line=line, signature=record["signature"],
                                interpretation=explanation.get("text", "")))
        page = DEST / "production" / (path.stem + ".md")
        pages[page] = render_page(path, page, header, entries, True)
        modules.append((path.stem, page, entries))
    if seen != {r["name"] for r in authored}:
        raise ValueError("Production catalog does not exactly cover audited source theorems")
    reviews = []
    review_hashes = {}
    for path in sorted(REVIEWS.rglob("*.lean")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        review_hashes[path] = digest
        header, thms = base.scan(path, digest, full_docs=True)
        if not thms:
            continue
        entries = []
        for n, d, line in thms:
            if n in seen:
                raise ValueError(f"Duplicate theorem name across catalog: {n}")
            seen.add(n)
            explanation = explanations.get(n, {})
            if explanation and explanation.get("source_sha256") != digest:
                raise ValueError(f"Interpretation source changed: {n}")
            entries.append(dict(name=n, doc=d, line=line, signature=review_signature(path, line),
                                interpretation=explanation.get("text", "")))
        page = DEST / "reviews" / (path.stem + ".md")
        pages[page] = render_page(path, page, header, entries, False)
        reviews.append((path.stem, page, entries))

    if set(explanations) - seen:
        raise ValueError("Interpretations refer to declarations absent from the catalog")

    readme = ["# CGB 定理渐进检索入口（自动生成）", "",
              "先按下列主题找模块，再打开模块查看每条源码说明与完整形式陈述；"
              "最后核对源码中的定义、前提及实现对应。主题由名称关键词匹配，只是检索辅助，不是完整语义分类。", "",
              "按关键词查询：`python scripts/search_theorems.py gain --limit 10`；"
              "多个关键词要求同时匹配，`--scope production` 或 `--scope reviews` 可限制范围。", "",
              f"生产区：{len(authored)} 条已审计源码定理/引理，{len(modules)} 个有定理的模块；"
              f"其中 {no_doc} 条没有源码文字说明，页面提供完整形式陈述并明确标记。", "",
              f"已逐条补充中文解释：{len(explanations)} 条。解释与模块源码哈希绑定；源码变化时生成器会提示复核。", "",
              f"探索区：{sum(len(e) for _, _, e in reviews)} 条源码声明，来自 {len(reviews)} 个文件；"
              "独立列出，未视为生产区审计通过。", "",
              "需要新增机制前，也检索 [未入 Lean 的手工推导](../CGB_HAND_DERIVATIONS.md)、"
              "[关系图](../CGB_THEOREM_RELATION_MAP_20261028.md) 和 [原完整索引](../CGB_PROVEN_INDEX.md)。", "",
              "## 按主题查找", ""]
    all_modules = [(n, p, e, "生产") for n, p, e in modules] + [(n, p, e, "探索") for n, p, e in reviews]
    for title, pattern in TOPICS.items():
        hits = []
        for name, page, entries, kind in all_modules:
            found = sum(bool(re.search(pattern, name + " " + e["name"] + " " + e["doc"], re.I)) for e in entries)
            if found:
                hits.append(f"[{name}]({relative_link(DEST / 'README.md', page)})（{kind}；{found} 条关键词命中）")
        readme += [f"### {title}", "", "；".join(hits) if hits else "无名称或注释命中；请使用全文搜索。", ""]
    for heading, collection in (("生产模块", modules), ("探索证明文件", reviews)):
        readme += ["## " + heading, ""]
        for name, page, entries in collection:
            missing = sum(not bool(e["doc"]) for e in entries)
            readme.append(f"- [{name}]({relative_link(DEST / 'README.md', page)})：{len(entries)} 条，{missing} 条缺源码说明")
        readme.append("")
    pages[DEST / "README.md"] = "\n".join(readme)
    if base.checked_inventory(args.inventory) != inventory:
        raise ValueError("Production inventory changed during generation")
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in review_hashes.items()):
        raise ValueError("Review source changed during generation")
    stale = [str(p) for p, content in pages.items() if not p.exists() or p.read_text(encoding="utf-8") != content]
    if args.check:
        if stale:
            raise SystemExit(f"Catalog stale: {len(stale)} pages; first: {stale[0]}")
    else:
        for page, content in pages.items():
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(content, encoding="utf-8")
    print(json.dumps(dict(production=len(authored), missing_source_docs=no_doc,
                          reviews=sum(len(e) for _, _, e in reviews), pages=len(pages), checked=args.check)))


if __name__ == "__main__":
    main()
