"""Build docs/information_boltzmann/CGB_PROVEN_INDEX.md: every Lean theorem with the first
sentence of its doc comment, grouped by module, so existing results are searched before new work.

usage: python scripts/build_proven_index.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "CGBAxioms" / "CGBAxioms"
OUT = ROOT / "docs" / "information_boltzmann" / "CGB_PROVEN_INDEX.md"
INVENTORY = LEAN.parent / ".lake" / "audit_inventory.json"
DECL = re.compile(r"^(?:private\s+)?(theorem|lemma|def|structure|noncomputable def|abbrev)\s+([^\s:({\[]+)")
AUDIT_SPEC = importlib.util.spec_from_file_location("cgb_index_audit", LEAN.parent / "scripts/audit.py")
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def first_sentence(doc: str) -> str:
    doc = " ".join(doc.split())
    m = re.match(r"(.+?[.。])(\s|$)", doc)
    return (m.group(1) if m else doc)[:220].rstrip()


def scan(path: Path, expected_hash: str, *, full_docs: bool = False):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"Source changed before indexing: {path}")
    source = raw.decode("utf-8")
    lines = source.splitlines()
    code = AUDIT.code_only(source).splitlines()
    out, doc, in_doc, buf = [], None, False, []
    scopes = []
    header = ""
    for number, (ln, clean) in enumerate(zip(lines, code), 1):
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
        control = re.match(r"^(namespace|section|end)(?:\s+(\S+))?\s*$", clean.strip())
        if control:
            kind, name = control.groups()
            if kind == "end":
                scopes.pop()
            else:
                scopes.append(name if kind == "namespace" else None)
        m = DECL.match(clean.strip())
        if m:
            kind, name = m.group(1), m.group(2)
            if kind in ("theorem", "lemma"):
                qualified = ".".join([n for n in scopes if n] + [name])
                out.append((qualified, (" ".join(doc.split()) if full_docs else first_sentence(doc)) if doc else "", number))
            doc = None
        elif s and not s.startswith("--") and not s.startswith("@[") and not s.startswith("omit") \
                and not s.startswith("open") and not s.startswith("variable"):
            pass
    return header, out


def checked_inventory(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not (data["module_coverage_passed"] and data["proof_trust_passed"]):
        raise ValueError("Lean trust/coverage audit must pass before indexing")
    files = sorted(LEAN.rglob("*.lean")) + [LEAN.parent / "CGBAxioms.lean"]
    hashes = {p.relative_to(LEAN.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    if hashes != data["source_hashes"]:
        raise ValueError("Lean audit is stale; build and audit current sources before indexing")
    return data


def source_link(path, name, line):
    target = "../../" + path.relative_to(ROOT).as_posix()
    return f"[`{name}`]({target}#L{line})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed index needs regeneration")
    parser.add_argument("--inventory", type=Path, default=INVENTORY, help="trusted inventory from audit.py")
    args = parser.parse_args()
    inventory = checked_inventory(args.inventory)
    authored = [r for r in inventory["declarations"] if r["kind"] == "theorem" and r["source_declared"]]
    modules, covered = [], set()
    for path in sorted(LEAN.rglob("*.lean")):
        relative = path.relative_to(LEAN.parent).as_posix()
        header, thms = scan(path, inventory["source_hashes"][relative])
        records = [r for r in authored if r["module"] == "CGBAxioms." + path.stem]
        for name, _, line in thms:
            matches = [r for r in records if r["name"] == name or r["name"].endswith("." + name)]
            if len(matches) != 1 or matches[0]["name"] in covered:
                raise ValueError(f"Unresolved or duplicate source theorem: {path}:{line}: {name}")
            covered.add(matches[0]["name"])
        if thms:
            modules.append((path, header, thms))
    if covered != {r["name"] for r in authored}:
        raise ValueError("Source index does not exactly cover audited authored theorems")
    parts = ["# CGB 已证定理索引（自动生成，勿手改）",
             "",
             "由 `scripts/build_proven_index.py` 从 Lean 源码生成，并逐条核对当前审计清单。"
             "新工作开始前先检索已有机制；点击完整定理名查看源码、前提及结论。",
             "渐进检索入口：[主题导航与逐条完整陈述](CGB_THEOREMS/README.md)；"
             "先查主题，再按模块阅读。无源码解释的条目明确标注，避免把机器猜测当作定理含义。",
             "",
             f"共 **{len(authored)}** 条已审计源码定理/引理，来自 **{len(inventory['summary']['modules'])}** 个生产模块。"
             f"编译声明共 **{inventory['summary']['declarations']}** 项，含定义及自动生成项，不作为机制数量。",
             "生成时全部生产源码哈希与通过信任/覆盖检查的库存一致；运行 `--check` 检测过时索引。"
             "标准入口 `lean/CGBAxioms/scripts/audit.py` 在构建、审计成功后自动刷新本页；"
             "其 `--check-index` 模式只核对索引、不改写索引。",
             "",
             "**未入 Lean 的手工推导与数值见证**另列在 `CGB_HAND_DERIVATIONS.md`，检索时两处都要看。",
             "`reviews/` 中的探索证明见渐进检索入口的独立目录，未混入此生产清单。证明通过不替代其局部前提与实现对应检查。",
             "",
             "既有中文机制解读见 [Claude 生成链](CGB_GENERATION_LAYER1_20261025.md)；"
             "完整接线记录见 [统一参考](CGB_UNIFIED_REFERENCE_20261026.md)。这些文档保留各自证据范围。",
             "",
             "## 生成链导航（自动提取源码标注）",
             "",
             "以下从全部 `Generation*.lean` 定理注释的开头加粗标注自动提取，"
             "不维护手选机制名单。标注供检索，成立范围以链接中的定理签名为准；"
             "未带标注的定理仍全部收入后面的完整清单。",
             "",
             "| 模块 | 源码标注与位置 |",
             "|---|---|",
             ""]
    for path, _, thms in modules:
        if not path.stem.startswith("Generation"):
            continue
        links = []
        for name, doc, line in thms:
            marker = re.match(r"\*\*([^*]+)\*\*", doc)
            if marker:
                label = marker[1].replace("|", "&#124;")
                target = "../../" + path.relative_to(ROOT).as_posix()
                links.append(f"[{label}]({target}#L{line})")
        if links:
            parts.insert(-1, f"| {path.stem} | {'；'.join(links)} |")
    parts.extend(["## 完整模块目录", ""])
    parts.extend(f"- [{path.stem}（{len(thms)}）](#{path.stem.lower()}{len(thms)})"
                 for path, _, thms in modules)
    parts.append("")
    for path, header, thms in modules:
        parts.append(f"## {path.stem}（{len(thms)}）")
        if header:
            parts.append(f"_{header}_")
        parts.append("")
        for name, doc, line in thms:
            parts.append(f"- {source_link(path, name, line)}" + (f"：{doc}" if doc else ""))
        parts.append("")
    rendered = "\n".join(parts) + "\n"
    if checked_inventory(args.inventory) != inventory:
        raise ValueError("Lean inventory changed during indexing; retry after auditing")
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Proven index is stale; run scripts/build_proven_index.py")
    else:
        OUT.write_text(rendered, encoding="utf-8")
    print(OUT, len(authored), "current" if args.check else "updated")


if __name__ == "__main__":
    main()
