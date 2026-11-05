"""Build the theorem basis from audited declarations and per-theorem reviews."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "lean/CGBAxioms/.lake/audit_inventory.json"
CLASSES = ROOT / "docs/information_boltzmann/CGB_THEOREM_CLASSES.json"
OUT = ROOT / "docs/information_boltzmann/CGB_THEOREM_BASIS_20261030.md"


def render(inventory: dict, classifications: dict) -> str:
    authored = {r["name"] for r in inventory["declarations"]
                if r["source_declared"] and r["kind"] == "theorem"}
    records = classifications["theorems"]
    missing, extra = authored - records.keys(), records.keys() - authored
    if missing or extra:
        raise ValueError(f"Classification coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    for name, entry in records.items():
        for key in ("class", "scope", "status"):
            if entry.get(key) not in classifications["enums"][key] and not (
                    key == "scope" and entry.get(key) is None and entry.get("status") != "reviewed"):
                raise ValueError(f"Invalid {key} for {name}")
        if entry.get("status") == "reviewed" and entry.get("scope") == "candidate" and not entry.get("candidate"):
            raise ValueError(f"Missing candidate for {name}")
    status = Counter(e["status"] for e in records.values())
    lines = ["# CGB 定理基座索引（自动生成，勿手改）", "",
             "逐条分类来自 [CGB_THEOREM_CLASSES.json](CGB_THEOREM_CLASSES.json)；"
             "draft 仅为未审核种子，不作为可靠分类。完整条件请核对 Lean 源码。", "",
             f"源码定理：**{len(authored)}**；已审核：**{status['reviewed']}**；"
             f"未审核：**{len(authored) - status['reviewed']}**。", "",
             "| 分类 | 已审核 | 未审核 |", "|---|---:|---:|"]
    for category in classifications["enums"]["class"]:
        entries = [e for e in records.values() if e["class"] == category]
        reviewed = sum(e["status"] == "reviewed" for e in entries)
        lines.append(f"| {category} | {reviewed} | {len(entries) - reviewed} |")
    universal = sum(e["status"] == "reviewed" and e["class"] == "necessity"
                    and e["scope"] == "universal" for e in records.values())
    lines += ["", f"已审核的 universal necessity：**{universal}**。", "",
              "检索入口：[完整索引](CGB_PROVEN_INDEX.md) · [主题导航](CGB_THEOREMS/README.md)。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    classes = json.loads(CLASSES.read_text(encoding="utf-8"))
    rendered = render(inventory, classes)
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Theorem basis is stale; run scripts/build_theorem_basis.py")
    else:
        OUT.write_text(rendered, encoding="utf-8")
    print(OUT, "checked" if args.check else "updated")


if __name__ == "__main__":
    main()
