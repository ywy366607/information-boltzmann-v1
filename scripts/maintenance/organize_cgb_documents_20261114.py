"""Archive completed research records, preserving hashes and old link targets.

Documentation only. Does not load a model or touch runtime data.
Explicit destinations must be absent; reruns fail instead of overwriting history.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "information_boltzmann"
ARCHIVE = DOCS / "archive" / "20261114_closed_records"
REPORTS = [
    "CGB_RELATION_CAUSE_MODEL_20261114.md",
    "CGB_RELATION_NONLINEARITY_AUDIT_20261114.md",
    "CGB_SHARED_RELATION_BASIS_20261114.md",
    "CGB_DESIGN_CONTRACT_AUDIT_20261114.md",
    "CGB_JOINT_RELATION_REPAIR_20261114.md",
    "CGB_CANONICAL_RELATION_EVIDENCE_20261114.md",
    "CGB_DELAYED_FACT_IDENTITY_REPAIR_20261114.md",
    "CGB_CONTENT_QUESTION_REPAIR_20261114.md",
]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Outside workspace: {path}")
    return resolved


def relocated_links(text: str, source: Path, destination: Path,
                    relocations: dict[Path, Path]) -> tuple[str, int]:
    changed = 0

    def replace(match: re.Match) -> str:
        nonlocal changed
        target = match.group(1)
        if target.startswith(("#", "/", "\\")) or re.match(r"[a-zA-Z][\w+.-]*:", target):
            return match.group(0)
        angle = target.startswith("<") and target.endswith(">")
        plain = target[1:-1] if angle else target
        path, sep, fragment = plain.partition("#")
        if not path:
            return match.group(0)
        absolute = (source.parent / path).resolve()
        absolute = relocations.get(absolute, absolute)
        relative = Path(os.path.relpath(absolute, destination.parent)).as_posix()
        updated = relative + (sep + fragment if sep else "")
        if angle:
            updated = "<" + updated + ">"
        changed += 1
        return "](" + updated + ")"

    return re.sub(r"\]\(([^)\n]+)\)", replace, text), changed


def anchors(text: str) -> str:
    result, seen = [], {}
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        heading = re.sub(r"^#+\s*", "", line).lower()
        heading = re.sub(r"[^\w\-\s]", "", heading).replace(" ", "-")
        n = seen.get(heading, 0)
        seen[heading] = n + 1
        result.append(f'<a id="{heading}{"-" + str(n) if n else ""}"></a>')
    return "\n".join(result)


def main() -> None:
    paths = {checked(DOCS / name): checked(ARCHIVE / name) for name in REPORTS}
    snapshots = {
        checked(ROOT / "AGENTS.md"): checked(DOCS / "history" / "AGENTS_BEFORE_MASTER_PLAN_20261114.md"),
        checked(DOCS / "CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md"): checked(DOCS / "history" / "RESEARCH_LEDGER_THROUGH_20261114.md"),
    }
    all_paths = {**paths, **snapshots}
    for source, destination in all_paths.items():
        if not source.is_file() or destination.exists():
            raise ValueError(f"Source absent or destination already exists: {source} -> {destination}")
    if (ARCHIVE / "manifest.json").exists():
        raise ValueError("Existing archive manifest")

    entries = []
    for source, destination in all_paths.items():
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig")
        moved = source in paths
        # Snapshots stay byte-exact; record link base explicitly in the manifest.
        archived, count = relocated_links(text, source, destination, paths) if moved else (text, 0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if moved:
            destination.write_text(archived, encoding="utf-8", newline="\n")
            title = next(line for line in text.splitlines() if line.startswith("#"))
            target = destination.relative_to(DOCS).as_posix()
            stub = (title + "\n\n"
                    "> 2026-11-14：本项诊断/接线记录已完成归档，原文中的运行状态和下一步按记录日期阅读。\n\n"
                    f"[阅读全文与原始证据]({target}) · "
                    "[当前总计划](CGB_MASTER_PLAN_20261114.md) · "
                    "[当前身体](CGB_CURRENT_STATE_AND_RESEARCH_LEDGER.md)\n\n"
                    "下列锚点保留旧链接入口；对应章节在归档原文。\n\n" + anchors(text) + "\n")
            source.write_text(stub, encoding="utf-8", newline="\n")
        else:
            destination.write_bytes(raw)
        entries.append({
            "source": source.relative_to(ROOT).as_posix(),
            "archive": destination.relative_to(ROOT).as_posix(),
            "original_sha256": digest(raw),
            "archive_sha256": digest(destination.read_bytes()),
            "mode": "completed_report_with_compatibility_entry" if moved else "byte_exact_entry_snapshot",
            "relative_links_rebased": count,
            "snapshot_relative_link_base": source.parent.relative_to(ROOT).as_posix() if not moved else None,
            "reason": "Retire dated execution instructions; retain full evidence and provenance",
        })
    manifest = {"date": "2026-11-14", "scope": "documentation_only", "entries": entries}
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed_reports": len(paths), "entry_snapshots": len(snapshots),
                      "manifest": str(ARCHIVE / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
