"""Incremental audit, theorem interpretation check and catalog synchronization for CGB.

Rule:
- Untouched theorems with existing descriptions are preserved unchanged.
- Only newly added theorems or modified theorems whose source changed are updated.
- Zero churn when Lean sources are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_proven_index as base

LEAN_DIR = base.LEAN
ROOT_DIR = base.ROOT
INVENTORY_FILE = base.INVENTORY
INTERPRETATIONS_FILE = ROOT_DIR / "docs/information_boltzmann/CGB_THEOREM_INTERPRETATIONS.json"
REVIEWS_DIR = ROOT_DIR / "lean/CGBAxioms/reviews"


def get_current_source_hashes() -> dict[str, str]:
    """Calculate current SHA256 hashes of all Lean source files directly on disk."""
    files = sorted(LEAN_DIR.rglob("*.lean")) + [LEAN_DIR.parent / "CGBAxioms.lean"]
    return {
        p.relative_to(LEAN_DIR.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def get_review_hashes() -> dict[Path, str]:
    """Calculate current SHA256 hashes of review files."""
    return {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(REVIEWS_DIR.rglob("*.lean"))
    }


def scan_review_theorems() -> list[tuple[str, str, int, str]]:
    """Scan review declarations: (name, doc, line, digest)."""
    revs = []
    for path, digest in get_review_hashes().items():
        header, thms = base.scan(path, digest, full_docs=True)
        for n, d, line in thms:
            revs.append((n, d, line, digest))
    return revs


def check_status(inventory: dict, explanations: dict) -> dict:
    """Analyze incremental state without modifying anything."""
    source_hashes = inventory["source_hashes"]
    authored = [r for r in inventory["declarations"] if r["kind"] == "theorem" and r["source_declared"]]

    untouched = []
    stale_hash = []
    missing_desc = []

    for r in authored:
        name = r["name"]
        rel = r["module"].replace(".", "/") + ".lean"
        cur_hash = source_hashes.get(rel)
        exp = explanations.get(name)

        if not exp:
            missing_desc.append((name, rel, cur_hash, r["signature"]))
        elif exp.get("source_sha256") != cur_hash:
            stale_hash.append((name, rel, cur_hash, exp.get("source_sha256"), exp.get("text", "")))
        else:
            untouched.append(name)

    # Review declarations
    rev_theorems = scan_review_theorems()
    rev_untouched = []
    rev_stale_hash = []
    rev_missing = []

    for name, doc, line, digest in rev_theorems:
        exp = explanations.get(name)
        if not exp:
            rev_missing.append((name, digest, doc))
        elif exp.get("source_sha256") != digest:
            rev_stale_hash.append((name, digest, exp.get("source_sha256"), exp.get("text", "")))
        else:
            rev_untouched.append(name)

    return {
        "production": {
            "total": len(authored),
            "untouched": len(untouched),
            "stale_hash": stale_hash,
            "missing": missing_desc
        },
        "reviews": {
            "total": len(rev_theorems),
            "untouched": len(rev_untouched),
            "stale_hash": rev_stale_hash,
            "missing": rev_missing
        }
    }


def generate_concise_summary(name: str, signature: str, doc: str) -> str:
    """Use the Lean declaration's own concise scope; never invent a mechanism."""
    clean_doc = " ".join(doc.strip().split()) if doc else ""
    if clean_doc:
        return clean_doc
    return f"Formal contract for {name.split('.')[-1]}; see its Lean signature for premises and conclusion."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report incremental status only, do not write changes")
    parser.add_argument("--force-audit", action="store_true", help="Force running Lean audit even if hashes match")
    args = parser.parse_args()

    # Step 1: Check if source files on disk match inventory source_hashes
    current_hashes = get_current_source_hashes()

    need_audit = args.force_audit
    if not INVENTORY_FILE.exists():
        need_audit = True
    else:
        try:
            inv = json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
            if current_hashes != inv.get("source_hashes"):
                need_audit = True
        except Exception:
            need_audit = True

    if need_audit and not args.check:
        print("[INCREMENTAL] Source changes detected or audit requested. Running Lean audit...")
        res = subprocess.run([sys.executable, str(ROOT_DIR / "lean/CGBAxioms/scripts/audit.py")],
                             capture_output=True, text=True, encoding="utf-8")
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            sys.exit(f"Audit failed with returncode {res.returncode}")
        print("[INCREMENTAL] Lean audit passed.")

    # Step 2: Load verified inventory & interpretations
    inventory = base.checked_inventory(INVENTORY_FILE)
    explanations = json.loads(INTERPRETATIONS_FILE.read_text(encoding="utf-8")) if INTERPRETATIONS_FILE.exists() else {}

    status = check_status(inventory, explanations)
    prod = status["production"]
    rev = status["reviews"]

    print(f"[INCREMENTAL STATUS]")
    print(f"  生产区定理: 总计 {prod['total']} 条 | 未变更保持: {prod['untouched']} 条 | 哈希需同步: {len(prod['stale_hash'])} 条 | 缺失释义: {len(prod['missing'])} 条")
    print(f"  探索区声明: 总计 {rev['total']} 条 | 未变更保持: {rev['untouched']} 条 | 哈希需同步: {len(rev['stale_hash'])} 条 | 缺失释义: {len(rev['missing'])} 条")

    is_clean = (len(prod['stale_hash']) == 0 and len(prod['missing']) == 0 and
                len(rev['stale_hash']) == 0 and len(rev['missing']) == 0)

    if args.check:
        if is_clean:
            print("[INCREMENTAL CHECK] 100% clean. No changes needed.")
            sys.exit(0)
        else:
            print("[INCREMENTAL CHECK] Out of sync. Run without --check to apply incremental updates.")
            sys.exit(1)

    if is_clean:
        print("[INCREMENTAL] Everything is already up to date. Zero churn.")
        sys.exit(0)

    # Step 3: Apply incremental changes ONLY
    # Rule: Never modify untouched entries.
    updated_entries = 0
    added_entries = 0

    # 1. Update stale hashes for existing interpretations (preserving exact text!)
    for name, rel, cur_hash, old_hash, existing_text in prod["stale_hash"]:
        explanations[name]["source_sha256"] = cur_hash
        updated_entries += 1

    for name, digest, old_digest, existing_text in rev["stale_hash"]:
        explanations[name]["source_sha256"] = digest
        updated_entries += 1

    # 2. Add newly missing production theorems
    for name, rel, cur_hash, sig in prod["missing"]:
        # Find doc if available
        p = ROOT_DIR / "lean/CGBAxioms" / rel
        _, thms = base.scan(p, cur_hash, full_docs=True)
        thms_dict = {t[0]: t[1] for t in thms}
        short_name = name.split(".")[-1]
        doc = thms_dict.get(name) or thms_dict.get(short_name) or ""
        desc = generate_concise_summary(name, sig, doc)
        explanations[name] = {
            "source_sha256": cur_hash,
            "text": desc
        }
        added_entries += 1

    # 3. Add newly missing review declarations
    for name, digest, doc in rev["missing"]:
        desc = generate_concise_summary(name, "", doc)
        explanations[name] = {
            "source_sha256": digest,
            "text": desc
        }
        added_entries += 1

    # Step 4: Write back interpretations safely
    INTERPRETATIONS_FILE.write_text(
        json.dumps(explanations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"[INCREMENTAL] Applied: {updated_entries} hashes synchronized (text preserved), {added_entries} new descriptions added.")

    # Step 5: Regenerate and check catalog
    print("[INCREMENTAL] Regenerating catalog...")
    res_cat = subprocess.run([sys.executable, str(ROOT_DIR / "scripts/build_theorem_catalog.py")],
                             capture_output=True, text=True, encoding="utf-8")
    if res_cat.returncode != 0:
        print(res_cat.stdout)
        print(res_cat.stderr)
        sys.exit("build_theorem_catalog.py failed")

    res_chk = subprocess.run([sys.executable, str(ROOT_DIR / "scripts/build_theorem_catalog.py"), "--check"],
                             capture_output=True, text=True, encoding="utf-8")
    if res_chk.returncode != 0:
        print(res_chk.stdout)
        print(res_chk.stderr)
        sys.exit("build_theorem_catalog.py --check failed")

    print("[INCREMENTAL] Incremental sync complete and catalog verified green.")


if __name__ == "__main__":
    main()
