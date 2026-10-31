"""Build the dependency slice for the CGB/MengLi life requirements.

The full Lean inventory is authoritative.  This script makes no semantic claim
from a theorem name: the target table explicitly states which L requirement a
representative declaration covers, and whether the coverage is partial.  It
then follows audited project dependencies and preserves formal antecedents as
conditions.  Missing declarations are reported as structure/world boundaries,
never silently dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "lean" / "CGBAxioms" / ".lake" / "audit_inventory.json"
OUT_MD = ROOT / "docs" / "information_boltzmann" / "CGB_MENGLI_GENERATOR_SLICE.md"
OUT_JSON = ROOT / "results" / "published" / "cgb_mengli_generator_slice.json"


@dataclass(frozen=True)
class Target:
    requirement: str
    meaning: str
    roots: tuple[str, ...]
    coverage: str


# This table is the only semantic input.  L0--L6 are prose requirements in the
# original definition, not seven same-typed Lean predicates.  Empty roots mean
# exactly that no declaration currently claims direct coverage.
CONTRACT_GROUPS = (
    (
        "因果载体",
        "`reserve/encode/read/next/advanceHistory/required` 必须成为同一状态 Z 的投影，不能另置记忆模块。",
        (
            "CGB.LanguageLifeAdmission.LanguageCandidate.reserve",
            "CGB.LanguageLifeAdmission.LanguageCandidate.encode",
            "CGB.LanguageLifeAdmission.LanguageCandidate.read",
            "CGB.LanguageLifeAdmission.LanguageCandidate.next",
            "CGB.LanguageLifeAdmission.LanguageCandidate.advanceHistory",
            "CGB.LanguageLifeAdmission.LanguageCandidate.required",
            "CGB.LanguageLifeAdmission.LanguageCandidate.read_correct",
            "CGB.LanguageLifeAdmission.LanguageCandidate.next_correct",
        ),
    ),
    (
        "选择写入与可塑性",
        "候选、费用、保留和写入精化必须是 Φ 对同一 Z 的局部变化，而非外置优化器状态。",
        (
            "CGB.CausalLearning.Memory",
            "CGB.CausalLearning.Support",
            "CGB.CausalLearning.advance",
            "CGB.CausalLearning.fee",
            "CGB.CausalLearning.retain_consistent",
            "CGB.LanguageLifeAdmission.LanguageCandidate.write_refines",
        ),
    ),
    (
        "边界、行动与世界后果",
        "`context/action/outcome/policy/possible/world_response` 必须成为边界事件和 Φ 输出，而非预装世界模型。",
        (
            "CGB.LanguageLifeAdmission.LanguageCandidate.context",
            "CGB.LanguageLifeAdmission.LanguageCandidate.action",
            "CGB.LanguageLifeAdmission.LanguageCandidate.outcome",
            "CGB.LanguageLifeAdmission.LanguageCandidate.policy",
            "CGB.LanguageLifeAdmission.LanguageCandidate.possible",
            "CGB.SelfProduction.LanguageSelfProductionCandidate.world_response",
            "CGB.SelfProduction.Contract.response",
            "CGB.SelfProduction.Contract.step",
        ),
    ),
    (
        "出生、连续性与主动性",
        "`seed/viable/signal` 必须成为可达初态、可存活区域和世界条件；不能偷换成奖励或食物账户。",
        (
            "CGB.SelfProduction.LanguageSelfProductionCandidate.seed",
            "CGB.SelfProduction.LanguageSelfProductionCandidate.seed_nonempty",
            "CGB.SelfProduction.language_viable",
            "CGB.SelfProduction.language_viable_next",
            "CGB.ActiveInference.unbounded",
            "CGB.ActiveInference.passiveBounded",
            "CGB.ActiveInference.actionSensitive",
        ),
    ),
)


TARGETS = (
    Target(
        "L0/L6", "生存连续性与自身可存活区间",
        (
            "CGB.SelfProduction.language_infinite_stream_life_complete",
            "CGB.SelfProduction.language_controlled_trajectory_viable",
        ),
        "条件合同；尚非共同动力学的无条件存在。",
    ),
    Target(
        "L1", "自举形成并维持以自身存在为核心的内在目的",
        ("CGB.SelfProduction.language_self_consequence_model_exists",),
        "仅有自我后果模型的条件存在；目的涌现本身没有直接根定理。",
    ),
    Target(
        "L2", "记忆、学习与压缩形成智能",
        (
            "CGB.InformationEconomics.causal_carrier_contract_iff",
            "CGB.InformationEconomics.selective_write_contract",
            "CGB.InformationEconomics.predictive_growth_contract",
            "CGB.LanguageLifeKernel.delayed_credit_requires_history_carrier",
        ),
        "机制链已形式化；尚待同一语言状态精化。",
    ),
    Target(
        "L3", "局部世界模型、预测未来",
        ("CGB.SelfProduction.language_self_consequence_model_exists",),
        "条件性行动后果模型；尚非经干预验收的学得世界模型。",
    ),
    Target(
        "L4", "选择行动并形成操作闭合",
        ("CGB.ActiveInference.active_closure",),
        "给定供给界与行动敏感世界的条件结论。",
    ),
    Target(
        "L5", "构建并维持生态位",
        (),
        "没有精确的顶层 Lean 根声明；不能由邻近行动定理代填。",
    ),
)


def source_path(module: str) -> str:
    name = module.removeprefix("CGBAxioms.")
    return f"lean/CGBAxioms/CGBAxioms/{name}.lean"


def top_level_implications(signature: str) -> list[str]:
    """Return explicit top-level theorem antecedents.

    Arrows inside a function-valued binder (for example ``response : A → I``)
    are not theorem conditions.  Only arrows at bracket depth zero after the
    universal binder comma are retained.
    """
    depth = 0
    commas: list[int] = []
    arrows: list[int] = []
    pairs = {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}
    for i, char in enumerate(signature):
        depth += pairs.get(char, 0)
        if depth == 0 and char == ",":
            commas.append(i)
        if depth == 0 and char == "→":
            arrows.append(i)
    if not arrows:
        return []
    # A top-level equivalence is a complete proposition.  Arrows later inside
    # quantified right-hand clauses are not its admission conditions.
    first_equiv = signature.find("↔")
    if 0 <= first_equiv < arrows[0]:
        return []
    start = max((comma for comma in commas if comma < arrows[0]), default=-1)
    if start < 0:
        return []
    positions = [start, *arrows, len(signature)]
    parts = [signature[positions[i] + 1 : positions[i + 1]].strip()
             for i in range(len(positions) - 1)]
    return [part for part in parts[:-1] if part]


def closure(root: str, declarations: dict[str, dict[str, Any]]) -> tuple[set[str], set[str], dict[str, int]]:
    seen: set[str] = set()
    missing: set[str] = set()
    distance: dict[str, int] = {}
    todo: deque[tuple[str, int]] = deque([(root, 0)])
    while todo:
        name, depth = todo.popleft()
        if name in seen:
            continue
        seen.add(name)
        distance[name] = depth
        record = declarations.get(name)
        if record is None:
            missing.add(name)
            continue
        for dep in record["project_dependencies"]:
            todo.append((dep, depth + 1))
    return seen, missing, distance


def short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build(inventory: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    declarations = {row["name"]: row for row in inventory["declarations"]}
    target_rows: list[dict[str, Any]] = []
    support: Counter[str] = Counter()
    all_seen: set[str] = set()
    all_missing: set[str] = set()
    direct_conditions: list[dict[str, str]] = []

    for target in TARGETS:
        covered: set[str] = set()
        missing: set[str] = set()
        root_rows: list[dict[str, Any]] = []
        for root in target.roots:
            seen, absent, distance = closure(root, declarations)
            covered |= seen
            missing |= absent
            record = declarations.get(root)
            if record is None:
                root_rows.append({"name": root, "missing": True})
                continue
            conditions = top_level_implications(record["signature"])
            root_rows.append({
                "name": root,
                "module": record["module"],
                "kind": record["kind"],
                "conditions": conditions,
                "closure_size": len(seen),
                "distance": distance,
            })
            for condition in conditions:
                direct_conditions.append({
                    "requirement": target.requirement,
                    "root": root,
                    "condition": condition,
                })
        local = {name for name in covered if name in declarations}
        for name in local:
            support[name] += 1
        all_seen |= covered
        all_missing |= missing
        target_rows.append({
            "requirement": target.requirement,
            "meaning": target.meaning,
            "coverage": target.coverage,
            "roots": root_rows,
            "closure": sorted(local),
            "external_boundary": sorted(missing),
        })

    local_rows = [declarations[n] for n in all_seen if n in declarations]
    # Nodes reached from two independently specified life requirements are the
    # current cross-requirement spine.  This is not claimed to be a unique
    # semantic minimum; equivalent proofs may yield a different spine.
    shared = sorted(
        (n for n, count in support.items()
         if count >= 2
         and declarations[n]["kind"] == "theorem"
         and "._proof_" not in n
         and ".inst" not in n),
        key=lambda n: (-support[n], n),
    )
    foundations = sorted(
        (
            r["name"]
            for r in local_rows
            if not any(dep in declarations for dep in r["project_dependencies"])
        ),
        key=lambda n: (declarations[n]["module"], n),
    )
    contract_groups = []
    for label, obligation, prefixes in CONTRACT_GROUPS:
        members = sorted(
            name for name in all_seen
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        )
        contract_groups.append({
            "group": label,
            "obligation": obligation,
            "members": members,
        })

    result = {
        "schema": 1,
        "inventory_declarations": inventory["summary"]["declarations"],
        "inventory_source_theorems": inventory["source_declared_theorem_count"],
        "targets": target_rows,
        "union": {
            "local_declarations": len(local_rows),
            "theorems": sum(r["kind"] == "theorem" for r in local_rows),
            "definitions": sum(r["kind"] == "definition" for r in local_rows),
            "external_boundaries": sorted(all_missing),
            "modules": dict(sorted(Counter(r["module"] for r in local_rows).items())),
        },
        "cross_requirement_spine": [
            {
                "name": name,
                "requirements": support[name],
                "kind": declarations[name]["kind"],
                "module": declarations[name]["module"],
            }
            for name in shared
        ],
        "foundations": foundations,
        "direct_conditions": direct_conditions,
        "contract_groups": contract_groups,
    }

    lines = [
        "# MengLi 最小生成核：自动反向依赖切片（勿手改）", "",
        "来源：Lean 审计库存。此页不把相邻定理算作覆盖，不从名称猜测条件；"
        "它保留声明的直接形式前提和审计到的项目依赖。完整机器数据见同名 JSON。", "",
        f"库存：{result['inventory_source_theorems']} 条源码定理。首轮生命切片："
        f"{result['union']['local_declarations']} 个本地声明（"
        f"{result['union']['theorems']} 定理、{result['union']['definitions']} 定义），"
        f"{len(result['union']['external_boundaries'])} 个结构/世界边界。", "",
        "## L0–L6 根与覆盖边界", "",
        "| 需求 | 代表性根声明 | 覆盖状态 |", "|---|---|---|",
    ]
    for target in result["targets"]:
        roots = "<br>".join(f"`{r['name']}`" for r in target["roots"]) or "—"
        lines.append(f"| {target['requirement']} | {roots} | {target['coverage']} |")

    lines += ["", "## 直接条件边", "",
        "下列是根声明签名中明确写出的蕴含前提；构造 `Φ` 时必须逐项落实为"
        "公理结果、共同状态不变式或登记的世界合同。", ""]
    if not direct_conditions:
        lines.append("本轮根声明没有可解析的直接蕴含前提。")
    else:
        lines += ["| 服务需求 | 根声明 | 形式条件 |", "|---|---|---|"]
        for item in direct_conditions:
            lines.append(
                f"| {item['requirement']} | `{item['root'].rsplit('.', 1)[-1]}` | `{short(item['condition'])}` |"
            )

    lines += ["", "## 跨需求脊柱", "",
        "以下**定理**被至少两个独立需求切片共享。状态字段和编译辅助项留在 JSON，"
        "避免把文档重新膨胀为索引；这不是唯一的语义最小公理集。", ""]
    if shared:
        lines += ["| 声明 | 服务需求数 | 类别 | 源模块 |", "|---|---:|---|---|"]
        for item in result["cross_requirement_spine"]:
            lines.append(
                f"| `{item['name']}` | {item['requirements']} | {item['kind']} | `{item['module']}` |"
            )
    else:
        lines.append("当前根集合没有共享的审计声明；这表示目标层还未被统一为同一形式对象。")

    lines += ["", "## 共同 Φ 必须消化的合同字段", "",
        "这四组来自切片中实际出现的结构字段和声明，不是新增模块。它们说明最小生成器"
        "必须把哪些既有合同投影到同一局部状态、边界事件和递推上。完整成员表在 JSON。", "",
        "| 合同组 | 切片成员数 | 对 Φ 的义务 |", "|---|---:|---|"]
    for group in contract_groups:
        lines.append(f"| {group['group']} | {len(group['members'])} | {group['obligation']} |")

    lines += ["", "## 未自动消去的边界", "",
        "这些名称未出现在审计声明表中，通常是结构字段、类型类或外部世界条件。"
        "它们不能被误写成已经由 Lean 推出的机制。", ""]
    if result["union"]["external_boundaries"]:
        for name in result["union"]["external_boundaries"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("无。")

    lines += ["", "## 使用", "",
        "1. 新机制先选择其服务的 L 需求和根声明；",
        "2. 检查该根的直接条件能否由同一 `Φ` 满足；",
        "3. 仅在反例显示缺少可区分自由度时增加状态；",
        "4. 运行 `python scripts/build_mengli_generator_slice.py --check`，防止索引更新后文档漂移。", ""]
    return "\n".join(lines), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory_text = args.inventory.read_text(encoding="utf-8")
    inventory = json.loads(inventory_text)
    markdown, result = build(inventory)
    result["inventory_sha256"] = hashlib.sha256(inventory_text.encode()).hexdigest()
    rendered_json = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.out_md.exists() or args.out_md.read_text(encoding="utf-8") != markdown:
            raise SystemExit("MengLi generator slice Markdown is stale; rerun without --check")
        if not args.out_json.exists() or args.out_json.read_text(encoding="utf-8") != rendered_json:
            raise SystemExit("MengLi generator slice JSON is stale; rerun without --check")
        print(args.out_md, "and", args.out_json, "checked")
        return
    args.out_md.write_text(markdown, encoding="utf-8")
    args.out_json.write_text(rendered_json, encoding="utf-8")
    print(args.out_md, "and", args.out_json, "updated")


if __name__ == "__main__":
    main()
