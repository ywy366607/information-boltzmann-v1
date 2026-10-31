"""Build a compact, machine-generated CGB theorem basis index.

The full proven index remains the source of truth.  This file only groups
production modules into a short first-pass reading order; it does not delete
or reinterpret any theorem.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "lean" / "CGBAxioms" / ".lake" / "audit_inventory.json"
OUT = ROOT / "docs" / "information_boltzmann" / "CGB_THEOREM_BASIS_20261030.md"

CORE = {
    "Language", "Symmetry", "Medium", "Life", "Foundation", "Inference",
    "Interaction", "InteractionBridge", "InteractionControls", "CausalLearning",
}
CAUSAL = {
    "LanguageLifeKernel", "InformationEconomics", "GenerationActiveInference",
    "GenerationExport", "GenerationPredictiveHarvest", "GenerationInformationEnergy",
    "GenerationGrowth", "GenerationLinearMemory", "GenerationCascadeMemory",
    "GenerationColumnMemory", "GenerationLearning", "GenerationLeaveRule",
    "GenerationPersistence", "GenerationGradientFilter", "Generation",
    "GenerationSelfProduction",
}
MEDIUM = {
    "GenerationMedium", "GenerationCRN", "GenerationLattice", "GenerationTable",
    "GenerationNestedWorld", "GenerationNesting", "NoetherBridge",
    "AlgebraicCore", "AlgebraicHomology", "PortFlowHomology",
    "EnergyBudget", "EnergyControls", "InformationThermodynamics",
}

WHY = {
    "core": "一般对象、生命目标、开放交互和基本可观测语义。",
    "causal": "因果压缩、有限容量、学习、收益、主动性和持续状态的生成桥。",
    "medium": "守恒、局部介质、同调、能量和具体生成介质的条件。",
    "instances": "控制、见证、物种实例、恢复协议和历史审计；首轮检索不作为生成骨架。",
}


def category(module: str) -> str:
    name = module.removeprefix("CGBAxioms.")
    if name in CORE:
        return "core"
    if name in CAUSAL:
        return "causal"
    if name in MEDIUM:
        return "medium"
    return "instances"


def render(inventory: dict) -> str:
    authored = [r for r in inventory["declarations"]
                if r["source_declared"] and r["kind"] == "theorem"]
    by_module = Counter(r["module"].removeprefix("CGBAxioms.") for r in authored)
    modules = sorted(inventory["summary"]["modules"])
    groups: dict[str, list[tuple[str, int]]] = {k: [] for k in WHY}
    for full in modules:
        name = full.removeprefix("CGBAxioms.")
        groups[category(full)].append((name, by_module.get(name, 0)))

    lines = [
        "# CGB 定理基座索引（自动生成，勿手改）", "",
        "本页不是第二份定理清单，而是 Agent 的首轮检索顺序。完整声明和前提仍以 "
        "[CGB_PROVEN_INDEX.md](CGB_PROVEN_INDEX.md) 与 Lean 源码为准。", "",
        f"当前审计库存：**{len(authored)}** 条源码定理，**{len(modules)}** 个生产模块。", "",
        "## 最小阅读链", "",
        "```text",
        "一般公理 / 生命定义",
        "        ↓",
        "开放交互 + 因果状态 + 有限容量",
        "        ↓",
        "选择性保留 + 预测收益 + 学习更新",
        "        ↓",
        "主动行动 + 行动条件化预测",
        "        ↓",
        "具体介质、物种实例和运行实现",
        "```", "",
        "## 四层模块基座", "",
        "| 层 | 首轮用途 | 定理数 | 模块 |", "|---|---|---:|---|",
    ]
    order = ["core", "causal", "medium", "instances"]
    for key in order:
        entries = groups[key]
        total = sum(n for _, n in entries)
        names = ", ".join(f"`{name}` ({n})" for name, n in entries)
        lines.append(f"| {key} | {WHY[key]} | {total} | {names} |")
    lines += [
        "", "## 三个信息经济学基座", "",
        "这些是压缩后的共同接口，不是新增机制：", "",
        "1. `InformationEconomics.causal_carrier_contract_iff`：因果状态必须同时支持未来读出和递归转移。",
        "2. `InformationEconomics.selective_write_contract`：只有排除未来候选的信息才支付写入并占用容量。",
        "3. `InformationEconomics.predictive_growth_contract`：登记的再投资世界中，收益由交叉熵决定并受熵上界约束。",
        "", "## 使用规则", "",
        "- 先读 `core`，确定对象和量词；再读 `causal`，寻找可复用桥定理。",
        "- 只有当目标需要具体守恒介质或速率条件时才进入 `medium`。",
        "- `instances` 是证书、反例和控制，不把它们当成新的普遍机制。",
        "- 新定理应说明自己属于哪个基座；无法归类时先检查是否重复或缺少共同桥。",
        "- 本页由 `scripts/build_theorem_basis.py` 从审计库存生成，审计成功后由 `audit.py` 自动刷新。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    rendered = render(inventory)
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Theorem basis is stale; run scripts/build_theorem_basis.py")
    else:
        OUT.write_text(rendered, encoding="utf-8")
    print(OUT, "checked" if args.check else "updated")


if __name__ == "__main__":
    main()
