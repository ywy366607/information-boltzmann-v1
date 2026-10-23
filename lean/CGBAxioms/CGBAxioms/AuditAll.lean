import CGBAxioms.Foundation
import CGBAxioms.RecoveryControls
import CGBAxioms.InteractionControls
import Lean

/-!
Audit every named project theorem and definition, not only a hand-written list.
Signatures and project dependencies expose local premises for human review; this does
not prove that a premise is physically valid, necessary, or a minimal assumption.
-/

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let allowed := #[`propext, `Classical.choice, `Quot.sound]
  let mut theoremCount := 0
  let mut declarationCount := 0
  let entries := env.constants.toList.toArray.qsort (fun a b => Name.lt a.1 b.1)
  for (name, ci) in entries do
    let some idx := env.getModuleIdxFor? name | continue
    let modName := env.header.moduleNames[idx.toNat]!
    unless modName.toString.startsWith "CGBAxioms." do continue
    let kind := match ci with
      | .thmInfo _ => "theorem"
      | .defnInfo _ => "definition"
      | .opaqueInfo _ => "opaque"
      | .axiomInfo _ => "axiom"
      | _ => "structural"
    if kind == "structural" then continue
    if kind == "axiom" then throwError "Unregistered Lean axiom: {name}"
    let axioms ← collectAxioms name
    for ax in axioms do
      unless allowed.contains ax do
        throwError "Non-foundational axiom {ax} in {name}"
    let signature ← liftTermElabM do
      return (← Meta.ppExpr ci.type).pretty
    let typeDeps := ci.type.getUsedConstants.filter (fun n =>
      n.toString.startsWith "CGB.")
    let projectDeps := ci.getUsedConstantsAsSet.toArray.filter (fun n =>
      n.toString.startsWith "CGB.")
    let record := Json.mkObj [
      ("name", toJson name.toString), ("module", toJson modName.toString),
      ("kind", toJson kind), ("signature", toJson signature),
      ("axioms", toJson (axioms.map Name.toString)),
      ("type_dependencies", toJson (typeDeps.map Name.toString)),
      ("project_dependencies", toJson ((projectDeps.qsort Name.lt).map Name.toString))]
    logInfo m!"CGB_AUDIT_RECORD {record.compress}"
    declarationCount := declarationCount + 1
    if kind == "theorem" then theoremCount := theoremCount + 1
  let modules := env.header.moduleNames.filter (fun n =>
    n.toString.startsWith "CGBAxioms.")
  let summary := Json.mkObj [
    ("declarations", toJson declarationCount), ("theorems", toJson theoremCount),
    ("modules", toJson (modules.map Name.toString))]
  logInfo m!"CGB_AUDIT_SUMMARY {summary.compress}"
