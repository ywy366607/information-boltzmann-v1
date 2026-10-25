import CGBAxioms.LanguageLifeKernel

open Lean CGB CGB.RenewableLearning CGB.LanguageLifeKernel

private def cells : List Cell :=
  [.vacuum, .search, .aSpent, .bSpent, .aReady, .bReady, .aHeld, .bHeld]

private def cellIndex : Cell → Nat
  | .vacuum => 0 | .search => 1 | .aSpent => 2 | .bSpent => 3
  | .aReady => 4 | .bReady => 5 | .aHeld => 6 | .bHeld => 7

private def stimuli : List SensoryMaintenance.Stim :=
  [none, some .calm, some .answerA, some .answerB,
    some .feed, some .scarce, some .miss, some .probe]

private def index (x : SensoryMaintenance.State) : Nat := cellIndex x.1 * 3 + x.2.val

private def row (x : SensoryMaintenance.State) : Json := Json.mkObj [
  ("index", toJson (index x)), ("cell", toJson (cellIndex x.1)),
  ("reserve", toJson x.2.val),
  ("admissible", toJson (decide (SensoryRecord.admissible x))),
  ("record", toJson (SensoryRecord.encode x).val),
  ("action", toJson (policy x.1).val),
  ("content", toJson (SensoryMaintenance.cost x)),
  ("transitions", toJson (stimuli.map (fun s => index (SensoryMaintenance.next s x)))),
  ("ports", toJson (stimuli.map (fun s =>
    [SensoryMaintenance.incoming s x, SensoryMaintenance.outgoing s x,
     SensoryMaintenance.writes s x, SensoryMaintenance.topups s x,
     SensoryMaintenance.lostCredit s x])))]

private def payload : Json := Json.mkObj [
  ("format", toJson "CGB.SensoryRecord.v1"),
  ("source", toJson "CGB.SensoryMaintenance"),
  ("initial", toJson (index (.search, 2))),
  ("stimulus_labels", toJson
    ["missing", "calm", "answerA", "answerB", "feed", "scarce", "miss", "probe"]),
  ("cell_labels", toJson
    ["vacuum", "search", "aSpent", "bSpent", "aReady", "bReady", "aHeld", "bHeld"]),
  ("port_labels", toJson ["incoming", "outgoing", "writes", "topups", "lostCredit"]),
  ("rows", Json.arr ((cells.flatMap (fun q =>
    (List.finRange 3).map (fun reserve => row (q, reserve)))).toArray)),
  ("record_decode", toJson ((List.finRange 16).map (fun r => index (SensoryRecord.read r)))),
  ("record_transitions", toJson ((List.finRange 16).map (fun r =>
    stimuli.map (fun s => (SensoryRecord.write s r).val))))]

#eval IO.println payload.compress
