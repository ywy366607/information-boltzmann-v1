import CGBAxioms.LanguageLifeKernel

open Lean CGB.RenewableLearning CGB.LanguageLifeKernel

private def worlds : List Env :=
  [.feedA, .feedB, .missA, .missB, .probeA, .probeB, .answerA, .answerB, .trap]

private def signals : List Signal :=
  [.calm, .answerA, .answerB, .feed, .scarce, .miss, .probe]

private def prior (mask : Fin 512) : Finset Env :=
  (worlds.zipIdx.filter (fun (_, i) => mask.val / 2^i % 2 = 1)).map Prod.fst |>.toFinset

private def mask (states : Finset Env) : Nat :=
  ((worlds.zipIdx.filter (fun (w, _) => w ∈ states)).map (fun (_, i) => 2^i)).sum

private def paidRows : List (List Nat) :=
  [0, 1, 2, 3, 4, 8, 511].flatMap (fun p => (List.finRange 4).flatMap (fun r =>
    (List.finRange 3).flatMap (fun a => signals.zipIdx.flatMap (fun (o, oi) =>
      (List.range 4).map (fun cost =>
        let m : CGB.CausalLearning.Memory Env 3 :=
          ⟨prior ⟨p % 512, Nat.mod_lt _ (by decide)⟩, r⟩
        let next := PaidRecord.settle m a o cost
        [p, r.val, a.val, oi, cost, mask next.candidates, next.reserve.val,
         PaidRecord.fee m a o cost])))))

private def payload : Json := Json.mkObj [
  ("format", toJson "CGB.CausalRecord.v1"),
  ("world_labels", toJson ["feedA", "feedB", "missA", "missB", "probeA", "probeB",
    "answerA", "answerB", "trap"]),
  ("signal_labels", toJson ["calm", "answerA", "answerB", "feed", "scarce", "miss", "probe"]),
  ("world_sensors", toJson (worlds.map (fun w => signals.idxOf (sense w ())))),
  ("weights", toJson (worlds.map (fun w => (List.finRange 3).map (fun a =>
    worlds.map (CausalRecord.weight w a))))),
  ("source_supports", toJson (worlds.map (fun w => (List.finRange 3).map (fun a =>
    signals.map (PaidRecord.possible w () a))))),
  ("paid_cases", toJson paidRows),
  ("updates", toJson ((List.finRange 512).map (fun p => (List.finRange 3).map (fun a =>
    signals.map (fun o => mask (CausalRecord.update (prior p) a o))))))]

#eval IO.println payload.compress
