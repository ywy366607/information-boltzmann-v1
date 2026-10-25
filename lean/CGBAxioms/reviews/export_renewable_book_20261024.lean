import CGBAxioms.RenewableEncoding

/- Export ordinary instance data, not its potentially large Encodable numeral.
The archival JSON length is reported separately from Description.bits. -/

open CGB CGB.Description CGB.RenewableLearning CGB.RenewableEncoding Lean

private def portJson (p : Port) : Json := Json.mkObj [
  ("site", toJson p.site), ("before", toJson p.before),
  ("stimulus", toJson p.stimulus), ("after", toJson p.after)]

private def eventJson (e : Description.Event) : Json := Json.mkObj [
  ("ports", Json.arr (e.ports.map portJson).toArray),
  ("numerator", toJson e.numerator), ("denominator", toJson e.denominator)]

private def payload : Json := Json.mkObj [
  ("format", toJson "CGB.ExchangeBook.data.v1"),
  ("body_book", Json.mkObj [
    ("sites", toJson book.base.sites), ("states", toJson book.base.states),
    ("stimuli", toJson book.base.stimuli),
    ("contacts", Json.arr (book.base.contacts.map (fun (u,v) => toJson [u,v])).toArray),
    ("identities", toJson book.base.identities), ("boundary", toJson book.base.boundary),
    ("contents", toJson book.base.contents), ("initial", toJson book.base.initial),
    ("calm", toJson book.base.calm),
    ("events", Json.arr (book.base.events.map eventJson).toArray)]),
  ("transfers", Json.arr (book.transfers.map (fun (i,d) => toJson [i,d])).toArray),
  ("separate_readout_table", toJson ((List.finRange 8).map (fun q => (policy (cell q)).val))),
  ("cell_labels", toJson ["vacuum", "search", "aSpent", "bSpent", "aReady", "bReady", "aHeld", "bHeld"]),
  ("stimulus_labels", toJson ["calm", "answerA", "answerB", "feed", "scarce", "miss", "probe"])]

#eval IO.println payload.compress
