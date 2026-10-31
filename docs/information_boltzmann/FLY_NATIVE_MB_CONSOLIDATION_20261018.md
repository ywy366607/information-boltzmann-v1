# Native DPM/APL memory regulation (2026-10-18)

> Superseded historical proxy. This extension pools bilateral cells into scalar
> threshold/forgetting feedback. It is quarantined when the anatomical MB terminal
> route is attached; its old parameters, moments, traces and pending derivative
> history remain archived. Do not call this a native per-cell mechanism or use its
> old acute results to certify the replacement. See
> `FLY_NATIVE_MB_TERMINAL_20261018.md` for the actual source/target implementation.

The second registry system adds persistent DPM and APL state around the
existing KC→MBON compartment memory.

- DPM activity becomes a slow native trace and a learnable retention factor on
  the existing compartment forgetting term.
- APL is already a graded inhibitory cell in the corrected graph. Its membrane
  activity now also has a slow trace and a learnable threshold feedback on the
  same Kenyon population; the fast APL anatomical edges remain the primary
  route.
- All four parameters are zero-equivalent at their coupling terms and are
  registered in optimizer/native-credit state by semantic name.
- The DPM/APL traces are persistent `FlyPhysicalState` fields and therefore
  cross BPTT windows and full checkpoint restore.

The implementation is an interface closure. It still requires a real-language
continuation and same-state DPM/APL-off or sham control before any claim about
consolidation, forgetting, or language learning benefit.
