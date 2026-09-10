"""Substep work/OU fluctuation accounting, with no automatic NESS label."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    config=json.loads(args.config.read_text())
    model=InformationBoltzmann.from_config(config).to(device=args.device, dtype=torch.float64)
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
        model.load_state_dict(saved["model"])
        model.force.gamma=float(saved.get("force_gamma",model.force.gamma))
        model.force.kappa=float(saved.get("force_kappa",model.force.kappa))
    model.adaptive_gamma=False
    gen=torch.Generator(device=args.device).manual_seed(11)
    tokens=np.load(args.tokens,mmap_mode="r")[:args.steps]
    energy=lambda s: float((s.v.square()+model.force.kappa*s.x.square()).sum(-1).mean()/2)
    rows=[]
    with torch.no_grad():
        state=model.initialize(torch.tensor([1],device=args.device),gen)
        current=1
        for token in tokens:
            before=energy(state)
            budget={}
            state,_,_=model._advance_steps(state,current,gen,budget=budget)
            current=int(token)
            delta=energy(state)-before
            predicted=(budget["drive_work"]+budget["trap_work"]-budget["deterministic_damping_loss"]
                       +budget["ou_fluctuation_energy"]+budget["drift_potential_change"]
                       +budget.get("collision_energy_error",0.))
            rows.append({"energy_change":delta,"accounting_residual":delta-predicted,
                         "trap_splitting_residual":budget["trap_work"]+budget["drift_potential_change"],**budget})
    report={"status":"discrete_substep_accounting_only_not_NESS_proof","events":len(rows),
            "max_abs_accounting_residual":max(abs(r["accounting_residual"]) for r in rows),
            "means":{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},
            "definition":"OU fluctuation energy is endpoint stochastic increment, not raw continuous heat injection",
            "rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="rows"}))

if __name__=="__main__":
    main()
