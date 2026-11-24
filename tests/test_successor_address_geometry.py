"""Native-address vectorization and preserved raw scope semantics."""
import torch

from scripts.ib_local import successor_address_geometry as geometry
from scripts.ib_local.prospective_decision import lift_boundary
from test_relation_retention_use import configured


def test_snapshot_matches_original_halfspaces_cosine_and_rng(tmp_path):
    e,a=configured(tmp_path)
    try:
        e.step(torch.randn(8,4))
        model=e.model;bank=model.causal_relations
        raw=e._internal_source[None];z=model.encoder(raw).detach()
        # Exercise the production raw halfspace rule, including both sides.
        bank.scope_axes[0,0]=torch.randn_like(raw[0])
        bank.scope_bounds[0,0]=bank.scope_axes[0,0]@raw[0]
        bank.scope_signs[0,0]=1;bank.scope_valid[0,0]=True
        points=torch.cat((z,z+torch.randn(19,z.shape[-1])*.3))
        rng=torch.get_rng_state().clone();sql=a.db.total_changes
        packet=geometry.snapshot(model,raw,z)
        check=geometry.native_check(model,packet,points)
        assert check['identities_equal'] and check['max_similarity_error']<3e-6
        assert check['max_encoder_residual']<3e-6
        assert torch.equal(rng,torch.get_rng_state()) and a.db.total_changes==sql
        one=geometry.addresses(packet,points,chunk=1)
        batch=geometry.addresses(packet,points,chunk=7)
        assert torch.equal(one['identity'],batch['identity'])
        torch.testing.assert_close(one['similarity'],batch['similarity'],rtol=2e-5,atol=3e-6)
    finally:a.close()
