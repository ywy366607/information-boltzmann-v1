"""Change the label contract of one continuing body, without clearing it."""
import torch

SETTLEMENT_CODES = {'witness': 0, 'continuation': 1, 'continuation_joint': 3}
# 2 remains the mixed-label code used in existing snapshots.


def settlement_code(kind):
    return SETTLEMENT_CODES[kind]


def complete_action_labels(memory):
    """Slots whose existing labels cover only their declared action.

    A real measured mean may mix several actions. It remains an observation,
    but cannot be recalled as the conditional gain of the latest action.
    Missing provenance is conservatively unavailable for that read contract.
    """
    fields = ('count', 'wood_target_count', 'wood_option', 'option')
    if not all(hasattr(memory, name) for name in fields):
        return torch.zeros_like(memory.occupied)
    return (memory.occupied & (memory.count > 0)
            & (memory.wood_target_count == memory.count)
            & (memory.wood_option >= 0) & (memory.wood_option == memory.option))


@torch.no_grad()
def configure_reward_memory(engine, kind):
    """Choose what the existing progress slot means, without adding a pool.

    Legacy slots hold estimated TD returns. Observed slots hold mean measured
    gains. A complete, consistently labelled wood target can recover that mean;
    an unknown/mixed history remains stored but is not used as measured gain.
    """
    if kind not in ('td', 'observed'):
        raise ValueError(kind)
    previous = getattr(engine, 'reward_memory_kind', 'td')
    if previous == kind:
        return
    if previous != 'td':
        raise ValueError('reverting reward-memory semantics requires restoring its checkpoint')
    if getattr(engine, 'value_normalization', 'none') != 'none':
        raise ValueError('observed-gain repair is admitted on the unnormalized reference only')
    memory = engine.model.memory
    known = torch.zeros_like(memory.occupied)
    fields = ('wood_target', 'wood_target_count', 'wood_option', 'option', 'count')
    if all(hasattr(memory, name) for name in fields):
        known = (complete_action_labels(memory)
                 & torch.isfinite(memory.wood_target[:, 3]))
        versions = getattr(engine, 'wood_settlement_versions', None)
        if versions is not None:
            recognized = torch.zeros_like(known)
            for code in SETTLEMENT_CODES.values():
                recognized |= versions == code
            known &= recognized
            if not hasattr(engine, 'settlement_versions'):
                raise ValueError('wood provenance exists without reward provenance')
        else:
            # Untagged checkpoints predate objective migration and mean witness.
            if getattr(engine, 'settlement_kind', 'witness') != 'witness':
                raise ValueError('cannot recover untagged non-witness gains')
        memory.progress[known] = memory.wood_target[known, 3] / 10.
        if versions is not None:
            engine.settlement_versions[known] = versions[known]
    memory.progress_valid.copy_(known)
    engine.reward_memory_kind = kind
    engine.reward_memory_migration = dict(
        old_kind=previous, new_kind=kind, recovered_slots=int(known.sum()),
        unknown_occupied_slots=int((memory.occupied & ~known).sum()),
        extra_persistent_array_bytes=0,
        policy='Recover only complete consistent measured labels; preserve unknown content '
               'and mark unavailable for measured-gain reads. TD remains pending/critic credit.')


def configure_settlement(engine, kind):
    if kind not in SETTLEMENT_CODES:
        raise ValueError(f'unknown settlement: {kind}')
    previous = getattr(engine, 'settlement_kind', 'witness')
    if previous == kind:
        engine.settlement_kind = kind
        return
    if hasattr(engine, 'settlement_versions'):
        raise ValueError('switching an already migrated objective requires an explicit contract')
    size = engine.model.memory.capacity
    device = engine.model.memory.occupied.device
    # These describe existing labels, not new memory/credit slots. Source
    # snapshots use the historical witness objective (0); continuation is 1.
    engine.settlement_versions = torch.zeros(size, dtype=torch.uint8, device=device)
    engine.wood_settlement_versions = torch.zeros_like(engine.settlement_versions)
    engine.settlement_kind = kind
    engine.settlement_migration = dict(
        old_kind=previous, new_kind=kind, metadata_bytes=2*size,
        policy='retain all data, parameters and moments; exclude incompatible labels from reward reads/replay',
        pending_credit='pay previous reward once without cross-objective bootstrap',
        learned_priors='retain critic, wood means/errors, interoception and optimizer as transfer priors',
    )
