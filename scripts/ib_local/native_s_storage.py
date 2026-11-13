"""Passive joint-archive policy; leaves local replays and learning untouched."""
from dataclasses import fields, is_dataclass


def stop_future_joint_archives(learner):
    """Stop new all-layer audit copies; never alter existing observations."""
    previous = bool(learner.joint_capture_enabled)
    learner.joint_capture_enabled = False
    return dict(previous_joint_capture_enabled=previous, joint_capture_enabled=False,
                existing_capsules_preserved=True, local_capture_preserved=True,
                qualification_preserved=True, model_or_optimizer_changed=False)


def learning_state_view(learner_state, stream_state):
    """Values used by the current local-replay learner, excluding passive audit.

    Only the root joint_capture_enabled flag and LocalExperience.joint differ
    between policies. Keep local inputs/boundaries, weights, optimizer, clocks,
    RNG, stream and qualification. No tensor values are cloned or changed.
    This is an equivalence diagnostic, not the full fault-recovery state.
    """
    def view(value):
        if is_dataclass(value):
            name = type(value).__name__
            return dict(dataclass=name, fields={f.name: view(getattr(value, f.name))
                        for f in fields(value)
                        if not (name == 'LocalExperience' and f.name == 'joint')})
        if isinstance(value, dict):
            return {k: view(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(view(v) for v in value)
        if isinstance(value, list):
            return [view(v) for v in value]
        return value
    return dict(learner=view({k:v for k,v in learner_state.items()
                             if k != 'joint_capture_enabled'}), stream=view(stream_state))
