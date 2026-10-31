import pytest
import math
import numpy as np
from fine_grain.information_boltzmann.source_sink_cfe import SourceSinkConstructalEngine

def test_source_sink_cfe_initialization():
    engine = SourceSinkConstructalEngine(initial_internal_slots=5, max_slots=20)
    assert engine.num_active == 8 # 2 sources + 1 sink + 5 internal
    assert engine.active_mask[0] and engine.active_mask[1] and engine.active_mask[2]
    # 验证几何位置: 源在左 x=-0.85, 汇在右 x=+0.85
    assert engine.pos_3d[0, 0] < -0.8
    assert engine.pos_3d[1, 0] < -0.8
    assert engine.pos_3d[2, 0] > 0.8

def test_source_sink_cfe_forward():
    engine = SourceSinkConstructalEngine(initial_internal_slots=4, max_slots=16)
    out, v_all = engine.forward_flow(2.0, 3.0)
    assert isinstance(out, float)
    assert len(v_all) == 16
    assert v_all[0] == 2.0
    assert v_all[1] == 3.0

def test_source_sink_cfe_step_and_constructal_split():
    engine = SourceSinkConstructalEngine(initial_internal_slots=4, max_slots=16, lr=0.1)
    init_active = engine.num_active
    # 人为施加极大的任务误差，验证是否引发汇端反压与 Hopf 余乘 Delta 分裂
    res = engine.step(a_val=3.0, b_val=3.0, target_val=20.0)
    assert res['error'] > 0
    assert res['resistance'] > 0
    if res['splits'] > 0:
        assert engine.num_active == init_active + 1
