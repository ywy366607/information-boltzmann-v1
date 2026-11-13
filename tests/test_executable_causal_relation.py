import pytest

from scripts.ib_local.executable_causal_relation import Instruction, check_program, execute, linear_program, parse_candidate


def test_bounded_program_executes_a_relation_without_history_or_side_effects():
    code=linear_program(-12,3,4)
    assert execute(code,[7,0]) == 9
    assert execute(code,[7,1]) == 13
    assert len(code) == 9


def test_untrusted_program_cannot_access_missing_state_or_exceed_budget():
    with pytest.raises(ValueError):
        check_program((Instruction('input',2),),2)
    with pytest.raises(ValueError):
        check_program((Instruction('exec',0),),1)
    with pytest.raises(ValueError):
        check_program(tuple(Instruction('const',0) for _ in range(17)),1)
    with pytest.raises(OverflowError):
        execute((Instruction('const',10**9),Instruction('const',10**9),Instruction('mul')), [0])


def test_model_candidate_is_data_with_strict_schema_and_no_source_execution():
    program=parse_candidate([{'op':'input','arg':0},{'op':'const','arg':2},
                             {'op':'mul','arg':0}],1)
    assert execute(program,[9]) == 18
    for raw in ([{'op':'input','arg':True}], [{'op':'import os','arg':0}],
                [{'op':'const','arg':1,'code':'untrusted'}]):
        with pytest.raises(ValueError):
            parse_candidate(raw,1)
