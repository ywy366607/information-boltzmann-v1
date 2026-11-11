"""Bounded arithmetic program for an observed local response relation.

Candidate programs are data, never Python/Lean source. This research-side
interpreter does not mutate a body and does not claim observational causality.
"""
from dataclasses import dataclass

MAX_OPS = 16
MAX_INPUTS = 4
MAX_LITERAL = 10**9
MAX_VALUE = 10**15


@dataclass(frozen=True)
class Instruction:
    op: str
    arg: int = 0


def parse_candidate(raw, input_count):
    """Accept a model-proposed JSON instruction list without executing code."""
    if not isinstance(raw, list) or len(raw) > MAX_OPS:
        raise ValueError('candidate is not a bounded instruction list')
    program=[]
    for item in raw:
        if not isinstance(item,dict) or set(item) != {'op','arg'}:
            raise ValueError('instruction schema mismatch')
        if type(item['op']) is not str or type(item['arg']) is not int:
            raise ValueError('instruction fields must be text and integer')
        program.append(Instruction(item['op'],item['arg']))
    check_program(program,input_count)
    return tuple(program)


def check_program(program, input_count):
    if not 1 <= input_count <= MAX_INPUTS or not 1 <= len(program) <= MAX_OPS:
        raise ValueError('program or input budget exceeded')
    depth = 0
    for inst in program:
        if not isinstance(inst, Instruction) or type(inst.arg) is not int:
            raise ValueError('invalid typed instruction')
        if inst.op == 'input':
            if not 0 <= inst.arg < input_count:
                raise ValueError('unavailable input')
            depth += 1
        elif inst.op == 'const':
            if abs(inst.arg) > MAX_LITERAL:
                raise ValueError('literal budget exceeded')
            depth += 1
        elif inst.op in ('add', 'mul'):
            if inst.arg != 0 or depth < 2:
                raise ValueError('invalid binary operation')
            depth -= 1
        else:
            raise ValueError('operation not permitted')
    if depth != 1:
        raise ValueError('program must return exactly one value')
    return True


def execute(program, inputs):
    check_program(program,len(inputs))
    if any(type(x) is not int or abs(x) > MAX_LITERAL for x in inputs):
        raise ValueError('input out of range')
    stack=[]
    for inst in program:
        if inst.op == 'input':
            stack.append(inputs[inst.arg])
        elif inst.op == 'const':
            stack.append(inst.arg)
        else:
            right,left=stack.pop(),stack.pop()
            value=left+right if inst.op == 'add' else left*right
            if abs(value)>MAX_VALUE:
                raise OverflowError('arithmetic budget exceeded')
            stack.append(value)
    return stack[0]


def linear_program(bias, teacher_slope, layer_effect):
    """Encode an estimated relation as a short, editable program."""
    p=(Instruction('const',bias),Instruction('input',0),Instruction('const',teacher_slope),
       Instruction('mul'),Instruction('add'),Instruction('input',1),
       Instruction('const',layer_effect),Instruction('mul'),Instruction('add'))
    check_program(p,2)
    return p
