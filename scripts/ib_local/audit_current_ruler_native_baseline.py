"""Same frozen RULER diagnostic, with only extra experience reads suppressed.

The original S/conv/weights/experience contents are inherited unchanged. This
intervention distinguishes a failed extra read from lost native competence.
"""
import hashlib
from pathlib import Path


def main():
    from scripts.ib_local import evaluate_pretrained_gdn_ruler as evaluation
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local.probe_continuous_readiness import native_only
    from scripts.ib_local import audit_current_ruler_readiness as audit
    original_run = evaluation.run_case
    original_write = old.write_json
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def native_run(learner, *args, **kwargs):
        with native_only(learner, True):
            return original_run(learner, *args, **kwargs)

    def write(path, value):
        if Path(path).name == 'report.json':
            value['extra_experience_read_disabled'] = True
            value['intervention'] = 'zero only extra replay read; native S/conv/parameters and source unchanged'
            value['intervention_script_sha256'] = wrapper_sha
        return original_write(path, value)

    evaluation.run_case = native_run
    old.write_json = write
    try:
        audit.main()
    finally:
        evaluation.run_case = original_run
        old.write_json = original_write


if __name__ == '__main__':
    main()
