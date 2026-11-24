"""Finish the already-running private measurement without interactive polling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'scratch/native_transport_curl_use_2401_20261117'
OUTPUT = ROOT/'results/published/cgb_native_transport_curl_use_2401_20261117.json'


def status(**data):
    (WORK/'finalizer_status.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    status(status='waiting_for_existing_three_arms', pid=os.getpid())
    deadline = time.monotonic()+3600
    while time.monotonic() < deadline:
        state = json.loads((WORK/'status.json').read_text(encoding='utf8'))
        if state['status'] == 'failed':
            status(status='experiment_failed', experiment=state)
            return
        if state['status'] == 'completed':
            break
        time.sleep(20)
    else:
        status(status='waiting_timeout', experiment=state)
        return
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    status(status='analyzing')
    with (WORK/'analysis.stdout.log').open('w', encoding='utf8') as out, (WORK/'analysis.stderr.log').open('w', encoding='utf8') as err:
        code = subprocess.call([sys.executable, '-m', 'scripts.analyze_native_transport_curl_use', str(WORK), str(OUTPUT)],
                               cwd=ROOT, env=env, stdout=out, stderr=err)
    if code:
        status(status='analysis_failed', exit_code=code, log=str(WORK/'analysis.stderr.log'))
        return
    result = json.loads(OUTPUT.read_text(encoding='utf8'))
    doc = ROOT/'docs/information_boltzmann/CGB_NATIVE_TRANSPORT_CURL_USE_20261117.md'
    section = '\n\n## 后台完成结果\n\n三条完成分支各8次真实文本交互、32次更新；主源哈希核对通过。完整数值见 `results/published/cgb_native_transport_curl_use_2401_20261117.json`。\n\n'
    section += '| 比较 | 初次协调预测差L2 | 其中精度权重变化项L2 | 初次内部选择概率差 | 终点Xo差L2 | 外部选择改变 |\n|---|---:|---:|---:|---:|---:|\n'
    for key, comparison in result['comparisons'].items():
        first = result['first_path_comparisons'][key]
        section += f"| {key} | {first['coordinated_mean']['norm']:.8g} | {first['consensus_decomposition']['trust_change_norm']:.8g} | {first['internal_probability_delta']:.8g} | {comparison['final_X_o_difference_norm']:.8g} | {comparison['changed_choices']}/8 |\n"
    text = doc.read_text(encoding='utf8')
    if '## 后台完成结果' not in text:
        doc.write_text(text+section, encoding='utf8')
    status(status='completed', result=str(OUTPUT), document=str(doc), main_source_hashes_preserved=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status(status='finalizer_failed', error=repr(error))
        raise
