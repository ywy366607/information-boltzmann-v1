"""Official NIAH function with one background block for byte-window admission.

The official length-search wrapper has a minimum of five background blocks and
can retry forever below that size. Call its unmodified generate_input_output(1)
instead. This is a named short-layout variant, not the official length benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('official','model','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    script=args.official/'scripts/data/synthetic/niah.py'
    constants=runpy.run_path(str(script.parent/'constants.py'))['TASKS']['niah']
    argv=sys.argv
    sys.path.insert(0,str(script.parent))
    try:
        sys.argv=[str(script),'--save_dir',str(args.output),'--save_name','niah_single_1',
            '--tokenizer_path',str(args.model),'--tokenizer_type','hf','--max_seq_length','512',
            '--tokens_to_generate','128','--num_samples',str(args.samples),'--random_seed',str(args.seed),
            '--type_haystack','noise','--type_needle_k','words','--type_needle_v','numbers',
            '--template',constants['template']+constants['answer_prefix']]
        module=runpy.run_path(str(script),run_name='official_niah_library')
    finally:
        sys.argv=argv
    path=args.output/'test.jsonl'
    lengths=[]
    with path.open('w',encoding='utf-8') as stream:
        for i in range(args.samples):
            text,answer=module['generate_input_output'](1)
            boundary=text.rfind(constants['answer_prefix'][:10])
            if boundary<0:
                raise ValueError('official answer prefix missing')
            row=dict(index=i,input=text[:boundary],answer_prefix=text[boundary:],outputs=answer)
            stream.write(json.dumps(row)+'\n')
            lengths.append(len(text.encode('utf-8'))+len((' '+', '.join(answer)).encode('utf-8'))+1)
    report=dict(status='complete',scope=__doc__,seed=args.seed,samples=args.samples,
        official_revision=subprocess.check_output(['git','-C',str(args.official),'rev-parse','HEAD'],text=True).strip(),
        generator_sha256=sha(script),tokenizer=str(args.model),haystack_count=1,
        input_with_answer_bytes_min=min(lengths),input_with_answer_bytes_max=max(lengths),
        eligible_512=sum(n<=512 for n in lengths),
        data=[dict(task='niah_single_1',family='niah',path=str(path),sha256=sha(path),
                   length=None,max_new_tokens=128,haystack_count=1)])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
