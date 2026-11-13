"""Materialize the one registered RULER/real-OWT scalar-credit curriculum."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.audit_pretrained_continued_answers import insert_interference
from scripts.ib_local.run_pretrained_gdn_cross_window import read_text
from scripts.ib_local.ruler_readout_adaptation import validate_splits


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[2]
    output=root/'results/cgb_water_last_attempt_20261109/scalar_curriculum.json'
    if output.exists():
        raise FileExistsError(output)
    data_root=root/'results/cgb_ruler_adaptation_20261109'
    manifests=[data_root/name/'manifest.json' for name in ('data_train','data_validation')]
    rows=[]
    for p in manifests:
        m=json.loads(p.read_text())
        cell=m['data'][0]
        if m['status']!='complete' or sha(root/cell['path'])!=cell['sha256']:
            raise ValueError('registered RULER source changed')
        rows.append([json.loads(line) for line in (root/cell['path']).read_text().splitlines()])
    training,validation=rows
    validate_splits(training,validation)
    tok=AutoTokenizer.from_pretrained('E:/cgb_gdn340_probe/model',local_files_only=True)
    dataset=root/'data/ib_owt_10000'
    offset=json.loads((dataset/'tokenizer.json').read_text())['offset']
    train_text,train_sha=read_text(dataset/'train.npy',offset,17_000_000,length=1_500_000)
    val_text,val_sha=read_text(dataset/'train.npy',offset,19_000_000,length=196_608)
    traffic=tok.encode(train_text,add_special_tokens=False)
    val_gap=tok.encode(val_text,add_special_tokens=False)[:256]
    if len(traffic)<48*384 or len(val_gap)!=256:
        raise ValueError('insufficient real interference')
    train=[]
    for i,row in enumerate(training[:48]):
        gap=traffic[i*384:i*384+256]
        tail=traffic[i*384+256:(i+1)*384]
        prefix,gold=encode_case(tok,insert_interference(row,tok.decode(gap,skip_special_tokens=True)))
        train.append(dict(index=row['index'],prefix=prefix,gold=gold,tail=tail))
    evaluation=[]
    for row in validation:
        for gap in (0,256):
            effective=row if not gap else insert_interference(row,tok.decode(val_gap,skip_special_tokens=True))
            prefix,gold=encode_case(tok,effective)
            evaluation.append(dict(index=row['index'],gap=gap,prefix=prefix,gold=gold,
                                   outputs=row['outputs']))
    total=sum(len(x['prefix'])+len(x['gold'])+1+len(x['tail']) for x in train)
    if len(train)!=48 or len(evaluation)!=16 or total>=60000:
        raise ValueError('registered case/token budget violated')
    value=dict(status='prepared',source_manifests={str(p):sha(p) for p in manifests},
        task_seeds={'train':91109,'validation':91110},owt_revision='433fe0f44ed7894fea29c08b3202aa348ccc6369',
        owt_train_raw_start=17_000_000,owt_train_text_sha256=train_sha,
        owt_validation_raw_start=19_000_000,owt_validation_text_sha256=val_sha,
        tokenizer='E:/cgb_gdn340_probe/model',eos=tok.eos_token_id,
        training=train,evaluation=evaluation,train_arrival_tokens=total,
        supervised_tokens=sum(len(x['gold'])+1 for x in train),
        max_actual_windows=512,preparation_script_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(value),encoding='utf-8')
    print(json.dumps(dict(path=str(output),sha256=sha(output),training_cases=48,
        evaluation_cases=16,arrival_tokens=total,supervised_tokens=value['supervised_tokens'],
        max_windows_per_arm=512)))


if __name__=='__main__':
    main()
