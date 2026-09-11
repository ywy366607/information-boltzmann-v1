"""Re-encode existing verified OWT documents with nanoGPT's GPT-2 BPE."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import tiktoken

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.source/'manifest.json').read_text());assert manifest['tokenizer']=='byte'
 enc=tiktoken.get_encoding('gpt2');a.output.mkdir(parents=True,exist_ok=False)
 output={**manifest,'tokenizer':'gpt2','vocab_size':enc.n_vocab,'bos_token':enc.eot_token,'eos_token':enc.eot_token,'source_manifest_sha256':digest(a.source/'manifest.json'),'boundary_policy':'encode_ordinary(document) followed by GPT2 EOT; no state reset','token_counts':{},'files':{}}
 for split in ['train','validation','test']:
  path=a.source/f'{split}.npy';assert digest(path)==manifest['files'][path.name]
  arr=np.load(path,mmap_mode='r');ends=np.flatnonzero(arr==2);start=0;ids=[];docs=[r for r in manifest['documents'] if r['split']==split]
  assert len(ends)==len(docs)
  for end,doc in zip(ends,docs):
   raw=np.asarray(arr[start:end]-4,dtype=np.uint8).tobytes();assert hashlib.sha256(raw).hexdigest()==doc['sha256']
   text=raw.decode('utf-8');encoded=enc.encode_ordinary(text);assert enc.decode(encoded)==text
   ids.extend(encoded);ids.append(enc.eot_token);assert arr[end+1]==1;start=end+2
  assert start==len(arr)
  dest=a.output/f'{split}.npy';np.save(dest,np.asarray(ids,dtype=np.uint16));output['files'][dest.name]=digest(dest);output['token_counts'][split]=len(ids)
  print(split,len(ids),flush=True)
 (a.output/'tokenizer.json').write_text(json.dumps({'encoding':'gpt2','library':'tiktoken','n_vocab':enc.n_vocab,'eot_token':enc.eot_token,'ordinary_text':True},indent=2))
 output['files']['tokenizer.json']=digest(a.output/'tokenizer.json')
 (a.output/'manifest.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
if __name__=='__main__':main()
