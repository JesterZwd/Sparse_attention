"""Obtain and verify the pinned Meta-Llama-3.1-8B-Instruct mirror snapshot."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener, urlopen

from scripts.download_model import download_file

REPO='meta-llama/Llama-3.1-8B-Instruct'
REV='0e9e39f249a16976918f6564b8830bc894c89659'
MIRROR='https://modelscope.cn/api/v1/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/repo'
TARGET=Path('models/Llama-3.1-8B-Instruct')
CACHE=Path('.cache/round13-cloud/llama32k')
DIRECT=build_opener(ProxyHandler({}))

def git_blob(data):
    return sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()

def metadata_snapshot(filename, endpoint, opener):
    path=CACHE/filename
    if not path.exists():
        with opener(endpoint,timeout=25) as response:
            data=response.read()
        temporary=path.with_suffix('.download')
        temporary.write_bytes(data)
        temporary.replace(path)
    return json.loads(path.read_text())

tree=metadata_snapshot('official-tree.json', f'https://huggingface.co/api/models/{REPO}/tree/{REV}', urlopen)
mirror=metadata_snapshot('mirror-tree.json', MIRROR+'/files?Revision=master&Root=', DIRECT.open)
official={x['path']:x for x in tree if x['type']=='file'}
peers={x['Name']:x for x in mirror['Data']['Files']}
metadata=('LICENSE','USE_POLICY.md','config.json','generation_config.json',
          'model.safetensors.index.json','tokenizer.json','tokenizer_config.json')
names=[name for name in metadata if name!='USE_POLICY.md' or name in official]
shards=sorted(name for name in official if name.endswith('.safetensors'))
assert len(shards)==4
for name in shards:
    ref, obj=official[name],peers[name]
    pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{obj['Sha256']}\nsize {obj['Size']}\n".encode()
    assert git_blob(pointer)==ref['oid'] and len(pointer)==ref['lfs']['pointerSize'] and ref['size']==obj['Size'], name
print('All four ModelScope shard hashes match Meta Git LFS blob IDs',flush=True)

def fetch(name):
    item=peers[name]
    url=MIRROR+'?'+urlencode(dict(Revision=item['Revision'],FilePath=name))
    path=TARGET/name
    download_file(item,path,url)
    if name not in shards:
        blob=path.read_bytes()
        assert git_blob(blob)==official[name]['oid'], name
    return dict(Path=name,Sha256=item['Sha256'])

with ThreadPoolExecutor(max_workers=4) as pool:
    files=list(pool.map(fetch,names))
    files.extend(pool.map(fetch,shards))
receipt=dict(repository=REPO,service='ModelScope',mirror='LLM-Research/Meta-Llama-3.1-8B-Instruct',
             revision=REV,verified=True,files=files,verification='SHA256 plus official Git/LFS blob IDs')
target=TARGET/'source_manifest.json'
draft=target.with_suffix('.tmp')
draft.write_text(json.dumps(receipt,indent=2)+'\n')
draft.replace(target)
print('Pinned Llama checkpoint verified',flush=True)
