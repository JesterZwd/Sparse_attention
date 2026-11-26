import socket
_original_getaddrinfo=socket.getaddrinfo
socket.getaddrinfo=lambda host, port, family=0, type=0, proto=0, flags=0: _original_getaddrinfo(host,port,socket.AF_INET,type,proto,flags)
import concurrent.futures, hashlib, io, json, pathlib, urllib.request, zipfile
url='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip'
size=4319076

def fetch(a,b):
    request=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data=response.read()
        if len(data)!=b-a+1: raise ValueError((len(data),a,b,response.headers))
        return data

archive=bytearray(size)
archive[-65536:]=fetch(size-65536,size-1)
z=zipfile.ZipFile(io.BytesIO(archive))
infos=[i for i in z.infolist() if i.filename.startswith('punkt_tab/english/') or i.filename=='punkt_tab/README']

def member(info):
    start=info.header_offset
    data=fetch(start,min(size-1,start+info.compress_size+1023))
    return start,data

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    for start,data in pool.map(member,infos): archive[start:start+len(data)]=data
z=zipfile.ZipFile(io.BytesIO(archive))
root=pathlib.Path('.cache/ruler-nltk/tokenizers')
manifest={}
for info in infos:
    if info.is_dir(): continue
    path=root/info.filename
    path.parent.mkdir(exist_ok=True,parents=True)
    data=z.read(info)
    path.write_bytes(data)
    manifest[info.filename]=hashlib.sha256(data).hexdigest()
(root.parent/'english_source.json').write_text(json.dumps({'source_url':url,'source_etag':'9b42eb53d8a886811f083a4ad1a95b818d401c4b5ad1cfec7b2f06c74f838a0a','files':manifest},indent=2))
print(manifest,flush=True)
