import socket
_original_getaddrinfo=socket.getaddrinfo
socket.getaddrinfo=lambda host, port, family=0, type=0, proto=0, flags=0: _original_getaddrinfo(host,port,socket.AF_INET,type,proto,flags)
import concurrent.futures as cf, hashlib, json, os, pathlib, time, urllib.request
import html2text
from bs4 import BeautifulSoup
root = pathlib.Path('.cache/ruler-corpus')
urls = pathlib.Path('.cache/ruler-upstream/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt').read_text().splitlines()

def fetch(original):
    url = original.replace('http://www.paulgraham.com/', 'https://www.paulgraham.com/')
    url = url.replace('https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/', 'https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/')
    filename = original.rsplit('/',1)[-1]
    path = root / 'raw' / filename
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        for retry in range(4):
            try:
                with urllib.request.urlopen(url,timeout=25) as response: data=response.read()
                path.write_bytes(data)
                break
            except Exception:
                if retry == 3: raise
                time.sleep(1)
    data=path.read_bytes()
    if '.html' in original:
        h=html2text.HTML2Text()
        h.ignore_images=True; h.ignore_tables=True; h.escape_all=True; h.reference_links=False; h.mark_code=False
        parsed=h.handle(str(BeautifulSoup(data.decode('unicode_escape','utf-8'),'html.parser').find('font')))
        kind='html'; name=filename.replace('.html','.txt')
    else:
        parsed=data.decode('utf-8');kind='repo';name=filename
    return {'original_url':original,'fetched_url':url,'raw_sha256':hashlib.sha256(data).hexdigest(),'kind':kind,'name':name,'text':parsed}

with cf.ThreadPoolExecutor(max_workers=12) as pool:
    rows=[]
    for i,row in enumerate(pool.map(fetch,urls)):
        rows.append(row)
        print(i+1,len(urls),row['name'],flush=True)
rows.sort(key=lambda row:(0 if row['kind']=='repo' else 1,row['name']))
text=''.join(row['text'] for row in rows)
(root/'PaulGrahamEssays.json').write_text(json.dumps({'text':text}))
(root/'sources.json').write_text(json.dumps([{k:v for k,v in row.items() if k!='text'} for row in rows],indent=2))
print('CORPUS',hashlib.sha256((root/'PaulGrahamEssays.json').read_bytes()).hexdigest(),flush=True)
