"""Immutable protocol, RNG namespaces, source identity and atomic receipts."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from .common import ROOT,canonical,read,sha,require,safe_id,safe_child,write_csv,write_text

CONFIG=ROOT/'configs/lfbo_gp_unified/protocol_v2.json'
OUTPUT=ROOT/'results/lfbo_gp_unified'


def save(path,value,replace=False):
    """Publish complete JSON atomically, refusing to clobber checkpoints."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        with temp.open('x') as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        if replace:os.replace(temp,path)
        else:os.link(temp,path)
    finally:
        if temp.exists():temp.unlink()


def seal(path,value):save(path,dict(payload=value,sha256=canonical(value)))


def unseal(path):
    item=read(path);require(canonical(item['payload'])==item['sha256'],'Checkpoint checksum: '+str(path))
    return item['payload']


def seed(family,stream,task):
    raw=json.dumps([family,'unified-v1',stream,task],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4],'little')


def config():
    c=read(CONFIG);require(sha(CONFIG)==CONFIG.with_suffix('.sha256').read_text().strip(),'Protocol config checksum changed')
    require(c['dataset']['seeds']==list(range(44,144)) and c['dataset']['tasks']==200,'Invalid final population')
    require(c['classification']['alpha']==.8 and c['classification']['gamma']==1/3,'Unexpected metric settings')
    for item in c['provenance']['sources']:
        p=ROOT/item['local_path'];raw=p.read_bytes()
        require(sha(p)==item['sha256'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['git_blob_sha'],'Pinned author source changed')
    for item in c['provenance']['additional_audit_inputs']:require(sha(ROOT/item['path'])==item['sha256'],'Audit evidence changed')
    return c


def environment(c):
    rt=c['runtime'];packages={n:importlib.metadata.version(n) for n in rt['packages']}
    require(packages==rt['packages'] and sys.version.split()[0]==rt['python'],'Use locked .venv Python/dependencies')
    return dict(python=sys.version,executable=sys.executable,packages=packages)


def source_identity(c):
    files=list((ROOT/'experiments/lfbo_gp_unified').glob('*.py'))+[CONFIG,CONFIG.with_suffix('.sha256'),ROOT/c['dataset']['split']['registered_manifest']]
    files += [ROOT/p for p in ('main.py','requirements.txt','models/lfbo.py','models/gp.py','models/base_model.py','models/__init__.py',
        'experiments/__init__.py','experiments/metrics.py','configs/lfbo_gp_unified/source_contract.json',
        'vendor/metagamebo/benchmarks/forrester.py')]
    files += [ROOT/e['local_path'] for e in c['provenance']['sources']]
    files += [ROOT/e['path'] for e in c['provenance']['additional_audit_inputs']]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}


def lock(folder,job,c):
    sources=source_identity(c)
    for name in sources:
        p=folder/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    payload=dict(sources=sources,files={n:sha(folder/n) for n in ('config.json','job.json','environment.json','registration.json')})
    require(sources==source_identity(c),'Sources changed during registration')
    save(folder/'protocol_lock.json',dict(payload=payload,protocol_id=canonical(payload)))


def check_lock(folder,current=False):
    folder=Path(folder);item=read(folder/'protocol_lock.json');p=item['payload']
    require(canonical(p)==item['protocol_id'],'Protocol lock checksum')
    for name,digest in p['files'].items():require(sha(safe_child(folder,name))==digest,'Locked file changed: '+name)
    for name,digest in p['sources'].items():
        require(sha(safe_child(folder/'source_snapshot',name))==digest,'Source snapshot changed: '+name)
        if current:require(sha(ROOT/name)==digest,'Current source changed: '+name)
    return item['protocol_id']


def methods(c,track):return c['classification']['track_'+track]['methods']
