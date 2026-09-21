import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()


def safe_path(root, relative):
    root=Path(root).resolve()
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('unsafe_artifact_path')
    resolved=(root/p).resolve()
    if not resolved.is_relative_to(root) or resolved == root: raise ValueError('unsafe_artifact_path')
    return resolved


def write_artifact(root,path,data,format='application/json'):
    target=safe_path(root,path)
    target.parent.mkdir(parents=True,exist_ok=True)
    raw=data if isinstance(data,bytes) else canonical(data)
    with target.open('xb') as f: f.write(raw)
    return {'path':path,'format':format,'sha256':digest(raw)}


def verify_artifacts(root, artifacts):
    for name,entry in artifacts.items():
        p=safe_path(root,entry['path'])
        if not p.is_file(): raise ValueError(f'missing_artifact: {name}')
        if digest(p.read_bytes()) != entry['sha256']: raise ValueError(f'artifact_hash_mismatch: {name}')
