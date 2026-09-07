"""Restore a recent opponent pool while restarting games at a declared boundary."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_context(path, checkpoint, step):
    context=json.loads(Path(path).read_text())
    if context['schema']!=1 or context['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('Continuation context does not identify the loaded checkpoint')
    versions=[row['step'] for row in context['historical']]
    if not 1<=len(versions)<=4 or versions!=sorted(set(versions)) or versions[-1]>step:
        raise ValueError('Invalid historical continuation pool')
    if not step<context['next_checkpoint']<=step+context['checkpoint_every']:
        raise ValueError('Invalid next historical checkpoint boundary')
    for row in context['historical']:
        row['path']=str((Path(checkpoint).resolve().parent/row['path']).resolve())
        if digest(row['path'])!=row['sha256']:
            raise ValueError('Historical checkpoint identity changed')
    return context
