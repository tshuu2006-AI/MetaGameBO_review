"""Native family and observation streams; no SHA-reseeded target population."""
from __future__ import annotations
import numpy as np
from .objective import AuthorTarget


def plan(seed, tasks):
    rng = np.random.default_rng(seed)
    descriptors=[]
    for uid in range(tasks+1):
        descriptors.append(dict(uid=uid, task_id=f'meta{uid:04d}' if uid else 'target0000',
            a=float(rng.uniform(.2,3.)),b=float(rng.uniform(-5.,15.)),c=float(rng.uniform(-5.,5.))))
    coordinates=np.random.default_rng(seed)
    X={d['task_id']:coordinates.uniform(0,1,(32,1)).tolist() for d in descriptors[1:]}
    return dict(seed=seed,target=descriptors[0],meta_tasks=descriptors[1:],X=X,
        semantics='native target-first descriptor stream and independent coordinate RNG reset to SAME seed')


def observe(manifest):
    raw={}
    for spec in manifest['meta_tasks']:
        X=manifest['X'][spec['task_id']];target=AuthorTarget(spec);y=target.evaluate(np.asarray(X)).tolist()
        if target.evaluations!=32: raise ValueError('Unexpected data observation count')
        raw[spec['task_id']]=dict(X=X,y=y)
    return raw
