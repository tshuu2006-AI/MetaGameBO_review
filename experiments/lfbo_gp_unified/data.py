"""One raw family manifest, training-only constants and shared rows for both methods."""
from __future__ import annotations
import copy
import functools
import numpy as np
from .generator import plan,observe
from .source_ei import prepare_ei_rows
from .protocol import require,seed,canonical,ROOT,read,sha


@functools.lru_cache(maxsize=1)
def registered(path,digest):
    path=ROOT/path;require(sha(path)==digest,'Registered split manifest checksum')
    return {(r['family_seed'],r['task_id']):r for r in read(path)['splits']}


def prepare_task(spec,raw,family,c):
    X=np.asarray(raw['X'],float);y=np.asarray(raw['y'],float);n=c['dataset']['observations']
    require(X.shape==(n,1) and y.shape==(n,) and np.isfinite(X).all() and np.isfinite(y).all(),'Invalid raw observations')
    require(np.all((X>=0)&(X<=1)),'Raw domain')
    rng=np.random.default_rng(seed(family,c['dataset']['split']['namespace'],spec['task_id']));before=copy.deepcopy(rng.bit_generator.state)
    permutation=rng.permutation(n).tolist();count=c['dataset']['split']['train_count'];tr,va=permutation[:count],permutation[count:]
    spec_split=c['dataset']['split'];pre=registered(spec_split['registered_manifest'],spec_split['manifest_sha256'])[(family,spec['task_id'])]
    require(pre['train']==tr and pre['validation']==va and pre['seed']==seed(family,'raw-split',spec['task_id']),'Split differs from preregistered indices')
    require(not set(map(tuple,X[tr]))&set(map(tuple,X[va])),'Raw coordinate overlap; refusing replacement/resampling')
    values=y[tr];tau=float(np.quantile(np.unique(values),c['classification']['gamma']))
    good=values<tau;m=int(good.sum());U=float(np.maximum(tau-values,0).mean());balance=(len(tr)+m)/(2*len(tr))
    require(0<m<len(tr) and U>0,'Degenerate training event support/U; no seed replacement')
    constants=dict(tau=tau,U_train=U,c=balance,n_train=len(tr),n_good=m,
        positive_coefficient=balance/U,background_weight=balance,normalization_source_indices=tr)
    def rows(indices,augmented):
        positive=[i for i in indices if y[i]<tau] if augmented else []
        out=[]
        for i in positive:
            out.append(dict(point_index=i,label=1,weight=balance*(tau-float(y[i]))/U,role='positive_improvement'))
        for i in indices:
            out.append(dict(point_index=i,label=0 if augmented else int(y[i]<tau),weight=balance if augmented else 1.,role='background_negative' if augmented else 'raw_event'))
        return out
    tracks={a:dict(training=rows(tr,a=='B'),validation=rows(va,a=='B')) for a in ('A','B')}
    # Verify the training recipe against the checksum-pinned original component.
    original=prepare_ei_rows(X[tr],values,c['classification']['gamma']);b=tracks['B']['training']
    np.testing.assert_array_equal(X[[r['point_index'] for r in b]],original['X'])
    np.testing.assert_array_equal([r['label'] for r in b],original['labels'])
    np.testing.assert_allclose([r['weight'] for r in b],original['weights'],rtol=1e-14,atol=1e-14)
    return dict(task_id=spec['task_id'],uid=spec['uid'],descriptor=spec,raw=raw,
        split=dict(seed=seed(family,'raw-split',spec['task_id']),train=tr,validation=va,RNG_before=before,RNG_after=copy.deepcopy(rng.bit_generator.state)),
        constants=constants,tracks=tracks,streams={s:seed(family,s,spec['task_id']) for s in c['dataset']['rng']['streams']},raw_overlap=0)


def family_manifest(family,tasks,c):
    manifest=plan(family,tasks);raw=observe(manifest)
    # States recorded from the same source generator streams, independently replayed.
    descriptors=np.random.default_rng(family);coordinates=np.random.default_rng(family)
    before=dict(descriptors=copy.deepcopy(descriptors.bit_generator.state),coordinates=copy.deepcopy(coordinates.bit_generator.state))
    for _ in range(tasks+1):descriptors.uniform(.2,3);descriptors.uniform(-5,15);descriptors.uniform(-5,5)
    coordinates.uniform(0,1,(tasks*32,1))
    after=dict(descriptors=descriptors.bit_generator.state,coordinates=coordinates.bit_generator.state)
    entries=[prepare_task(s,raw[s['task_id']],family,c) for s in manifest['meta_tasks']]
    return dict(version=c['dataset']['version'],family_seed=family,plan=manifest,tasks=entries,
        RNG_before=before,RNG_after=after,objective_observations=32*tasks,excluded_target_uid=0,content_id=canonical(manifest))
