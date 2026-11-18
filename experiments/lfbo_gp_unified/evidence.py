"""Compact measured arrays and read-only checks; never fit or generate data."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from .protocol import read, save, sha, require, safe_child, methods, unseal
from .verification import verify, family, close
from .metrics import aggregate, task_summary


def export_evidence(folder, destination):
    import numpy as np
    folder, destination = Path(folder), Path(destination)
    require(not destination.exists(), 'Evidence destination exists; choose a new directory')
    summary, stats = verify(folder)
    job, c = read(folder/'job.json'), read(folder/'config.json')
    require(job['evaluation'] == 'classifier', 'Evidence export supports classifier runs only')
    names = methods(c, job['track'])
    data = {k: [] for k in ('X', 'y', 'train_indices', 'validation_indices',
                           'family_index', 'seed', 'task_id', 'tau', 'U_train', 'balance')}
    fields = ('point_index', 'label', 'weight', 'probability', 'log_p', 'log_not_p',
              'logit', 'BCE', 'weighted_BCE')
    predictions = {k: [] for k in fields}
    predictions.update(posterior_mean=[], latent_variance=[], offsets=[0], task_losses=[])
    checkpoint_sources = []
    for i in range(len(job['seeds'])):
        _, units, dataset = family(folder, i)
        lookup = {(task, r['method']): r for task, r in units}
        for t in dataset['tasks']:
            for key, value in dict(X=t['raw']['X'], y=t['raw']['y'],
                    train_indices=t['split']['train'], validation_indices=t['split']['validation'],
                    family_index=i, seed=job['seeds'][i], task_id=t['task_id'],
                    tau=t['constants']['tau'], U_train=t['constants']['U_train'],
                    balance=t['constants']['c']).items():
                data[key].append(value)
            losses = []
            for method in names:
                r = lookup[t['task_id'], method]
                losses.append(r['task_loss'])
                for row in r['predictions']:
                    for field in fields:
                        predictions[field].append(row[field])
                    # NaN explicitly marks a field not applicable to RF, never a measured value.
                    for field in ('posterior_mean', 'latent_variance'):
                        predictions[field].append(row.get(field, float('nan')))
                predictions['offsets'].append(len(predictions['label']))
                if len(data['task_id']) == 1 and 'model_checkpoint' in r:
                    from .checkpoints import model_path
                    path = model_path(folder/'families'/f'{i:03d}', t['task_id'], method)
                    checkpoint_sources.append((method, path, r['model_checkpoint']))
            predictions['task_losses'].append(losses)
        print(json.dumps(dict(event='EVIDENCE_FAMILY_READ', family_index=i, model_fits=0)), flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-', dir=destination.parent))
    try:
        np.savez_compressed(temp/'meta_dataset.npz', **{k: np.asarray(v) for k,v in data.items()})
        np.savez_compressed(temp/'classifier_predictions.npz', **{k: np.asarray(v) for k,v in predictions.items()})
        for name in ('summary.json', 'run_metrics.json', 'config.json', 'environment.json',
                     'job.json', 'registration.json', 'protocol_lock.json'):
            shutil.copyfile(folder/name, temp/name)
        checkpoints = []
        for method, source, receipt in checkpoint_sources:
            path = 'checkpoints/'+source.name
            (temp/path).parent.mkdir(exist_ok=True)
            shutil.copyfile(source, temp/path)
            checkpoints.append(dict(method=method, task_index=0, path=path, **receipt))
        files = {str(p.relative_to(temp)): sha(p) for p in temp.rglob('*') if p.is_file()}
        manifest = dict(schema_version=1, source_run_id=job['run_id'],
            source_protocol_id=summary['protocol_id'], track=job['track'], methods=names,
            status=job['status'], paper_equivalence=c['paper_equivalence'],
            families=len(stats), tasks_per_family=job['tasks'], task_count=len(data['task_id']),
            observations=len(data['task_id'])*c['dataset']['observations'],
            classifier_rows=len(predictions['label']), model_fits_during_export=0,
            objective_calls_during_export=0, files=files, checkpoints=checkpoints,
            checkpoint_scope='first task of first family only',
            checkpoint_status='SAVED_DURING_FIT' if checkpoints else 'NOT_SAVED_DURING_ORIGINAL_RUN',
            missing_field_policy='posterior_mean/latent_variance are NaN only for RF rows (not applicable)',
            verification_scope='Complete raw observation arrays, splits, validation predictions and task losses; source snapshot and all model checkpoints are not bundled. Original protocol lock is provenance metadata, not a full source archive.')
        save(temp/'evidence_manifest.json', manifest)
        verify_evidence(temp)
        require(not destination.exists(), 'Evidence destination appeared during export')
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return dict(status='PASS', destination=str(destination), **{
        k: manifest[k] for k in ('task_count', 'observations', 'classifier_rows', 'checkpoint_status')})


def verify_evidence(folder):
    """Check saved arrays and recompute BCE/CVaR; no model imports or writes."""
    import numpy as np
    from .probability import gaussian, rf_scores
    p = Path(folder)
    m = read(p/'evidence_manifest.json')
    require(m['schema_version'] == 1, 'Unsupported evidence schema')
    for name, digest in m['files'].items():
        require(sha(safe_child(p, name)) == digest, 'Evidence checksum changed: '+name)
    c, job, summary = read(p/'config.json'), read(p/'job.json'), read(p/'summary.json')
    require(m['methods'] == methods(c, job['track']) and m['track'] == job['track'], 'Evidence methods/track')
    require(m['source_run_id'] == summary['run_id'] == job['run_id'], 'Evidence source run')
    require(m['source_protocol_id'] == summary['protocol_id'], 'Evidence protocol identity')
    require(m['status'] == job['status'] == summary['status'] and
            m['paper_equivalence'] == summary['paper_equivalence'] == c['paper_equivalence'], 'Evidence status')
    require(m['families'] == len(job['seeds']) == summary['families'] and
            m['tasks_per_family'] == job['tasks'] == summary['tasks_per_family'], 'Evidence population')
    n, nmethods = m['task_count'], len(m['methods'])
    require(nmethods == 2 and n == len(job['seeds'])*job['tasks'], 'Evidence task count')
    with np.load(p/'meta_dataset.npz', allow_pickle=False) as stored_d, np.load(p/'classifier_predictions.npz', allow_pickle=False) as stored_a:
        # NpzFile is lazy and does not cache decompressed arrays. Materialize
        # once before per-task/per-row checks, including on full100 datasets.
        d = {key: stored_d[key] for key in stored_d.files}
        a = {key: stored_a[key] for key in stored_a.files}
        observations = c['dataset']['observations']
        require(d['X'].shape == (n, observations, 1) and d['y'].shape == (n, observations), 'Raw array shape')
        require(m['observations'] == n*observations, 'Raw observation count')
        require(np.isfinite(d['X']).all() and np.isfinite(d['y']).all(), 'Nonfinite raw arrays')
        require(((d['X'] >= 0) & (d['X'] <= 1)).all(), 'Raw domain')
        offsets = a['offsets']
        require(offsets.shape == (n*nmethods+1,) and offsets[0] == 0 and
                np.all(np.diff(offsets) > 0) and offsets[-1] == len(a['label']) == m['classifier_rows'], 'Prediction offsets')
        require(a['task_losses'].shape == (n, nmethods), 'Task loss shape')
        for key in ('point_index', 'label', 'weight', 'probability', 'log_p', 'log_not_p', 'logit', 'BCE', 'weighted_BCE', 'posterior_mean', 'latent_variance'):
            require(a[key].shape == a['label'].shape, 'Prediction field shape: '+key)
        for t in range(n):
            fi = t//job['tasks']
            require(d['family_index'][t] == fi and d['seed'][t] == job['seeds'][fi] and
                    d['task_id'][t] == f'meta{t%job["tasks"]+1:04d}', 'Task order/seed/identity')
            tr, va = d['train_indices'][t], d['validation_indices'][t]
            require(len(tr) == c['dataset']['split']['train_count'] and len(va) == c['dataset']['split']['validation_count'] and
                    sorted([*tr, *va]) == list(range(observations)), 'Raw partition')
            require(not set(d['X'][t, tr, 0]) & set(d['X'][t, va, 0]), 'Coordinate leakage')
            y = d['y'][t]
            tau = float(np.quantile(np.unique(y[tr]), c['classification']['gamma']))
            U = float(np.maximum(tau-y[tr], 0).mean())
            balance = (len(tr)+int((y[tr] < tau).sum()))/(2*len(tr))
            require(U > 0, 'Degenerate training utility')
            close(d['tau'][t], tau, 'Training tau'); close(d['U_train'][t], U, 'Training utility')
            close(d['balance'][t], balance, 'Training balance')
            good = [int(i) for i in va if y[i] < tau] if job['track'] == 'B' else []
            expected_indices = good+va.tolist()
            labels = [1]*len(good)+([0]*len(va) if job['track']=='B' else (y[va]<tau).astype(int).tolist())
            weights = [balance*(tau-y[i])/U for i in good]+([balance]*len(va) if job['track']=='B' else [1.]*len(va))
            for mi in range(nmethods):
                unit = t*nmethods+mi
                sl = slice(int(offsets[unit]), int(offsets[unit+1]))
                np.testing.assert_array_equal(a['point_index'][sl], expected_indices)
                np.testing.assert_array_equal(a['label'][sl], labels)
                np.testing.assert_allclose(a['weight'][sl], weights, rtol=2e-11, atol=1e-12)
                for row in range(sl.start, sl.stop):
                    if mi == 0:
                        require(np.isnan(a['posterior_mean'][row]) and np.isnan(a['latent_variance'][row]), 'RF-only missing fields')
                        score = rf_scores(float(a['probability'][row]), c['classification']['RF_epsilon'])
                    else:
                        score = gaussian(float(a['posterior_mean'][row]), float(a['latent_variance'][row]), tau, U, c)
                        prefix = 'event' if job['track']=='A' else 'augmented'
                        score = dict(log_p=score[prefix+'_log_p'], log_not_p=score[prefix+'_log_not_p'], probability=score[prefix+'_probability'])
                    for key in ('probability', 'log_p', 'log_not_p'):
                        close(a[key][row], score[key], 'Saved probability/log semantics: '+key)
                lp, lm = a['log_p'][sl], a['log_not_p'][sl]
                bce = -np.where(a['label'][sl] == 1, lp, lm)
                np.testing.assert_allclose(a['logit'][sl], lp-lm, rtol=2e-11, atol=1e-12)
                np.testing.assert_allclose(a['BCE'][sl], bce, rtol=2e-11, atol=1e-12)
                np.testing.assert_allclose(a['weighted_BCE'][sl], bce*a['weight'][sl], rtol=2e-11, atol=1e-12)
                close(a['task_losses'][t,mi], float((bce*a['weight'][sl]).mean()), 'Task row-count BCE')
        stats = []
        for i, seed in enumerate(job['seeds']):
            block = a['task_losses'][i*job['tasks']:(i+1)*job['tasks']]
            stats.append(dict(family_index=i, seed=seed, methods={method: task_summary(block[:,mi].tolist(), c['classification']['alpha']) for mi,method in enumerate(m['methods'])}))
        recorded = read(p/'run_metrics.json')
        for actual, expected in zip(stats, recorded):
            require(actual['family_index'] == expected['family_index'] and actual['seed'] == expected['seed'], 'Family identity')
            for method in m['methods']:
                for key, value in actual['methods'][method].items():
                    close(value, expected['methods'][method][key], 'Family metric: '+key)
        require(len(recorded) == len(stats), 'Family count')
        for method in m['methods']:
            for metric in ('mean_BCE', 'CVaR'):
                actual = aggregate([r['methods'][method][metric] for r in stats])
                for key, value in actual.items():
                    expected = summary['methods'][method][metric][key]
                    require(expected is None, 'Singleton SE') if value is None else close(value, expected, 'Summary '+metric+'/'+key)
    for checkpoint in m['checkpoints']:
        require(checkpoint['task_index'] == 0 and checkpoint['method'] in m['methods'], 'Checkpoint scope')
        require(checkpoint['path'] in m['files'] and checkpoint['sha256'] == m['files'][checkpoint['path']], 'Checkpoint receipt')
    require(m['checkpoint_status'] == ('SAVED_DURING_FIT' if m['checkpoints'] else 'NOT_SAVED_DURING_ORIGINAL_RUN'), 'Checkpoint status')
    return dict(status='PASS', tasks=n, classifier_rows=m['classifier_rows'],
                checked='raw splits/weights/probabilities/logits/Weighted BCE/task CVaR/family SE/checksums',
                checkpoint_count=len(m['checkpoints']), model_fits=0, objective_calls=0)


def replay_checkpoints(folder):
    """Explicit inference-only replay of native estimators saved by this pipeline."""
    import numpy as np
    import joblib
    p = Path(folder)
    m = read(p/'evidence_manifest.json')
    require(bool(m['checkpoints']), 'No fitted model checkpoints were saved during this original run; no refit will be attempted')
    verify_evidence(p)
    from .protocol import environment
    c = read(p/'config.json'); environment(c)
    bounds = np.asarray(c['dataset']['domain'])
    with np.load(p/'meta_dataset.npz', allow_pickle=False) as stored_d, np.load(p/'classifier_predictions.npz', allow_pickle=False) as stored_a:
        d = {key: stored_d[key] for key in stored_d.files}
        a = {key: stored_a[key] for key in stored_a.files}
        for checkpoint in m['checkpoints']:
            t = checkpoint['task_index']; mi = m['methods'].index(checkpoint['method'])
            unit = t*len(m['methods'])+mi
            sl = slice(int(a['offsets'][unit]), int(a['offsets'][unit+1]))
            X = d['X'][t, a['point_index'][sl]]
            estimator = joblib.load(safe_child(p, checkpoint['path']))
            if checkpoint['kind'] == 'random_forest':
                column = np.flatnonzero(estimator.classes_ == 1)
                require(len(column) == 1, 'Checkpoint positive class')
                prediction = estimator.predict_proba(X)[:, column[0]]
                np.testing.assert_allclose(prediction, a['probability'][sl], rtol=2e-11, atol=1e-12)
            else:
                require(checkpoint['kind'] == 'gaussian_process', 'Checkpoint estimator kind')
                X = (X-bounds[:,0])/(bounds[:,1]-bounds[:,0])
                mean, std = estimator.predict(X, return_std=True)
                np.testing.assert_allclose(mean, a['posterior_mean'][sl], rtol=2e-11, atol=1e-12)
                np.testing.assert_allclose(std**2, a['latent_variance'][sl], rtol=2e-11, atol=1e-12)
    return dict(status='PASS', checkpoints_replayed=len(m['checkpoints']),
                operation='inference_only', model_fits=0, objective_calls=0)
