"""Independently verify saved checkpoints, BCE aggregates and objective traces."""
import argparse
import copy
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import sklearn
from benchmarks import ForresterBenchmark
from metrics import summarize_weighted_bce, calculate_normalized_regret
from models import MALIBOAgent, PFNAgent
from experiment import run_bo
from models.numerics import seed_everything


def verify(output):
    output = Path(output)
    config = json.loads((output/'config.json').read_text())
    torch.set_num_threads(1)
    benchmark = ForresterBenchmark(config['tasks'], config['seed'])
    tasks = benchmark.get_meta_train_tasks(config['tasks'], config['samples_per_task'])
    dataset_path = output/'meta_dataset.npz'
    arrays = {'X': np.stack([t.X for t in tasks]), 'y': np.stack([t.y for t in tasks])}
    if dataset_path.exists():
        with np.load(dataset_path) as stored:
            for key, array in arrays.items():
                np.testing.assert_array_equal(stored[key], array)
    else:
        # Runs begun before dataset export was added use the same deterministic generator.
        np.savez_compressed(dataset_path, **arrays)
    checked = []
    replay_agents = {}
    if 'MALIBO' in config['methods']:
        agent = MALIBOAgent(benchmark.bounds, seed=config['seed'])
        X, ids, labels, weights = [], [], [], []
        for i, task in enumerate(tasks, 1):
            x, z, w = agent.optimizer._reweight_samples(task.X, task.y[:,None], agent.optimizer.gamma)
            X.extend(x); labels.extend(z); weights.extend(w); ids.extend([i]*len(x))
        tensors = (torch.tensor(np.asarray(X), dtype=torch.float64), torch.tensor(ids),
                   torch.tensor(np.asarray(labels), dtype=torch.float64),
                   torch.tensor(np.asarray(weights), dtype=torch.float64).reshape(-1,1))
        dataset = torch.utils.data.TensorDataset(*tensors)
        seed_everything(config['seed'])
        train, validation = torch.utils.data.random_split(dataset, [.8, 1-.8])
        model = agent.model
        model.initialize(config['tasks'], device='cpu', dtype=torch.float64)
        model.load_state_dict(torch.load(output/'MALIBO_checkpoint.pt', map_location='cpu', weights_only=True))
        model.eval()
        agent._trained = True
        replay_agents['MALIBO'] = agent
        saved = list(csv.DictReader((output/'classifier_bce.csv').open(encoding='utf-8')))
        prediction_arrays = {}
        for split, subset in [('train',train),('validation',validation)]:
            x, task_ids, z, w = next(iter(torch.utils.data.DataLoader(subset,batch_size=len(subset))))
            with torch.no_grad():
                _, logits = model(x,task_ids)
            metrics = summarize_weighted_bce(logits,z,w,task_ids,alpha=.8)
            row = next(r for r in saved if r['method']=='MALIBO' and r['split']==split)
            for key in ['macro_bce','micro_bce','cvar_bce','tail_k','number_of_tasks','sample_count']:
                np.testing.assert_allclose(metrics[key],float(row[key]),rtol=1e-10,atol=1e-10)
            for key, value in [('logits',logits),('labels',z),('weights',w),('task_ids',task_ids)]:
                prediction_arrays[f'{split}_{key}'] = value.detach().numpy()
        np.savez_compressed(output/'classifier_predictions.npz', **prediction_arrays)
        assert all(r['method']=='MALIBO' for r in saved), 'PFN must not appear in BCE table'
        history = json.loads((output/'MALIBO_training.json').read_text())
        assert len(history)==config['epochs']
        checked.append('MALIBO checkpoint independently reproduces train/validation macro/micro/CVaR BCE')
    if 'PFN' in config['methods']:
        history = json.loads((output/'PFN_training.json').read_text())
        assert len(history)==config['epochs']
        assert all(r['tasks_seen']==config['tasks'] for r in history)
        assert len(set(c for r in history for c in r['context_lengths']))>1
        assert all(np.isfinite(r['train_gaussian_nll']) for r in history)
        agent = PFNAgent(benchmark.bounds, seed=config['seed'])
        agent.network.load_state_dict(torch.load(output/'PFN_checkpoint.pt',map_location='cpu',weights_only=True))
        agent._trained = True
        agent.network.eval()
        replay_agents['PFN'] = agent
        checked.append('PFN uses every task per epoch, variable contexts and separate finite NLL diagnostics')
    traces = json.loads((output/'bo_traces.json').read_text())
    curves = np.load(output/'curves.npz')
    for trace in traces:
        X, y = np.array(trace['X']), np.array(trace['y'])
        assert trace['objective_evaluations']==len(y)==config['steps']
        target = benchmark.get_target_task(evaluation_seed=trace['run'])
        np.testing.assert_array_equal(y, target.objective(X))
        curve = calculate_normalized_regret(np.minimum.accumulate(y), trace['f_min'], trace['f_max'])
        np.testing.assert_array_equal(curves[trace['method']][trace['run']], curve)
        assert ((curve>=-1e-6)&(curve<=1+1e-6)).all() and (np.diff(curve)<=0).all()
        if trace['method'] in replay_agents:
            replay = copy.deepcopy(replay_agents[trace['method']])
            replay_curve = run_bo(replay,target,config['steps'],config['initial_points'],
                                  config['candidates'],config['seed']+trace['run'])
            np.testing.assert_array_equal(replay.bo_trace['X'],X)
            np.testing.assert_array_equal(replay_curve,curve)
    checked.append('Every stored objective value and regret curve matches the target function and total budget')
    checked.append('Saved MALIBO/PFN checkpoints replay exactly the stored BO proposals and regret curves')
    provenance = json.loads((ROOT/'vendor/metagamebo_author/PROVENANCE.json').read_text())
    for name, entry in provenance['files'].items():
        assert hashlib.sha256((ROOT/'vendor/metagamebo_author'/name).read_bytes()).hexdigest()==entry['vendored_sha256']
    checked.append('Vendored source hashes match the pinned provenance manifest')
    report = {'passed': True, 'checks': checked, 'python': platform.python_version(),
              'numpy': np.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__,
              'author_commit': provenance['commit'], 'configuration': config}
    code_paths = ['experiment.py','metrics.py','models/malibo.py','models/pfn.py',
                  'models/author_metrics.py','scripts/verify_results.py']
    report['code_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_paths}
    report['artifact_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for pattern in ['*_checkpoint.pt','meta_dataset.npz','classifier_predictions.npz','curves.npz']
                                for p in output.glob(pattern)}
    (output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',nargs='?',default='results/author_reply')
    verify(parser.parse_args().output)
