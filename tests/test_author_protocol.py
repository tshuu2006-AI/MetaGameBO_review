import copy
import numpy as np
import torch
from benchmarks import ForresterBenchmark
from benchmarks.base import BenchmarkTask
from experiment import build_parser, run_bo
from models import MALIBOAgent
from models.author_metrics import evaluate_classifier


def test_defaults_and_exact_objective_budget():
    args = build_parser().parse_args([])
    assert (args.tasks,args.samples_per_task,args.epochs,args.task_batch_size,args.seed,args.steps,args.runs)==(200,32,100,10,44,16,1)
    calls=[]
    def objective(x):
        calls.extend(x.tolist())
        return x[:,0]**2
    class Probe:
        def suggest(self,x,y,pool): return pool[0]
    curve=run_bo(Probe(),BenchmarkTask(0,objective,f_min=0,f_max=1),16,2,20,44)
    assert len(calls)==len(curve)==16
    assert (np.diff(curve)<=0).all()


def test_real_author_implementation_and_deterministic_audit(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.chdir(tmp_path)
    tasks=ForresterBenchmark(3,44).get_meta_train_tasks(3,8)
    agent=MALIBOAgent(np.array([[0.,1.]]),meta_epochs=2)
    agent.meta_train(tasks)
    assert type(agent.model).__module__.startswith('vendor.metagamebo_author')
    assert len(agent.training_diagnostics)==2
    agent.model.train()
    rng_before = torch.random.get_rng_state().clone()
    a=evaluate_classifier(agent.model,agent.model.evaluation_data['validation'],agent.model.metric_loss_fn)
    b=evaluate_classifier(agent.model,agent.model.evaluation_data['validation'],agent.model.metric_loss_fn)
    assert a==b and agent.model.training
    torch.testing.assert_close(torch.random.get_rng_state(),rng_before,rtol=0,atol=0)
    assert a['objective_with_regularization']==a['micro_bce']+a['regularization']
    before=copy.deepcopy(agent.model.state_dict())
    for n in [2, 8]:
        selected=agent.suggest(tasks[0].X[:n],tasks[0].y[:n],np.linspace(0,1,16)[:,None])
        assert selected.shape==(1,)
    for key,value in before.items():
        torch.testing.assert_close(agent.model.state_dict()[key],value,rtol=0,atol=0)
