import copy
import numpy as np
import pytest
import torch
from benchmarks import ForresterBenchmark
from models.pfn import InContextSurrogate, PFNAgent, expected_improvement


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_query_predictions_do_not_depend_on_other_queries():
    torch.manual_seed(44)
    model = InContextSurrogate(1).eval()
    x, y, q = torch.rand(1,4,1), torch.randn(1,4), torch.rand(1,5,1)
    with torch.no_grad():
        full = model(x,y,q)
        single = model(x,y,q[:,:1])
        changed = q.clone(); changed[:,1:] += 100
        altered = model(x,y,changed)
    torch.testing.assert_close(full.mean[:,:1], single.mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(full.stddev[:,:1], single.stddev, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(full.mean[:,:1], altered.mean[:,:1], atol=1e-6, rtol=1e-5)
    mask = model.attention_mask(4,5)
    assert not mask[:,:4].any() and mask[:,4:].all()


def test_predictive_density_and_analytic_ei():
    mean = torch.tensor([0., 0., -2.], dtype=torch.float64)
    std = torch.tensor([1., 0., 0.], dtype=torch.float64)
    np.testing.assert_allclose(expected_improvement(mean,std,0.).numpy(), [1/np.sqrt(2*np.pi), 0., 2.])
    torch.manual_seed(44)
    samples = torch.randn(300000, dtype=torch.float64)
    assert expected_improvement(mean[:1],std[:1],0.).item() == pytest.approx((-samples).clamp_min(0).mean().item(), abs=.004)


def test_variable_context_full_epochs_and_frozen_inference():
    tasks = ForresterBenchmark(10,44).get_meta_train_tasks(10,12)
    agent = PFNAgent(np.array([[0.,1.]]),meta_epochs=3,task_batch_size=2)
    agent.meta_train(tasks)
    contexts = set(c for row in agent.training_diagnostics for c in row['context_lengths'])
    assert len(contexts)>1
    assert all(r['tasks_seen']==10 and r['optimizer_steps']==5 for r in agent.training_diagnostics)
    before = copy.deepcopy(agent.network.state_dict())
    x, y = tasks[0].X[:2], tasks[0].y[:2]
    pool = np.linspace(0,1,20)[:,None]
    d = agent.predict_distribution(x,y,pool)
    assert torch.isfinite(d.log_prob(d.mean)).all() and (d.stddev>0).all()
    selected = agent.suggest(x,y,pool)
    np.testing.assert_array_equal(selected,pool[int(expected_improvement(d.mean,d.stddev,float(y.min())).argmax())])
    for key, value in before.items():
        torch.testing.assert_close(agent.network.state_dict()[key],value,rtol=0,atol=0)
