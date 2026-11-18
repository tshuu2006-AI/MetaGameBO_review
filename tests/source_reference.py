"""Independent original-source task-loss/CVaR arithmetic reference."""
import ast
import numpy as np
import torch
from experiments.lfbo_gp_unified.source_ei import ROOT,source_check

def original_validation_metric():
    source_check()
    path = ROOT / 'vendor/metagamebo/meta_learning/trainer.py'
    tree = ast.parse(path.read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'task_id']
    loop = max(loops, key=lambda n: n.lineno)  # Original validation task loop, not training loop.
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('num_val_tasks', 'val_cvar'):
                    assignments[target.id] = node
    wrapper = ast.parse('def reference(task_ids, targets, predictions, weights, alpha):\n validation_losses=[]\n unique_task_ids=torch.unique(task_ids).cpu().numpy()\n').body[0]
    wrapper.body += [loop, assignments['num_val_tasks'], assignments['val_cvar'],
                     ast.parse('return validation_losses, num_val_tasks, val_cvar').body[0]]
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['reference']
