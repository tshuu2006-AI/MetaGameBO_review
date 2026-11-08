"""Evaluator-only original scalar Forrester objective; never passed to optimizers."""
from __future__ import annotations
import ast
import functools
import hashlib
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'configs/forrester_benchmark/author_snapshot/forrester.py'
AUTHOR_SHA='d9c01f7582534f85026df82f1bd6106300e758a03b99f95d565d3b16fc20194d'

@functools.lru_cache(maxsize=1)
def original_scalar():
    raw=SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=AUTHOR_SHA:raise ValueError('Untrusted evaluator objective source')
    tree=ast.parse(raw.decode());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forrester_function')
    namespace={'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace['forrester_function']

class AuthorTarget:
    """Evaluator-only wrapper of the original scalar function AST, unchanged.

    The original math.pow/math.sin semantics avoid cross-NumPy last-bit drift
    in the mandatory exact shared initialization. No reference/model is here.
    """
    def __init__(self,spec):
        self.task_id=spec['task_id'];self._parameters={k:float(spec[k]) for k in ('a','b','c')};self.evaluations=0
        a,b,c=self._parameters.values()
        if not (.2<=a<=3 and -5<=b<=15 and -5<=c<=5):raise ValueError('Task outside author ranges')
    def evaluate(self,X):
        import numpy as np
        from models.base_model import validate_points
        points=validate_points(X,np.array([[0.,1.]]));function=original_scalar()
        values=np.array([function(float(x[0]),**self._parameters) for x in points],dtype=float)
        self.evaluations+=len(values)
        return values
    def evaluate_scalar(self,x):return float(self.evaluate([[float(x)]])[0])
