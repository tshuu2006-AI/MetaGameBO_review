"""Adapter for the official BoschResearch MALIBO implementation."""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


class MALIBOFactory:
    def __init__(self, search_space: Any, config: Any, repo_path: str | Path):
        repo_root = Path(repo_path).expanduser().resolve()
        if not (repo_root / "malibo.py").is_file():
            raise FileNotFoundError(
                f"{repo_root} is not the official MALIBO repository root. "
                "Expected malibo.py, meta_learning/ and inference/. Clone the "
                "BoschResearch repository and pass its root via --malibo-repo."
            )
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        try:
            module = importlib.import_module("malibo")
            self._malibo_class = module.MALIBO
        except Exception as exc:

            raise RuntimeError(
                "Could not import official MALIBO. Check its pinned dependencies "
                "and make sure --malibo-repo points to the upstream repository."
            ) from exc

        self.search_space = search_space
        self.config = config
        self._meta_model_state = None

    def _new_optimizer(self, seed: int):
        import torch

        torch.manual_seed(int(seed))
        return self._malibo_class(
            self.search_space,
            seed=int(seed),
            gamma=self.config.malibo_gamma,
            num_samples_acquisition_function=self.config.candidate_count,
            num_layers=self.config.malibo_num_layers,
            num_features=self.config.malibo_num_features,
            num_hidden_units=self.config.malibo_num_hidden_units,
            device=self.config.malibo_device,
        )

    def meta_train(self, meta_data: dict[int, list[Any]]) -> None:
        """Meta-train once and retain the fully prepared base optimizer."""
        # Khởi tạo và chạy meta_fit trên tập task mẫu
        base_optimizer = self._new_optimizer(seed=self.config.meta_seed)
        base_optimizer.meta_fit(
            meta_data,
            num_epochs=self.config.meta_epochs,
            batch_size=self.config.meta_batch_size,
        )
        # Lưu lại toàn bộ instance đã hoàn tất chuyển đổi kiến trúc
        self._base_optimizer = base_optimizer

    def new_task_optimizer(self, seed: int):
        """Create a clean target-task optimizer with shared meta-weights."""
        if self._base_optimizer is None:
            raise RuntimeError("Call meta_train() before new_task_optimizer().")
        
        # Sao chép nguyên bản optimizer đã meta-train cho target run mới
        optimizer = copy.deepcopy(self._base_optimizer)
        
        # Cập nhật seed riêng cho từng run nếu có
        if hasattr(optimizer, "seed"):
            optimizer.seed = seed
        return optimizer

    
    @staticmethod
    def suggest(optimizer, X_obs, y_obs, X_pending, seed: int) -> int:
        X_obs = np.asarray(X_obs, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64).reshape(-1)
        X_pending = np.asarray(X_pending, dtype=np.float64)
        if X_pending.ndim != 2 or X_pending.shape[1] != X_obs.shape[1]:
            raise ValueError("X_pending must have shape (n_candidates, input_dim).")
        index = optimizer.observe_and_suggest(
            X_obs,
            y_obs,
            X_pen=X_pending,
            sampling="thompson_sampling",
            seed=int(seed),
        )
        return int(index)
