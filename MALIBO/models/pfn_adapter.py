"""Adapter for PFNs4BO's published pretrained TransformerBO interface."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


class PFNFactory:
    def __init__(self, model_name: str = "hebo_plus", device: str = "cpu:0"):
        if model_name not in {"hebo_plus", "bnn"}:
            raise ValueError("model_name must be 'hebo_plus' or 'bnn'.")
        try:
            import torch
            import pfns4bo
            import typing
            import torch.nn.modules.transformer

            # Sửa lỗi PyTorch 2.x bỏ export Optional trong module transformer
            if not hasattr(torch.nn.modules.transformer, "Optional"):
                torch.nn.modules.transformer.Optional = typing.Optional
            from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
        except ImportError as exc:
            raise ImportError(
                "PFN baseline requires PFNs4BO. Install with `pip install pfns4bo`."
            ) from exc

        model_attr = "hebo_plus_model" if model_name == "hebo_plus" else "bnn_model"
        model_path = getattr(pfns4bo, model_attr, None)
        if model_path is None:
            raise RuntimeError(
                f"Installed pfns4bo package does not expose {model_attr}. "
                "Check package version and its final_models assets."
            )

        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:  # Backward compatibility with older PyTorch versions.
            loaded = torch.load(model_path, map_location="cpu")

        # Checkpoint của PFNs4BO (đặc biệt là hebo_plus) lưu tuple dạng (model, style)
        if isinstance(loaded, (tuple, list)):
            model = loaded[0]
            style = loaded[1] if len(loaded) > 1 else None
        else:
            model = loaded
            style = None

        self._torch = torch
        self._transformer_bo_method = TransformerBOMethod
        self._model = model
        self._style = style
        self.model_name = model_name
        self.device = device

    def new_task_optimizer(self):
        # PFNs are pretrained; target adaptation is performed in-context from
        # the supplied observation set and does not meta-train on this task.
        kwargs: dict[str, Any] = {"device": self.device}
        if self._style is not None:
            kwargs["style"] = copy.deepcopy(self._style)

        return self._transformer_bo_method(
            copy.deepcopy(self._model),
            **kwargs,
        )

    @staticmethod
    def suggest(optimizer, X_obs, y_obs, X_pending) -> int:
        X_obs = np.asarray(X_obs, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64).reshape(-1)
        X_pending = np.asarray(X_pending, dtype=np.float64)
        if X_obs.ndim != 2 or X_pending.ndim != 2:
            raise ValueError("X_obs and X_pending must be 2D arrays.")
        if np.any(X_obs < 0.0) or np.any(X_obs > 1.0):
            raise ValueError("PFNs4BO expects X_obs normalized to [0, 1].")
        if np.any(X_pending < 0.0) or np.any(X_pending > 1.0):
            raise ValueError("PFNs4BO expects X_pending normalized to [0, 1].")
        result = optimizer.observe_and_suggest(
            X_obs,
            y_obs,
            X_pending,
            return_actual_ei=False,
        )
        return int(result)