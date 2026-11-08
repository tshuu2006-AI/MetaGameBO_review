"""Lazy official PFNs4BO backend; only a pinned, verified HEBO+ pickle is accepted.

Reference: automl/PFNs4BO @ 4c0d901e4e0f1d5afd4b12e33974525cf3493dd3.
No torch import, checkpoint load or network access happens at module import.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from time import perf_counter
import weakref

import numpy as np

REPOSITORY_COMMIT = "4c0d901e4e0f1d5afd4b12e33974525cf3493dd3"
CHECKPOINT_FILENAME = "model_hebo_morebudget_9_unused_features_3.pt"
CHECKPOINT_SHA256 = "cc7f6f4b9b06e59987e42845b7b0d31ffa5b414b9eddfe14d88b25120e3cd4f8"
COMPRESSED_SHA256 = "ad7af1d807db9eaffb152dc89eaae9262c10e2bb99b81bb12ae5cf8f80aba432"
COMPRESSED_BLOB_SHA1 = "f4f25d121742f88e72f478f40f0a6d5d184c8be5"
SOURCE_URI = (f"https://raw.githubusercontent.com/automl/PFNs4BO/{REPOSITORY_COMMIT}/"
              f"pfns4bo/final_models/{CHECKPOINT_FILENAME}.gz")
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / ".cache/pfns4bo" / CHECKPOINT_FILENAME
ACQUISITION_BLOB_SHA1 = "9fae1eba548a15a18e006a209da4f9e8fb2ffadb"


def checkpoint_metadata(checkpoint=None, model_name="hebo_plus"):
    """Verify a known official payload before any pickle load; never download implicitly."""
    if model_name != "hebo_plus":
        raise ValueError("Only the audited hebo_plus checkpoint is supported; BNN is an unrun ablation")
    path = DEFAULT_CHECKPOINT if checkpoint is None else Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PFN checkpoint missing: {path}; run python -m scripts.fetch_pfn_checkpoint explicitly")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != CHECKPOINT_SHA256 or path.stat().st_size != 107262245:
        raise ValueError("PFN checkpoint SHA256/size mismatch; refusing to unpickle an untrusted payload")
    return {"model_name": model_name, "path": str(path), "sha256": actual, "size_bytes": path.stat().st_size,
            "source_uri": SOURCE_URI, "repo_commit": REPOSITORY_COMMIT,
            "released_filename": CHECKPOINT_FILENAME + ".gz", "compressed_sha256": COMPRESSED_SHA256,
            "compressed_git_blob_sha1": COMPRESSED_BLOB_SHA1}


def standardize_observed_y(y):
    """Population standardization using observed values only; constant/one-point scale=1."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("PFN y context must be a nonempty finite vector")
    with np.errstate(over="ignore", invalid="ignore"):
        center, spread = float(values.mean()), float(values.std(ddof=0))
        scale = spread if spread > 1e-8 else 1.0
        result = (values - center) / scale
    if not np.isfinite(result).all() or not np.isfinite([center, scale]).all():
        raise ValueError("PFN observed-y standardization overflowed; observations will not be altered")
    return result, {"center": center, "scale": scale, "ddof": 0, "constant_scale": spread <= 1e-8}


class PFNs4BOBackend:
    """Read-only inference via official TransformerBOMethod, with strict finite-logit guards."""

    def __init__(self, *, checkpoint=None, model_name="hebo_plus", device="cpu", acquisition="ei",
                 candidate_batch_size=64, cpu_threads=2):
        if device != "cpu" or acquisition != "ei":
            raise ValueError("The audited PFN backend currently supports CPU and EI only")
        self.checkpoint = checkpoint_metadata(checkpoint, model_name)
        try:
            import torch
            import pfns4bo
            from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
        except ImportError as exc:
            raise RuntimeError("PFNs4BO dependencies unavailable; use the separate .venv-pfn/requirements-pfn.txt") from exc
        if importlib.metadata.version("pfns4bo") != "0.1.5":
            raise RuntimeError("PFN backend requires the audited pfns4bo==0.1.5 API")
        source = Path(pfns4bo.__file__).parent / "scripts/acquisition_functions.py"
        raw = source.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != ACQUISITION_BLOB_SHA1:
            raise RuntimeError("Installed PFNs4BO acquisition source differs from the audited official commit")
        self.torch = torch
        # Threads are a process setting, recorded explicitly; all measured PFN runs use 2.
        torch.set_num_threads(cpu_threads)
        start = perf_counter()
        # Serialized model is required by the official release. The known payload
        # hash above is checked BEFORE enabling full pickle deserialization.
        self.model = torch.load(self.checkpoint["path"], map_location="cpu", weights_only=False)
        self.model.eval()
        self.model.requires_grad_(False)
        self.load_seconds = perf_counter() - start
        self.forward_calls = 0
        # A bound-method hook would create model -> hook -> backend -> model,
        # retaining a full checkpoint between task runs until cyclic GC fires.
        # Keep the guard without retaining the backend/model through that cycle.
        backend_ref = weakref.ref(self)
        def finite_hook(module, inputs, output):
            backend = backend_ref()
            if backend is None:
                raise RuntimeError("PFN backend was released before inference")
            backend._finite_output(module, inputs, output)
        self._guard_handle = self.model.register_forward_hook(finite_hook)
        self.method = TransformerBOMethod(self.model, device="cpu", acq_function="ei",
                                         apply_power_transform=False, znormalize=False,
                                         max_dataset_size=candidate_batch_size)
        self.metadata = {"backend": "official_pfns4bo_TransformerBOMethod", "checkpoint": self.checkpoint,
                         "torch": torch.__version__, "pfns4bo": importlib.metadata.version("pfns4bo"),
                         "acquisition_blob_sha1": blob, "device": "cpu", "cpu_threads": torch.get_num_threads(),
                         "model_class": type(self.model).__module__ + "." + type(self.model).__name__,
                         "parameter_count": sum(p.numel() for p in self.model.parameters()),
                         "pretrained_weights_frozen": True, "model_eval": True,
                         "preprocessing": "observed_y_population_standardization", "apply_power_transform": False,
                         "score_units": "EI_in_standardized_negative_loss_units", "load_seconds": self.load_seconds}

    def _finite_output(self, module, inputs, output):
        if not isinstance(output, self.torch.Tensor) or not self.torch.isfinite(output).all():
            raise RuntimeError("PFN produced nonfinite logits; refusing upstream silent zero-score fallback")
        self.forward_calls += 1

    def reset_context(self):
        """No target tensors are retained by the official method; clear diagnostic state."""
        self.last_transform = None
        self.forward_calls = 0

    def score(self, X_obs, y_maximize, X_candidates, *, seed):
        torch = self.torch
        x = np.asarray(X_obs, dtype=np.float64)
        candidates = np.asarray(X_candidates, dtype=np.float64)
        y, transform = standardize_observed_y(y_maximize)
        if (x.ndim != 2 or candidates.ndim != 2 or x.shape[1] != candidates.shape[1]
                or len(x) != len(y) or len(candidates) == 0
                or not np.isfinite(x).all() or not np.isfinite(candidates).all()
                or np.any(x < 0) or np.any(x > 1) or np.any(candidates < 0) or np.any(candidates > 1)):
            raise ValueError("PFN features must be finite matching (n,d)/(m,d) in [0,1]")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("PFN must remain in eval mode with frozen pretrained weights")
        before = self.forward_calls
        # The official wrapper draws a random tie index. Isolate that draw from
        # process RNG and ignore its index; the adapter chooses a stable argmax.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(int(seed))
            _, scores = self.method.observe_and_suggest(x.copy(), y.copy(), candidates.copy(), return_actual_ei=True)
        if (scores.dtype == torch.bool or tuple(scores.shape) != (len(candidates),)
                or not torch.isfinite(scores).all() or torch.any(scores < 0)):
            raise RuntimeError("Official PFN EI scores must be a finite nonnegative numeric vector")
        if self.forward_calls == before:
            raise RuntimeError("PFN acquisition returned without a real model forward pass")
        self.last_transform = transform
        return scores.detach().cpu().numpy().astype(np.float64, copy=True)
