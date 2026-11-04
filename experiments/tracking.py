"""Lazy, optional W&B tracking. Disabled mode performs no SDK import or network call."""
from __future__ import annotations

import importlib
import json
import netrc
import os
from pathlib import Path
from urllib.request import Request, urlopen


class RunTracker:
    """One real W&B run per task/seed when enabled; otherwise a local no-op."""

    def __init__(self, settings: dict, directory: Path):
        self.settings = settings
        self.directory = Path(directory)
        self.mode = settings.get("mode", "disabled")
        if self.mode not in {"disabled", "offline", "online"}:
            raise ValueError("wandb mode must be disabled, offline or online")
        self.run = None
        self.sdk = None

    def start(self, run_id: str, config: dict) -> dict:
        """Start a run and return its actual ID/URL; offline mode never claims an upload."""
        if self.mode == "disabled":
            return {"mode": "disabled", "id": None, "url": None, "uploaded": False}
        if self.mode == "online":
            self._check_online_prerequisites()
        try:
            self.sdk = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError("Install requirements-wandb.txt inside the venv to enable W&B") from exc
        self.directory.mkdir(parents=True, exist_ok=True)
        kwargs = dict(mode=self.mode, id=run_id, name=run_id, config=config, dir=str(self.directory),
                      project=self.settings.get("project") or "forrester-phase1-local",
                      entity=self.settings.get("entity"), tags=["phase1", config["algorithm"]],
                      settings=self.sdk.Settings(init_timeout=30))
        if self.mode == "online":
            kwargs["resume"] = "allow"
        self.run = self.sdk.init(**kwargs)
        if self.run is None:
            raise RuntimeError("W&B did not return a run")
        return {"mode": self.mode, "id": self.run.id,
                "url": self.run.url if self.mode == "online" else None, "uploaded": False}

    def _check_online_prerequisites(self):
        project, entity = self.settings.get("project"), self.settings.get("entity")
        if not project or not entity or self.settings.get("public_project_confirmed") is not True:
            raise RuntimeError("Online W&B requires an existing entity/project and public_project_confirmed=true")
        credentials = bool(os.environ.get("WANDB_API_KEY"))
        if not credentials:
            try:
                credentials = bool(netrc.netrc().authenticators("api.wandb.ai"))
            except (OSError, netrc.NetrcParseError):
                pass
        if not credentials:
            raise RuntimeError("W&B online authentication is missing; no login will be attempted")
        # Anonymous read verifies that the requested existing project is public.
        # This query is also used by the pinned SDK's public Project API.
        payload = {"query": "query($projectName:String!,$entityName:String!){project(name:$projectName,entityName:$entityName){name}}",
                   "variables": {"projectName": project, "entityName": entity}}
        request = Request("https://api.wandb.ai/graphql", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=15) as response:
            result = json.load(response)
        if result.get("errors") or not result.get("data", {}).get("project"):
            raise RuntimeError("W&B project is not anonymously readable; online logging refused")

    def log(self, row: dict) -> None:
        """Log the metrics of one completed objective query, using a one-based step."""
        if self.run is not None:
            metrics = {key: row[key] for key in ("y", "best_so_far", "simple_regret", "normalized_regret", "elapsed_seconds")}
            self.run.log(metrics, step=row["step"])

    def finish(self, metrics: dict, artifacts) -> None:
        """Attach actual files and finish successfully; errors propagate to the evaluator."""
        if self.run is None:
            return
        self.run.summary.update(metrics)
        artifact = self.sdk.Artifact(f"evaluation-{self.run.id}", type="evaluation")
        for path in artifacts:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)
        self.run.finish(exit_code=0)

    def fail(self) -> None:
        """Mark an enabled tracking run failed without claiming successful completion."""
        if self.run is not None:
            self.run.finish(exit_code=1)
