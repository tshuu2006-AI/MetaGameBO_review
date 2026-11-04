"""No credentials/network in disabled mode; online requires explicit prerequisites."""
import io
import json
import types

import pytest

from experiments import tracking
from experiments.tracking import RunTracker


def test_disabled_does_not_import_sdk_or_access_network(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking.importlib, "import_module", lambda name: pytest.fail("Disabled mode imported SDK"))
    monkeypatch.setattr(tracking, "urlopen", lambda *a, **kw: pytest.fail("Disabled mode accessed network"))
    tracker = RunTracker({"mode": "disabled"}, tmp_path)
    assert tracker.start("test", {}) == {"mode": "disabled", "id": None, "url": None, "uploaded": False}
    tracker.log({})
    tracker.finish({}, [])


def test_online_never_attempts_login_or_network_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking.netrc, "netrc", lambda: types.SimpleNamespace(authenticators=lambda host: None))
    monkeypatch.setattr(tracking, "urlopen", lambda *a, **kw: pytest.fail("Unauthenticated online network call"))
    tracker = RunTracker({"mode": "online", "entity": "team", "project": "project", "public_project_confirmed": True}, tmp_path)
    with pytest.raises(RuntimeError, match="no login"):
        tracker.start("test", {})


def test_online_refuses_nonpublic_project(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(tracking, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps({"data": {"project": None}}).encode()))
    tracker = RunTracker({"mode": "online", "entity": "team", "project": "project", "public_project_confirmed": True}, tmp_path)
    with pytest.raises(RuntimeError, match="not anonymously readable"):
        tracker.start("test", {})


@pytest.mark.parametrize("settings", [{"mode": "unexpected"}, {"mode": "online"}])
def test_invalid_tracking_settings_fail(tmp_path, settings):
    with pytest.raises((ValueError, RuntimeError)):
        RunTracker(settings, tmp_path).start("test", {})
