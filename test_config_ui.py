#!/usr/bin/env python3
"""Offline checks that the UI exposes every config key, and that a project
inherits global defaults until it opts out. Run: python test_config_ui.py

No network: uses Flask's test client against a throwaway user dir.
"""

import tempfile
from pathlib import Path

import yaml

import app as A
from pipeline import DEFAULT_CONFIG, load_project_config

# Every leaf in DEFAULT_CONFIG -> the form field that edits it, or None for the
# ones deliberately shown read-only (they're fixed when the project is made).
# A new config key with no entry here fails the test: that's the point.
FIELD_FOR_KEY = {
    "source": None,
    "venue_id": None,
    "submission_invitation": "submission_invitation",
    "max_papers": "max_papers",
    "venue_filter": "venue_filter",
    "queries": "queries",
    "embedding.backend": "backend",
    "embedding.local.model": "local_model",
    "embedding.local.batch_size": "local_batch_size",
    "embedding.api.base_url": "api_base_url",
    "embedding.api.model": "api_model",
    "embedding.api.api_key_env": "api_key_env",
    "embedding.api.batch_size": "api_batch_size",
    "tiers.method": "tier_method",
    "tiers.good_threshold": "good_threshold",
    "tiers.medium_threshold": "medium_threshold",
}

BASE_FORM = {
    "queries": "graph neural networks", "max_papers": "1000",
    "submission_invitation": "Submission", "venue_filter": "accepted",
    "backend": "local", "local_model": "some/model", "local_batch_size": "",
    "api_base_url": "http://localhost:1234/v1", "api_model": "m",
    "api_key_env": "OPENAI_API_KEY", "api_batch_size": "",
    "tier_method": "percentile", "good_threshold": "80", "medium_threshold": "50",
}


def config_leaves(node, prefix=""):
    for key, value in node.items():
        if isinstance(value, dict):
            yield from config_leaves(value, f"{prefix}{key}.")
        else:
            yield prefix + key


def setup(root: Path):
    A.configure_user_dir(str(root))
    client = A.app.test_client()
    client.post("/projects", data={"source": "openreview", "venue_id": "ICLR.cc/2026/Conference"})
    client.post("/projects", data={"source": "cvf", "venue_id": "WACV2024"})
    return client


def test_global_config_seeded(root, client):
    written = yaml.safe_load((root / "global_config.yaml").read_text())
    assert written == {k: DEFAULT_CONFIG[k] for k in A.GLOBAL_CONFIG_KEYS}, written


def test_every_config_key_is_editable(root, client):
    unmapped = [k for k in config_leaves(DEFAULT_CONFIG) if k not in FIELD_FOR_KEY]
    assert not unmapped, f"config keys with no UI: {unmapped}"

    project = client.get("/project/iclr-cc-2026-conference").get_data(as_text=True)
    index = client.get("/").get_data(as_text=True)
    for key, field in FIELD_FOR_KEY.items():
        if field is None:
            continue
        assert f'name="{field}"' in project, f"{key} missing from the project config form"
    for key in config_leaves({k: DEFAULT_CONFIG[k] for k in A.GLOBAL_CONFIG_KEYS}):
        assert f'name="{FIELD_FOR_KEY[key]}"' in index, f"{key} missing from global settings"

    # source-specific fields stay off the pages that can't use them
    cvf = client.get("/project/wacv2024").get_data(as_text=True)
    assert 'name="venue_filter"' not in cvf and 'name="submission_invitation"' not in cvf


def test_globals_are_inherited_until_overridden(root, client):
    project_dir = root / "projects" / "iclr-cc-2026-conference"
    own = lambda: yaml.safe_load((project_dir / "config.yaml").read_text())

    client.post("/project/iclr-cc-2026-conference/config", data=BASE_FORM)
    assert not (set(A.GLOBAL_CONFIG_KEYS) & set(own())), own()

    # ...so a later edit to global_config.yaml still reaches the project
    globals_ = yaml.safe_load((root / "global_config.yaml").read_text())
    globals_["tiers"]["good_threshold"] = 91
    A.save_yaml(root / "global_config.yaml", globals_)
    merged = load_project_config(project_dir, root / "global_config.yaml")
    assert merged["tiers"]["good_threshold"] == 91, merged["tiers"]

    client.post("/project/iclr-cc-2026-conference/config",
                data={**BASE_FORM, "override_globals": "1",
                      "good_threshold": "70", "local_batch_size": "8"})
    assert own()["tiers"]["good_threshold"] == 70
    assert own()["embedding"]["local"]["batch_size"] == 8
    # blank batch size means "auto", not zero
    assert own()["embedding"]["api"]["batch_size"] == 96

    client.post("/project/iclr-cc-2026-conference/config", data=BASE_FORM)
    assert not (set(A.GLOBAL_CONFIG_KEYS) & set(own())), "unticking must restore inheritance"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        client = setup(root)
        test_global_config_seeded(root, client)
        test_every_config_key_is_editable(root, client)
        test_globals_are_inherited_until_overridden(root, client)
    print("ok")
