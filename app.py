#!/usr/bin/env python3
"""
app.py — PaperSieve web UI.

Manages multiple paper-ranking projects side by side, one per venue (see
sources.py for the supported venue hosts). Each project's
fetch -> embed -> rank pipeline (pipeline.py) is run as a subprocess so the
Flask process itself stays responsive for progress polling.

All user data lives under a single mandatory "user directory" argument, kept
separate from this code checkout so it's one self-contained, movable folder.
Everything else is a fixed relative path under it:

    <user_dir>/projects/           per-venue projects (config, cache, output)
    <user_dir>/global_config.yaml
    <user_dir>/.env

Run:
    python app.py /path/to/user_dir                # production (waitress)
    python app.py /path/to/user_dir --debug         # dev server, auto-reload
    -> http://localhost:5000/

See the README's "Deploying" section for running this behind a process
manager (systemd, etc).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

from pipeline import (
    DEFAULT_CONFIG,
    LOCAL_DEVICES,
    RUNTIMES,
    deep_merge,
    load_project_config,
    slugify,
    valid_devices,
    write_status,
)
from sources import DEFAULT_SOURCE, SOURCES, get_source

BASE_DIR = Path(__file__).parent  # code directory: templates/, static/, pipeline.py

app = Flask(__name__)

# The local runtime/device choices are pipeline.py's; exposing them as Jinja
# globals keeps templates/_config_fields.html from hardcoding (and drifting
# from) the sets that resolve_device() actually accepts.
app.jinja_env.globals.update(RUNTIMES=RUNTIMES, LOCAL_DEVICES=LOCAL_DEVICES)

REQUIRED_ENV_KEYS = ["OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD", "OPENAI_API_KEY", "HF_TOKEN"]

# The config sections global_config.yaml owns. A project inherits these unless
# it sets them itself; both forms that edit them post the identical field names
# (see templates/_config_fields.html).
GLOBAL_CONFIG_KEYS = ("embedding", "tiers")

# Fixed relative paths under the mandatory user_dir CLI argument -- set once
# by configure_user_dir() before any request is handled. One source of
# truth: nothing here is independently configurable.
PROJECTS_DIR = None
GLOBAL_CONFIG_PATH = None

# Tracks the currently-running pipeline subprocess per project slug, so
# /status can detect a crashed process (exited without writing a terminal
# status) and /stop can kill a running one. Waitress serves this app with
# threads in a single process, so a plain module-level dict is safe to share
# across requests -- no separate worker processes to coordinate across.
RUNNING_PROCESSES: Dict[str, subprocess.Popen] = {}


def configure_user_dir(user_dir: str) -> None:
    global PROJECTS_DIR, GLOBAL_CONFIG_PATH
    path = Path(user_dir).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"user directory does not exist: {path} (create it first)")
    if not os.access(path, os.W_OK):
        raise SystemExit(f"user directory is not writable: {path}")

    PROJECTS_DIR = path / "projects"
    GLOBAL_CONFIG_PATH = path / "global_config.yaml"
    PROJECTS_DIR.mkdir(exist_ok=True)
    if not GLOBAL_CONFIG_PATH.exists():
        # Write the defaults out on first launch rather than leaving the file
        # absent until someone happens to hit Save: otherwise the Settings page
        # shows values that exist nowhere on disk, and it isn't obvious that
        # what a project inherits is editable at all.
        save_yaml(GLOBAL_CONFIG_PATH, {k: DEFAULT_CONFIG[k] for k in GLOBAL_CONFIG_KEYS})
    load_dotenv(path / ".env")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_status(project_dir: Path) -> dict:
    status_path = project_dir / "status.json"
    if not status_path.exists():
        return {"state": "idle"}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"state": "idle"}


def list_projects() -> list:
    projects = []
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        config = load_yaml(project_dir / "config.yaml")
        status = read_status(project_dir)
        has_results = (project_dir / "output" / "rankings.json").exists()
        projects.append({
            "slug": project_dir.name,
            "venue_id": config.get("venue_id", project_dir.name),
            "source": config.get("source", DEFAULT_SOURCE),
            "status": status.get("state", "idle"),
            "has_results": has_results,
        })
    return projects


def parse_queries(raw: str) -> list:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def parse_optional_int(raw: str, empty=None):
    """Blank batch-size fields mean "pick one for me", not zero."""
    raw = (raw or "").strip()
    return int(raw) if raw else empty


def parse_global_fields(form) -> dict:
    """The embedding/tiers block. The global Settings form and each project's
    override form post exactly these fields, so they parse in one place."""
    # runtime/device come from <select>s, but this is still a trust boundary:
    # validate against the sets pipeline.py accepts and fall back to the
    # defaults rather than persisting whatever was posted.
    runtime = form.get("local_runtime", "torch")
    if runtime not in RUNTIMES:
        runtime = "torch"
    device = form.get("local_device", "auto")
    if device not in valid_devices(runtime):
        device = "auto"
    return {
        "embedding": {
            "backend": form.get("backend", "local"),
            "local": {
                "model": form.get("local_model", "").strip(),
                "runtime": runtime,
                "device": device,
                "batch_size": parse_optional_int(form.get("local_batch_size")),
            },
            "api": {
                "base_url": form.get("api_base_url", "").strip(),
                "model": form.get("api_model", "").strip(),
                "api_key_env": form.get("api_key_env", "OPENAI_API_KEY").strip(),
                # unlike the local backend, the API embedder has no device to
                # auto-size against, so it always needs a concrete number
                "batch_size": parse_optional_int(form.get("api_batch_size"), empty=96),
            },
        },
        "tiers": {
            "method": form.get("tier_method", "percentile"),
            "good_threshold": float(form.get("good_threshold", 80)),
            "medium_threshold": float(form.get("medium_threshold", 50)),
        },
    }


# --------------------------------------------------------------------------
# Routes: index / project creation / global config
# --------------------------------------------------------------------------

@app.route("/")
def index():
    global_config = deep_merge(DEFAULT_CONFIG, load_yaml(GLOBAL_CONFIG_PATH))
    env_status = {key: bool(os.environ.get(key)) for key in REQUIRED_ENV_KEYS}
    return render_template(
        "index.html", projects=list_projects(), config=global_config,
        env_status=env_status, sources=list(SOURCES.values()), default_source=DEFAULT_SOURCE,
    )


@app.route("/projects", methods=["POST"])
def create_project():
    venue_id = request.form["venue_id"].strip()
    source_name = request.form.get("source", DEFAULT_SOURCE)
    if source_name not in SOURCES:
        return f"Unknown source: {source_name}", 400
    source = SOURCES[source_name]
    slug = slugify(venue_id)
    project_dir = PROJECTS_DIR / slug
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "cache").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    # Everything else (queries, max_papers, venue_filter, embedding, tiers)
    # is left unset here and inherited from DEFAULT_CONFIG/global_config.yaml
    # until set on the project's own config form.
    config = {"source": source_name, "venue_id": venue_id}
    if "submission_invitation" in source.uses:
        config["submission_invitation"] = (
            request.form.get("submission_invitation", "Submission").strip() or "Submission"
        )
    save_yaml(project_dir / "config.yaml", config)
    return redirect(url_for("project_page", slug=slug))


@app.route("/settings", methods=["POST"])
def settings():
    config = deep_merge(DEFAULT_CONFIG, parse_global_fields(request.form))
    # only persist the sections global_config.yaml is meant to own
    save_yaml(GLOBAL_CONFIG_PATH, {k: config[k] for k in GLOBAL_CONFIG_KEYS})
    return redirect(url_for("index"))


# --------------------------------------------------------------------------
# Routes: project page / run / status / data / config
# --------------------------------------------------------------------------

@app.route("/project/<slug>")
def project_page(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404
    config = load_project_config(project_dir, GLOBAL_CONFIG_PATH)
    own_config = load_yaml(project_dir / "config.yaml")
    status = read_status(project_dir)
    has_results = (project_dir / "output" / "rankings.json").exists()
    show_results = has_results and status.get("state") != "running"
    return render_template(
        "project.html",
        slug=slug,
        config=config,
        status=status,
        show_results=show_results,
        source=get_source(config.get("source")),
        overrides_globals=any(k in own_config for k in GLOBAL_CONFIG_KEYS),
    )


@app.route("/project/<slug>/run", methods=["POST"])
def run_project(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404

    force_papers = request.form.get("force_refresh_papers") == "1"
    force_embeddings = request.form.get("force_refresh_embeddings") == "1"

    # Write the initial status synchronously, before the subprocess even
    # spawns, so the first /status poll never sees a stale prior run's
    # status while the child is still importing its dependencies.
    write_status(project_dir, state="running", phase="starting", current=0, total=0, error=None)

    cmd = [sys.executable, str(BASE_DIR / "pipeline.py"), "--project", str(project_dir),
           "--global-config", str(GLOBAL_CONFIG_PATH)]
    if force_papers:
        cmd.append("--force-refresh-papers")
    if force_embeddings:
        cmd.append("--force-refresh-embeddings")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(project_dir / "run.log", "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=log_file, stderr=subprocess.STDOUT, env=env)
    RUNNING_PROCESSES[slug] = proc
    return jsonify({"started": True}), 202


@app.route("/project/<slug>/status")
def project_status(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404

    proc = RUNNING_PROCESSES.get(slug)
    if proc is not None and proc.poll() is not None:
        # subprocess has exited -- if it never wrote a terminal status, it
        # crashed (e.g. segfault/OOM-kill) rather than failing cleanly.
        status = read_status(project_dir)
        if status.get("state") == "running":
            write_status(
                project_dir, state="error",
                error=f"pipeline process exited unexpectedly (code {proc.returncode}) — see log below",
            )
        RUNNING_PROCESSES.pop(slug, None)

    status = read_status(project_dir)
    log_path = project_dir / "run.log"
    if log_path.exists():
        status["log"] = log_path.read_text(encoding="utf-8", errors="replace")
    return jsonify(status)


@app.route("/project/<slug>/stop", methods=["POST"])
def stop_project(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404

    proc = RUNNING_PROCESSES.pop(slug, None)
    if proc is None or proc.poll() is not None:
        return jsonify({"stopped": False, "reason": "not running"}), 409

    proc.terminate()
    write_status(project_dir, state="error", error="Stopped by user")
    return jsonify({"stopped": True})


@app.route("/project/<slug>/data")
def project_data(slug):
    project_dir = PROJECTS_DIR / slug
    data_path = project_dir / "output" / "rankings.json"
    if not data_path.exists():
        return jsonify({"error": "no results yet"}), 404
    return app.response_class(data_path.read_bytes(), mimetype="application/json")


@app.route("/project/<slug>/config", methods=["POST"])
def project_config(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404
    config_path = project_dir / "config.yaml"

    config = load_yaml(config_path)
    # `source` is fixed at creation time -- the form never posts it back.
    source = get_source(config.get("source"))
    config["queries"] = parse_queries(request.form.get("queries", ""))
    config["max_papers"] = int(request.form.get("max_papers") or config.get("max_papers", 30000))
    if "submission_invitation" in source.uses:
        config["submission_invitation"] = (
            request.form.get("submission_invitation", "Submission").strip() or "Submission"
        )
    if "venue_filter" in source.uses:
        config["venue_filter"] = parse_queries(request.form.get("venue_filter", ""))
    # Only write the global sections when this project actually overrides them
    # -- writing them unconditionally would pin every project to whatever the
    # globals happened to be at creation time, so later edits to
    # global_config.yaml would never reach it again.
    if request.form.get("override_globals") == "1":
        overrides = parse_global_fields(request.form)
        for key in GLOBAL_CONFIG_KEYS:
            config[key] = deep_merge(config.get(key, {}), overrides[key])
    else:
        for key in GLOBAL_CONFIG_KEYS:
            config.pop(key, None)
    save_yaml(config_path, config)
    return redirect(url_for("project_page", slug=slug))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaperSieve web UI")
    parser.add_argument("user_dir", help="Directory holding all user data: projects/, global_config.yaml, .env")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--debug", action="store_true", default=os.environ.get("DEBUG") == "1",
        help="Run Flask's dev server with the debugger/auto-reload instead of the production server "
             "(same as setting DEBUG=1). Never use this outside your own machine: the interactive "
             "debugger is arbitrary code execution.",
    )
    args = parser.parse_args()
    configure_user_dir(args.user_dir)

    if args.debug:
        app.run(host=args.host, port=args.port, debug=True, threaded=True)
    else:
        from waitress import serve
        serve(app, host=args.host, port=args.port, threads=4)
