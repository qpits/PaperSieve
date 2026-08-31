#!/usr/bin/env python3
"""
app.py — PaperSieve web UI.

Manages multiple OpenReview ranking projects side by side. Each project's
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

import yaml
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

from pipeline import DEFAULT_CONFIG, deep_merge, load_project_config, slugify

BASE_DIR = Path(__file__).parent  # code directory: templates/, static/, pipeline.py

app = Flask(__name__)

REQUIRED_ENV_KEYS = ["OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD", "OPENAI_API_KEY", "HF_TOKEN"]

# Fixed relative paths under the mandatory user_dir CLI argument -- set once
# by configure_user_dir() before any request is handled. One source of
# truth: nothing here is independently configurable.
PROJECTS_DIR = None
GLOBAL_CONFIG_PATH = None


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
            "status": status.get("state", "idle"),
            "has_results": has_results,
        })
    return projects


def parse_queries(raw: str) -> list:
    return [line.strip() for line in raw.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Routes: index / project creation / settings
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", projects=list_projects())


@app.route("/projects", methods=["POST"])
def create_project():
    venue_id = request.form["venue_id"].strip()
    slug = slugify(venue_id)
    project_dir = PROJECTS_DIR / slug
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "cache").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)

    config = {
        "venue_id": venue_id,
        "submission_invitation": request.form.get("submission_invitation", "Submission").strip() or "Submission",
        "max_papers": int(request.form.get("max_papers") or 5000),
        "queries": parse_queries(request.form.get("queries", "")),
    }
    save_yaml(project_dir / "config.yaml", config)
    return redirect(url_for("project_page", slug=slug))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        config = deep_merge(DEFAULT_CONFIG, {
            "embedding": {
                "backend": request.form.get("backend", "local"),
                "local": {"model": request.form.get("local_model", "").strip()},
                "api": {
                    "base_url": request.form.get("api_base_url", "").strip(),
                    "model": request.form.get("api_model", "").strip(),
                    "api_key_env": request.form.get("api_key_env", "OPENAI_API_KEY").strip(),
                },
            },
            "tiers": {
                "method": request.form.get("tier_method", "percentile"),
                "good_threshold": float(request.form.get("good_threshold", 80)),
                "medium_threshold": float(request.form.get("medium_threshold", 50)),
            },
        })
        # only persist the fields global_config.yaml is meant to own
        save_yaml(GLOBAL_CONFIG_PATH, {
            "embedding": config["embedding"],
            "tiers": config["tiers"],
        })
        return redirect(url_for("settings"))

    global_config = deep_merge(DEFAULT_CONFIG, load_yaml(GLOBAL_CONFIG_PATH))
    env_status = {key: bool(__import__("os").environ.get(key)) for key in REQUIRED_ENV_KEYS}
    return render_template("settings.html", config=global_config, env_status=env_status)


# --------------------------------------------------------------------------
# Routes: project page / run / status / data / config
# --------------------------------------------------------------------------

@app.route("/project/<slug>")
def project_page(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404
    config = load_yaml(project_dir / "config.yaml")
    status = read_status(project_dir)
    has_results = (project_dir / "output" / "rankings.json").exists()
    show_results = has_results and status.get("state") != "running"
    return render_template(
        "project.html",
        slug=slug,
        config=config,
        status=status,
        show_results=show_results,
    )


@app.route("/project/<slug>/run", methods=["POST"])
def run_project(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404

    force_papers = request.form.get("force_refresh_papers") == "1"
    force_embeddings = request.form.get("force_refresh_embeddings") == "1"

    cmd = [sys.executable, str(BASE_DIR / "pipeline.py"), "--project", str(project_dir),
           "--global-config", str(GLOBAL_CONFIG_PATH)]
    if force_papers:
        cmd.append("--force-refresh-papers")
    if force_embeddings:
        cmd.append("--force-refresh-embeddings")

    subprocess.Popen(cmd, cwd=str(BASE_DIR))
    return jsonify({"started": True}), 202


@app.route("/project/<slug>/status")
def project_status(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404
    return jsonify(read_status(project_dir))


@app.route("/project/<slug>/data")
def project_data(slug):
    project_dir = PROJECTS_DIR / slug
    data_path = project_dir / "output" / "rankings.json"
    if not data_path.exists():
        return jsonify({"error": "no results yet"}), 404
    return app.response_class(data_path.read_bytes(), mimetype="application/json")


@app.route("/project/<slug>/config", methods=["GET", "POST"])
def project_config(slug):
    project_dir = PROJECTS_DIR / slug
    if not project_dir.is_dir():
        return "Project not found", 404
    config_path = project_dir / "config.yaml"

    if request.method == "POST":
        config = load_yaml(config_path)
        config["queries"] = parse_queries(request.form.get("queries", ""))
        config["max_papers"] = int(request.form.get("max_papers") or config.get("max_papers", 5000))
        config["venue_filter"] = parse_queries(request.form.get("venue_filter", ""))
        config["embedding"] = deep_merge(config.get("embedding", {}), {
            "backend": request.form.get("backend", "local"),
            "local": {"model": request.form.get("local_model", "").strip()},
        })
        config["tiers"] = {
            "method": request.form.get("tier_method", "percentile"),
            "good_threshold": float(request.form.get("good_threshold", 80)),
            "medium_threshold": float(request.form.get("medium_threshold", 50)),
        }
        save_yaml(config_path, config)
        return redirect(url_for("project_page", slug=slug))

    config = load_project_config(project_dir, GLOBAL_CONFIG_PATH)
    local_config = load_yaml(config_path)
    return render_template("project_config.html", slug=slug, config=config, local_config=local_config)


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
