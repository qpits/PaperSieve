#!/usr/bin/env python3
"""
pipeline.py

Core fetch -> embed -> rank pipeline for a single PaperSieve project.
Importable (run_pipeline) for use from app.py, and runnable standalone:

    python pipeline.py --project projects/iclr2026-conference
    python pipeline.py --project projects/iclr2026-conference --force-refresh-papers
    python pipeline.py --project projects/iclr2026-conference --force-refresh-embeddings
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import openreview
import requests

try:
    import yaml
except ImportError:
    yaml = None


ProgressCallback = Callable[[str, int, Optional[int]], None]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "venue_id": "",
    "submission_invitation": "Submission",
    "max_papers": 5000,
    # Case-insensitive substrings matched against each note's `venue` /
    # `venueid` content field (e.g. "poster", "oral", "spotlight" -- or
    # "Rejected"/"Withdrawn" to explicitly include those). "accepted" is a
    # special term matching any accepted paper regardless of oral/poster/
    # spotlight wording. Defaults to accepted-only; clear the field (save an
    # empty list) to fetch everything instead. Only meaningful once
    # decisions are released; before that, every note's venue field is empty
    # and nothing gets filtered out.
    "venue_filter": ["accepted"],
    "queries": [],
    "embedding": {
        "backend": "local",  # "local" or "api"
        "local": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "batch_size": None,  # None -> auto-picked from detected device
        },
        "api": {
            "base_url": "https://api.openai.com/v1",
            "model": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY",
            "batch_size": 96,
        },
    },
    "tiers": {
        "method": "percentile",  # "percentile" or "absolute"
        "good_threshold": 80,
        "medium_threshold": 50,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required. Run: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_project_config(project_dir: Path, global_config_path: Optional[Path] = None) -> Dict[str, Any]:
    global_cfg = load_yaml(global_config_path) if global_config_path else {}
    project_cfg = load_yaml(project_dir / "config.yaml")
    merged = deep_merge(DEFAULT_CONFIG, global_cfg)
    merged = deep_merge(merged, project_cfg)
    return merged


def slugify(venue_id: str) -> str:
    slug = venue_id.strip().lower()
    slug = "".join(c if c.isalnum() else "-" for c in slug)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "project"


# --------------------------------------------------------------------------
# Status reporting (used by app.py to drive the progress bar)
# --------------------------------------------------------------------------

def write_status(project_dir: Path, **fields: Any) -> None:
    status_path = project_dir / "status.json"
    status: Dict[str, Any] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status = {}
    status.update(fields)
    status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status_path.write_text(json.dumps(status), encoding="utf-8")


def make_status_callback(project_dir: Path) -> ProgressCallback:
    def callback(phase: str, current: int, total: int) -> None:
        write_status(project_dir, state="running", phase=phase, current=current, total=total)
    return callback


# --------------------------------------------------------------------------
# OpenReview fetching
# --------------------------------------------------------------------------

def get_field(content: dict, keys: List[str]) -> Any:
    for k in keys:
        if k in content and content[k] is not None:
            v = content[k]
            if isinstance(v, dict) and "value" in v:
                return v["value"]
            return v
    return None


def make_openreview_client(base_url: str):
    # openreview-py reads OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD from the
    # environment itself (see openreview.api.OpenReviewClient.__init__) and
    # logs in automatically if they're set; anonymous access is used
    # otherwise. As of 2026, OpenReview's /notes endpoint returns a 403
    # "ChallengeRequiredError" bot-detection wall for fully anonymous
    # requests, so a logged-in account is effectively required.
    client_cls = openreview.api.OpenReviewClient if base_url == "https://api2.openreview.net" else openreview.Client
    return client_cls(baseurl=base_url)


def fetch_notes_page(client, invitation: str, limit: int, offset: int,
                      content: Optional[dict] = None, max_retries: int = 5) -> list:
    backoff = 2.0
    for attempt in range(max_retries + 1):
        try:
            return client.get_notes(invitation=invitation, limit=limit, offset=offset, content=content)
        except openreview.OpenReviewException as e:
            details = e.args[0] if e.args else {}
            status = details.get("status") if isinstance(details, dict) else None
            if status == 429 and attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(f"OpenReview error for invitation {invitation!r}: {details}") from e


def fetch_total_count(client, invitation: str, content: Optional[dict] = None, max_retries: int = 5) -> Optional[int]:
    """Best-effort: OpenReview can report the total match count for an
    invitation up front (via with_count=True, only when offset is omitted).
    Used to make the fetch progress bar accurate instead of just tracking
    towards max_papers. Returns None if this isn't available for some reason
    -- callers should fall back to an indeterminate progress indicator."""
    backoff = 2.0
    for attempt in range(max_retries + 1):
        try:
            result = client.get_notes(invitation=invitation, limit=1, content=content, with_count=True)
            return result[1] if isinstance(result, tuple) else None
        except openreview.OpenReviewException as e:
            details = e.args[0] if e.args else {}
            status = details.get("status") if isinstance(details, dict) else None
            if status == 429 and attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None


def fetch_all_notes(client, invitation: str, cap: int, on_progress: Optional[ProgressCallback] = None,
                     content: Optional[dict] = None) -> List[dict]:
    total_count = fetch_total_count(client, invitation, content=content)
    total_for_progress = min(total_count, cap) if total_count is not None else None

    limit = 1000
    offset = 0
    all_notes: List[dict] = []
    while True:
        notes = fetch_notes_page(client, invitation, limit, offset, content=content)
        batch = [n.to_json() for n in notes]
        all_notes.extend(batch)
        if on_progress:
            on_progress("fetching", min(len(all_notes), cap), total_for_progress)
        if len(batch) < limit or len(all_notes) >= cap:
            break
        offset += limit
    return all_notes[:cap]


def discover_venue_decision_ids(client, venue_id: str) -> Dict[str, str]:
    """Best-effort: OpenReview venues expose their decision-outcome venueids
    (e.g. accepted/oral/poster/withdrawn/desk-rejected) as content fields
    ending in "_venue_id" on the venue's own Group entity -- populated from
    the venue's request-form settings. Returns {field_name: venueid_value};
    empty if the group isn't reachable or hasn't got any (e.g. decisions not
    released yet)."""
    try:
        group = client.get_group(venue_id)
    except Exception:
        return {}
    content = group.content or {}
    venue_ids = {}
    for key, val in content.items():
        if not key.endswith("_venue_id"):
            continue
        value = val.get("value") if isinstance(val, dict) else val
        if isinstance(value, str) and value:
            venue_ids[key] = value
    # Accepted papers (oral/poster/spotlight etc.) generally aren't split
    # into their own venueid -- they just keep venueid == venue_id itself,
    # distinguished only by the free-text "venue" field. Add it as a
    # synthetic "accepted" candidate so venue_filter: ["accepted"] can still
    # be resolved server-side (excluding rejected/withdrawn/desk-rejected),
    # even though the oral/poster/spotlight split still needs a client-side
    # match against "venue" after fetching.
    if "accepted_venue_id" not in venue_ids:
        venue_ids["accepted_venue_id"] = venue_id
    return venue_ids


def match_filtered_venue_ids(venue_id_map: Dict[str, str], venue_filter: List[str]) -> List[str]:
    """Match venue_filter terms (case-insensitive substrings) against both
    the decision field name (e.g. "poster_venue_id") and its venueid value
    (e.g. "ICLR.cc/2024/Conference/Poster"), so a filter term like "poster"
    matches regardless of which one it happens to appear in."""
    matched = []
    for key, value in venue_id_map.items():
        hay = f"{key} {value}".lower()
        if any(term.lower() in hay for term in venue_filter):
            matched.append(value)
    return matched


def normalize_note(note: dict) -> dict:
    content = note.get("content", {}) or {}
    authors = get_field(content, ["authors"]) or []
    if not isinstance(authors, list):
        authors = [authors]
    keywords = get_field(content, ["keywords"]) or []
    if not isinstance(keywords, list):
        keywords = [keywords]
    forum_id = note.get("forum") or note.get("id")
    return {
        "id": note.get("id"),
        "number": note.get("number"),
        "title": get_field(content, ["title"]) or "(untitled)",
        "abstract": get_field(content, ["abstract"]) or "",
        "tldr": get_field(content, ["TLDR", "tldr", "TL;DR"]) or "",
        "keywords": keywords,
        "authors": authors,
        "primary_area": get_field(content, ["primary_area"]) or "",
        "venue": get_field(content, ["venue"]) or "",
        "venueid": get_field(content, ["venueid"]) or "",
        "forum_url": f"https://openreview.net/forum?id={forum_id}",
    }


def matches_venue_filter(paper: dict, venue_filter: List[str], venue_id: Optional[str] = None) -> bool:
    if not venue_filter:
        return True
    hay = f"{paper.get('venue', '')} {paper.get('venueid', '')}".lower()
    for term in venue_filter:
        term_l = term.lower()
        if term_l in hay:
            return True
        # mirror discover_venue_decision_ids()'s synthetic "accepted" meaning:
        # accepted papers keep venueid == venue_id itself, with no "accepted"
        # substring anywhere in their venue/venueid text.
        if term_l == "accepted" and venue_id and paper.get("venueid") == venue_id:
            return True
    return False


def fetch_papers(
    config: Dict[str, Any],
    project_dir: Path,
    force_refresh: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> List[dict]:
    cache_path = project_dir / "cache" / "papers.json"
    meta_path = project_dir / "cache" / "fetch_meta.json"
    venue_filter = config.get("venue_filter") or []

    # cache is keyed by venue_filter too: if it changed since the last fetch
    # (server-side filtering means the cache may not contain everything
    # anymore), refetch automatically even without --force-refresh-papers.
    cached_filter = None
    if meta_path.exists():
        try:
            cached_filter = json.loads(meta_path.read_text(encoding="utf-8")).get("venue_filter")
        except (json.JSONDecodeError, OSError):
            cached_filter = None

    if not force_refresh and cache_path.exists() and cached_filter == venue_filter:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    venue_id = config["venue_id"]
    if not venue_id:
        raise ValueError("config.venue_id is required")
    inv_type = config["submission_invitation"]
    cap = config["max_papers"]
    invitation = f"{venue_id}/-/{inv_type}"

    v2_client = make_openreview_client("https://api2.openreview.net")

    notes: List[dict] = []
    if venue_filter:
        # pre-check: only request the decision-outcome subsets that match
        # venue_filter (e.g. "poster", "oral"), instead of pulling every
        # submission (including rejected/withdrawn) and filtering after.
        venue_id_map = discover_venue_decision_ids(v2_client, venue_id)
        matched_ids = match_filtered_venue_ids(venue_id_map, venue_filter)
        if matched_ids:
            seen_ids = set()
            for decision_venueid in matched_ids:
                batch = fetch_all_notes(
                    v2_client, invitation, cap, on_progress, content={"venueid": decision_venueid}
                )
                for n in batch:
                    if n.get("id") not in seen_ids:
                        seen_ids.add(n.get("id"))
                        notes.append(n)
                if len(notes) >= cap:
                    break
            notes = notes[:cap]
        # if the venue group isn't reachable yet or has no matching decision
        # ids (e.g. decisions not released), fall through to the full fetch
        # below and let matches_venue_filter() filter client-side instead.

    if not notes:
        notes = fetch_all_notes(v2_client, invitation, cap, on_progress)

    if not notes:
        v1_client = make_openreview_client("https://api.openreview.net")
        notes = fetch_all_notes(v1_client, invitation, cap, on_progress)

    if not notes and inv_type == "Submission":
        v1_client = make_openreview_client("https://api.openreview.net")
        alt_invitation = f"{venue_id}/-/Blind_Submission"
        notes = fetch_all_notes(v1_client, alt_invitation, cap, on_progress)

    if not notes:
        raise RuntimeError(
            "No papers found. Double-check venue_id and submission_invitation in the config."
        )

    papers = [normalize_note(n) for n in notes]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(papers, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"venue_filter": venue_filter}, f)

    return papers


# --------------------------------------------------------------------------
# Device auto-detection (simple, no memory-budget planning)
# --------------------------------------------------------------------------

def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def default_batch_size(device: str) -> int:
    return {"cuda": 64, "mps": 64, "cpu": 16}.get(device, 16)


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------

def chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def make_local_embedder(cfg: Dict[str, Any]) -> Tuple[Callable[[List[str]], np.ndarray], str]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is required for the 'local' embedding backend. "
            "Run: pip install sentence-transformers"
        ) from e

    model_name = cfg["model"]
    device = detect_device()
    batch_size = cfg.get("batch_size") or default_batch_size(device)
    hf_token = os.environ.get("HF_TOKEN") or None  # only needed for gated/private HF models
    model = SentenceTransformer(model_name, device=device, token=hf_token)

    def embed(texts: List[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return embed, device


def make_api_embedder(cfg: Dict[str, Any]) -> Callable[[List[str]], np.ndarray]:
    base_url = cfg["base_url"].rstrip("/")
    model = cfg["model"]
    batch_size = cfg.get("batch_size", 96)
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    # Local OpenAI-compatible servers (Ollama, vLLM, LM Studio, ...) usually
    # ignore Authorization entirely, so a missing key there isn't fatal —
    # only send the header if we actually have one.
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def embed(texts: List[str]) -> np.ndarray:
        out: List[List[float]] = []
        for batch in chunked(texts, batch_size):
            resp = requests.post(
                f"{base_url}/embeddings",
                headers=headers,
                json={"model": model, "input": batch},
                timeout=120,
            )
            if resp.status_code == 429:
                time.sleep(5)
                resp = requests.post(
                    f"{base_url}/embeddings",
                    headers=headers,
                    json={"model": model, "input": batch},
                    timeout=120,
                )
            if not resp.ok:
                raise RuntimeError(f"Embedding API error (HTTP {resp.status_code}): {resp.text[:300]}")
            data = resp.json()["data"]
            data.sort(key=lambda d: d["index"])
            out.extend([d["embedding"] for d in data])
        return np.array(out, dtype=np.float32)

    return embed


def get_embedder(embedding_cfg: Dict[str, Any]) -> Tuple[Callable[[List[str]], np.ndarray], str]:
    backend = embedding_cfg.get("backend", "local")
    if backend == "local":
        cfg = embedding_cfg["local"]
        embed, device = make_local_embedder(cfg)
        return embed, f"local:{cfg['model']}"
    elif backend == "api":
        cfg = embedding_cfg["api"]
        return make_api_embedder(cfg), f"api:{cfg['base_url']}:{cfg['model']}"
    else:
        raise ValueError(f"Unknown embedding backend: {backend!r} (use 'local' or 'api')")


# --------------------------------------------------------------------------
# Embedding cache (hash-keyed on model_tag + text -> automatically
# invalidated when the embedding model changes, no explicit bookkeeping needed)
# --------------------------------------------------------------------------

def text_hash(model_tag: str, text: str) -> str:
    return hashlib.sha256(f"{model_tag}::{text}".encode("utf-8")).hexdigest()


def load_embedding_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    hashes = data["hashes"].tolist()
    vectors = data["vectors"]
    return {h: vectors[i] for i, h in enumerate(hashes)}


def save_embedding_cache(path: Path, cache: Dict[str, np.ndarray]) -> None:
    if not cache:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = list(cache.keys())
    vectors = np.array([cache[h] for h in hashes], dtype=np.float32)
    np.savez_compressed(path, hashes=np.array(hashes, dtype=object), vectors=vectors)


def paper_text(paper: dict) -> str:
    # Title first (it's the strongest signal and reads naturally as a
    # header), then keywords as an explicit labeled list (they're bare terms,
    # not sentences -- a plain join would read as a run-on), then tldr/
    # abstract as prose. Blank-line-separated so the embedding model sees
    # clear section breaks rather than one run-on paragraph.
    keywords = paper.get("keywords") or []
    parts = [paper.get("title", "")]
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))
    parts.append(paper.get("tldr", ""))
    parts.append(paper.get("abstract", ""))
    return "\n\n".join(p for p in parts if p)[:6000]


def embed_with_cache(
    texts: List[str],
    model_tag: str,
    embedder: Callable[[List[str]], np.ndarray],
    cache_path: Path,
    force_refresh: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> np.ndarray:
    hashes = [text_hash(model_tag, t) for t in texts]

    cache: Dict[str, np.ndarray] = {}
    if not force_refresh:
        cache = load_embedding_cache(cache_path)

    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        done = 0
        total = len(missing_idx)
        if on_progress:
            on_progress("embedding", done, total)
        # embed in reasonably sized chunks so progress updates during a long run
        step = max(1, min(200, total))
        for chunk_start in range(0, total, step):
            chunk_idx = missing_idx[chunk_start : chunk_start + step]
            chunk_texts = [texts[i] for i in chunk_idx]
            new_vectors = embedder(chunk_texts)
            for idx, vec in zip(chunk_idx, new_vectors):
                cache[hashes[idx]] = np.asarray(vec, dtype=np.float32)
            done += len(chunk_idx)
            if on_progress:
                on_progress("embedding", done, total)
        save_embedding_cache(cache_path, cache)
    else:
        if on_progress:
            on_progress("embedding", len(texts), len(texts))

    return np.array([cache[h] for h in hashes], dtype=np.float32)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return mat / norms


def assign_tier(value: float, tier_cfg: Dict[str, Any]) -> str:
    good_t = tier_cfg.get("good_threshold", 80)
    med_t = tier_cfg.get("medium_threshold", 50)
    if value >= good_t:
        return "good"
    if value >= med_t:
        return "medium"
    return "low"


def compute_rankings(
    papers: List[dict],
    paper_embeddings: np.ndarray,
    queries: List[str],
    query_embeddings: np.ndarray,
    tier_cfg: Dict[str, Any],
) -> List[Dict[str, dict]]:
    n = len(papers)
    method = tier_cfg.get("method", "percentile")
    norm_papers = normalize_rows(paper_embeddings)
    norm_queries = normalize_rows(query_embeddings)

    per_paper_rankings: List[Dict[str, dict]] = [dict() for _ in range(n)]

    for qi, query in enumerate(queries):
        scores = norm_papers @ norm_queries[qi]
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        meta_scores = 100.0 * (n - 1 - ranks) / max(n - 1, 1)

        for i in range(n):
            score = float(scores[i])
            meta = float(meta_scores[i])
            tier_value = meta if method == "percentile" else score
            per_paper_rankings[i][query] = {
                "score": round(score, 4),
                "meta_score": round(meta, 2),
                "rank": int(ranks[i]) + 1,
                "tier": assign_tier(tier_value, tier_cfg),
            }

    return per_paper_rankings


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def build_output_data(
    papers: List[dict],
    rankings: List[Dict[str, dict]],
    queries: List[str],
    config: Dict[str, Any],
    model_tag: str,
) -> Dict[str, Any]:
    papers_out = []
    for paper, ranking in zip(papers, rankings):
        entry = dict(paper)
        entry["rankings"] = ranking
        papers_out.append(entry)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "venue_id": config["venue_id"],
        "queries": queries,
        "embedding_model": model_tag,
        "tier_config": config["tiers"],
        "paper_count": len(papers_out),
        "papers": papers_out,
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pipeline(
    project_dir: Path,
    global_config_path: Optional[Path] = None,
    force_refresh_papers: bool = False,
    force_refresh_embeddings: bool = False,
) -> None:
    write_status(project_dir, state="running", phase="starting", current=0, total=0, error=None)
    on_progress = make_status_callback(project_dir)

    try:
        config = load_project_config(project_dir, global_config_path)

        queries = config.get("queries") or []
        if not queries:
            raise ValueError("config.queries is empty — add at least one query string.")

        papers = fetch_papers(config, project_dir, force_refresh=force_refresh_papers, on_progress=on_progress)

        venue_filter = config.get("venue_filter") or []
        papers = [paper for paper in papers if matches_venue_filter(paper, venue_filter, config.get("venue_id"))]
        if not papers:
            raise RuntimeError(
                "No papers left after applying venue_filter — check the filter terms against "
                "the venue/venueid values actually present (see cache/papers.json)."
            )

        embedder, model_tag = get_embedder(config["embedding"])
        embeddings_cache_path = project_dir / "cache" / "embeddings.npz"

        paper_texts = [paper_text(p) for p in papers]
        paper_embeddings = embed_with_cache(
            paper_texts, model_tag, embedder, embeddings_cache_path,
            force_refresh=force_refresh_embeddings, on_progress=on_progress,
        )
        query_embeddings = embed_with_cache(
            queries, model_tag, embedder, embeddings_cache_path,
            force_refresh=force_refresh_embeddings, on_progress=on_progress,
        )

        write_status(project_dir, state="running", phase="ranking", current=0, total=1)
        rankings = compute_rankings(papers, paper_embeddings, queries, query_embeddings, config["tiers"])
        output_data = build_output_data(papers, rankings, queries, config, model_tag)
        write_json(project_dir / "output" / "rankings.json", output_data)

        write_status(project_dir, state="done", phase="done", current=1, total=1, error=None)
    except Exception as e:
        write_status(
            project_dir, state="error", error=f"{e}\n{traceback.format_exc(limit=3)}"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description="Fetch, embed, and rank OpenReview papers for one project.")
    parser.add_argument("--project", required=True, help="Path to a project directory (containing config.yaml).")
    parser.add_argument("--global-config", default=None, help="Path to global_config.yaml (defaults merged under project config).")
    parser.add_argument("--force-refresh-papers", action="store_true")
    parser.add_argument("--force-refresh-embeddings", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project)
    global_config_path = Path(args.global_config) if args.global_config else (project_dir.parent.parent / "global_config.yaml")

    try:
        run_pipeline(
            project_dir,
            global_config_path=global_config_path,
            force_refresh_papers=args.force_refresh_papers,
            force_refresh_embeddings=args.force_refresh_embeddings,
        )
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
