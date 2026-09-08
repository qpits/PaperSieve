#!/usr/bin/env python3
"""
sources.py

Paper sources: one adapter per venue host. An adapter is just a function
(config, on_progress) -> list of normalized paper dicts, plus a Source struct
describing how the UI should ask for that source's venue ID.

Adding a new source is one fetch_*() function and one Source(...) literal in
SOURCES -- nothing else in the app needs to know it exists.

Every adapter emits the same paper dict, so the embed/rank/render half of the
app is source-agnostic:

    id, number, title, abstract, tldr, keywords, authors, primary_area,
    venue, venueid, forum_url, arxiv_url
"""

import html
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import openreview
import requests

ProgressCallback = Callable[[str, int, Optional[int]], None]

USER_AGENT = "PaperSieve/0.1 (+https://github.com/; research paper indexing)"


# --------------------------------------------------------------------------
# OpenReview
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
        "arxiv_url": "",
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


def fetch_openreview(config: Dict[str, Any], on_progress: Optional[ProgressCallback] = None) -> List[dict]:
    venue_id = config["venue_id"]
    if not venue_id:
        raise ValueError("config.venue_id is required")
    inv_type = config["submission_invitation"]
    cap = config["max_papers"]
    venue_filter = config.get("venue_filter") or []
    invitation = f"{venue_id}/-/{inv_type}"

    print(f"Authenticating with OpenReview and fetching '{invitation}' (venue_filter={venue_filter})...", flush=True)
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

    # venue_filter is OpenReview vocabulary (it matches the venue/venueid
    # content fields), so it's applied here rather than in the shared
    # pipeline -- the server-side pre-filter above can only narrow to whole
    # decision outcomes, not to the oral/poster/spotlight split.
    papers = [p for p in papers if matches_venue_filter(p, venue_filter, venue_id)]
    if not papers:
        raise RuntimeError(
            "No papers left after applying venue_filter -- check the filter terms against "
            "the venue/venueid values actually present."
        )
    return papers


# --------------------------------------------------------------------------
# CVF (openaccess.thecvf.com -- CVPR / ICCV / WACV)
# --------------------------------------------------------------------------

CVF_BASE = "https://openaccess.thecvf.com"
CVF_WORKERS = 8

# ponytail: regexes over CVF's machine-generated markup -- it's uniform enough
# that a parser is overkill (the entry regex below matches 2716/2716 papers on
# CVPR2024). Swap in beautifulsoup4 if CVF ever restyles the listing.
_CVF_ENTRY_SPLIT = re.compile(r'(?=<dt class="ptitle">)')
_CVF_TITLE_RE = re.compile(r'<dt class="ptitle">.*?<a href="([^"]+)">(.*?)</a>', re.S)
_CVF_AUTHOR_RE = re.compile(r'name="query_author" value="([^"]*)"')
_CVF_ARXIV_RE = re.compile(r'href="(https?://arxiv\.org/abs/[^"]+)"')
_CVF_ABSTRACT_RE = re.compile(r'<div id="abstract"[^>]*>(.*?)</div>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _text(raw: str) -> str:
    """CVF markup carries HTML entities (&quot;, &amp;), the odd inline tag, and
    source indentation inside titles/abstracts; flatten all three to one line of
    plain text."""
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub("", raw))).strip()


_local = threading.local()


def _cvf_session() -> requests.Session:
    # requests.Session isn't documented as thread-safe, so give each worker
    # thread its own (they still pool connections individually).
    session = getattr(_local, "session", None)
    if session is None:
        session = _local.session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
    return session


def _cvf_get(url: str, timeout: int = 60) -> str:
    resp = _cvf_session().get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_cvf_listing(page: str, venue_id: str) -> List[dict]:
    """Parse the ?day=all listing into papers with everything except the
    abstract (which only exists on each paper's own detail page)."""
    papers = []
    for block in _CVF_ENTRY_SPLIT.split(page)[1:]:
        match = _CVF_TITLE_RE.match(block)
        if not match:
            continue
        href, raw_title = match.groups()
        arxiv = _CVF_ARXIV_RE.search(block)
        paper_id = href.rsplit("/", 1)[-1].removesuffix(".html").removesuffix("_paper")
        papers.append({
            "id": paper_id,
            "number": None,
            "title": _text(raw_title) or "(untitled)",
            "abstract": "",
            "tldr": "",
            "keywords": [],
            "authors": [_text(a) for a in _CVF_AUTHOR_RE.findall(block)],
            "primary_area": "",
            "venue": venue_id,
            "venueid": "",
            "forum_url": CVF_BASE + href if href.startswith("/") else href,
            "arxiv_url": arxiv.group(1) if arxiv else "",
        })
    return papers


def parse_cvf_abstract(page: str) -> str:
    match = _CVF_ABSTRACT_RE.search(page)
    return _text(match.group(1)) if match else ""


def fetch_cvf(config: Dict[str, Any], on_progress: Optional[ProgressCallback] = None) -> List[dict]:
    venue_id = (config.get("venue_id") or "").strip().strip("/")
    if not venue_id:
        raise ValueError("config.venue_id is required")

    listing_url = f"{CVF_BASE}/{venue_id}?day=all"
    bad_id_help = (
        f"Check the conference ID (e.g. 'CVPR2025', 'ICCV2025', 'WACV2024' -- see "
        f"{CVF_BASE}/menu). Venues from 2020 and earlier use a different, day-split "
        "URL layout and aren't supported."
    )
    print(f"Fetching CVF listing {listing_url} ...", flush=True)
    try:
        listing = _cvf_get(listing_url)
    except requests.HTTPError as e:
        raise RuntimeError(f"CVF returned {e.response.status_code} for {listing_url}. {bad_id_help}") from e

    papers = parse_cvf_listing(listing, venue_id)
    if not papers:
        raise RuntimeError(f"No papers found at {listing_url}. {bad_id_help}")

    papers = papers[: config["max_papers"]]
    total = len(papers)
    print(f"Found {total} papers; fetching abstracts from their detail pages...", flush=True)

    # The listing has no abstracts, and only about half of the entries carry an
    # arXiv link -- every paper's own CVF page has one, so take them from there.
    done = 0
    lock = threading.Lock()

    def load_abstract(paper: dict) -> None:
        nonlocal done
        try:
            paper["abstract"] = parse_cvf_abstract(_cvf_get(paper["forum_url"]))
        except Exception as e:  # a single unreachable page shouldn't kill the run
            print(f"  warning: no abstract for {paper['id']}: {e}", flush=True)
        with lock:
            done += 1
            # throttled: on_progress rewrites status.json, and there can be
            # thousands of papers
            if on_progress and (done % 25 == 0 or done == total):
                on_progress("fetching", done, total)

    if on_progress:
        on_progress("fetching", 0, total)
    with ThreadPoolExecutor(CVF_WORKERS) as pool:
        list(pool.map(load_abstract, papers))

    missing = sum(1 for p in papers if not p["abstract"])
    if missing:
        print(f"{missing}/{total} papers have no abstract.", flush=True)
    return papers


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    name: str                      # stored as `source` in a project's config.yaml
    label: str                     # shown in the source picker
    id_label: str                  # what to call the venue id in forms
    id_placeholder: str
    id_hint: str                   # where to find it / how to format it
    fetch: Callable[[Dict[str, Any], Optional[ProgressCallback]], List[dict]]
    # config fields this source honours -- drives which form sections show
    uses: Tuple[str, ...] = ()
    # what a refetch depends on: when this changes, the paper cache is stale
    cache_key: Callable[[Dict[str, Any]], Any] = field(default=lambda cfg: None)


OPENREVIEW = Source(
    name="openreview",
    label="OpenReview",
    id_label="Venue ID",
    id_placeholder="ICLR.cc/2026/Conference",
    id_hint=(
        "The venue's OpenReview ID, taken from its URL "
        "(openreview.net/group?id=<b>ICLR.cc/2026/Conference</b>)."
    ),
    fetch=fetch_openreview,
    uses=("submission_invitation", "venue_filter"),
    cache_key=lambda cfg: cfg.get("venue_filter") or [],
)

CVF = Source(
    name="cvf",
    label="CVF (CVPR / ICCV / WACV)",
    id_label="Conference ID",
    id_placeholder="CVPR2025",
    id_hint=(
        "The conference's path on the CVF Open Access site "
        "(openaccess.thecvf.com/<b>CVPR2025</b>) — e.g. <code>CVPR2025</code>, "
        "<code>ICCV2025</code>, <code>WACV2024</code>. The full list is at "
        '<a href="https://openaccess.thecvf.com/menu" target="_blank" rel="noopener">'
        "openaccess.thecvf.com/menu</a>; 2021 and later only. CVF publishes accepted "
        "papers only, so there's nothing to filter."
    ),
    fetch=fetch_cvf,
)

SOURCES: Dict[str, Source] = {s.name: s for s in (OPENREVIEW, CVF)}
DEFAULT_SOURCE = OPENREVIEW.name


def get_source(name: Optional[str]) -> Source:
    source = SOURCES.get(name or DEFAULT_SOURCE)
    if source is None:
        raise ValueError(f"Unknown source {name!r} (known: {', '.join(SOURCES)})")
    return source
