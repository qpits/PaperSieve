# PaperSieve

Fetches every paper from a conference venue — **OpenReview**-hosted (ICLR,
NeurIPS, …) or **CVF**-hosted (CVPR, ICCV, WACV) — ranks them against your own
search queries using embeddings, and gives you a browsable results page
— nothing is filtered out, papers are just sorted into "good / medium / low"
match tiers per query.

Each venue you index is a **project** with its own folder under `projects/`,
so you can keep ICLR, NeurIPS, CVPR, etc. side by side. Fetched papers and their
embeddings are cached, so adding a query or re-running later doesn't
recompute everything from scratch.

All user data — projects, `global_config.yaml`, `.env` — lives under one
**user directory** that you pass on the command line, separate from this
code checkout. That directory is the single source of truth: back it up or
move it to a new machine and everything (except the venv/install) comes
with it.

## Setup

Dependencies are declared in `pyproject.toml`. The base install is small and
pure-Python; everything to do with running an embedding model locally is
optional, so if you only use the `api` embedding backend (OpenAI, Ollama,
vLLM, …) no torch download happens at all.

```bash
# API-only (OpenAI-compatible embeddings, no local model)
uv sync
```

### Picking an install target

To run models on your own machine you also need `torch`, and there is no
single torch build that works everywhere — each accelerator has its own wheel
index. PaperSieve names one **install target** per accelerator, so you pick
your hardware instead of hand-assembling an index URL:

| Your hardware | Command |
|---|---|
| NVIDIA GPU, current driver | `uv sync --extra cu130` |
| NVIDIA GPU, older driver | `uv sync --extra cu128` |
| Apple Silicon (MPS) | `uv sync --extra mps` |
| AMD GPU (ROCm) | `uv sync --extra rocm` |
| Intel Arc / Core Ultra (XPU) | `uv sync --extra xpu` |
| No GPU, or unsure | `uv sync --extra cpu` |

Not sure which CUDA target? Run `nvidia-smi` — the version in the **top-right**
of its output is the newest CUDA your driver supports. 13.0 or newer takes
`cu130`; older drivers take `cu128`, which is capped at torch 2.11.

These targets are mutually exclusive — pick exactly one. They only work through
`uv sync` (see "Installing with pip instead" below for why).

> `uv sync` makes the environment match the lockfile exactly, so it will
> **uninstall** anything already in `.venv` that isn't part of the target you
> picked — including a torch you installed by hand earlier. Add `--inexact` to
> leave extraneous packages alone.

### Two ways to use a GPU

**A. The full torch build for your accelerator.** Simplest, and the right
default. Downloads ~3 GB.

```bash
uv sync --extra cu130
```

**B. CPU torch plus an ONNX Runtime or OpenVINO package.** `sentence-transformers`
always requires torch, so it can't be avoided — but it can be made small and
idle. The tiny CPU build (~200 MB) goes in, and a separate runtime does the
actual compute:

```bash
uv sync --extra cpu --extra onnx-gpu     # NVIDIA, via CUDAExecutionProvider
uv sync --extra cpu --extra openvino     # Intel CPU / iGPU / Arc
uv sync --extra cpu --extra onnx         # CPU only -- still ~2-3x faster than torch on CPU
```

Installing one of those does nothing on its own; the project (or the global
defaults) has to ask for it. In the UI that's **Embedding → Runtime + Device**;
in YAML:

```yaml
embedding:
  backend: local
  local:
    runtime: onnx      # torch | onnx | openvino
    device: cuda       # -> CUDAExecutionProvider
```

```yaml
# the Intel equivalent
embedding:
  backend: local
  local:
    runtime: openvino
    device: xpu        # -> OpenVINO "GPU"
```

**Choose B when** the download size matters, you're running on CPU (ONNX and
OpenVINO are both meaningfully faster there than torch), or you have an Intel
iGPU that `torch.xpu` doesn't cover. **Choose A when** you have an AMD GPU
(ONNX Runtime's ROCm package lags well behind torch's) or you'd rather not
have the first run pause to export the model to ONNX — that export is cached
afterwards, but some models fail to export at all.

### What each accelerator actually covers

- **CUDA** — any NVIDIA GPU the wheel's CUDA version supports.
- **ROCm** — AMD discrete GPUs on ROCm's support list (RX 7000/9000 series,
  Instinct). AMD **integrated** GPUs — Radeon 780M (`gfx1103`), Strix Halo
  (`gfx1151`) — are not officially supported but generally work by lying about
  the architecture: `HSA_OVERRIDE_GFX_VERSION=11.0.0 python app.py <user_dir>`.
- **XPU** — Intel Arc A/B-series discrete cards, Core Ultra processors with
  built-in Arc graphics (Meteor/Arrow/Lunar/Panther Lake), and Data Center GPU
  Max. Requires Intel's GPU driver. Older Intel iGPUs (Iris Xe and earlier) are
  **not** covered — use the `openvino` runtime for those.
- **MPS** — Apple Silicon. The plain PyPI torch wheel is already the MPS build.
- **Intel NPU** — there is no PyTorch backend for it, and `device: npu` under
  the `openvino` runtime is expected to fail on most embedding models:
  OpenVINO's NPU compiler needs static shapes, while sentence-transformer
  encoders tokenize to variable-length sequences. It's selectable so it works
  the day that changes; treat it as unsupported until then.

### Adding a torch + index combination that isn't listed

The targets above are just table entries. To add one — say CUDA 11.8 for an old
driver — edit four spots in `pyproject.toml`:

```toml
# 1. the index. Browse https://download.pytorch.org/whl/ for the names, and
#    https://download.pytorch.org/whl/cu118/torch/ for the versions it carries.
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

# 2. the extra
[project.optional-dependencies]
cu118 = ["papersieve[local]", "torch>=2.4,<2.8"]

# 3. route torch to the index for that extra
[tool.uv.sources]
torch = [..., { index = "pytorch-cu118", extra = "cu118" }]

# 4. add { extra = "cu118" } to the [tool.uv] conflicts list
```

Then `uv lock && uv sync --extra cu118`. If torch complains about a missing
`triton-*` package, add that to the same extra and route it to the same index —
newer torch builds require a matching one.

For a one-off machine you don't want to encode in the project at all, skip the
edits and install torch over the top:

```bash
uv sync --extra local --inexact          # sentence-transformers, no torch pin
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

### Installing with pip instead

pip has no way to attach an index to an extra, so `pip install ".[cu130]"`
would silently pull the default PyPI torch. `uv pip install` ignores
`[tool.uv.sources]` for the same pip-compatibility reason. With either, install
torch yourself first, then the `local` extra:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install ".[local]"    # sentence-transformers; torch already satisfied
```

### Running it

Pick (and create) a user directory — anywhere outside this checkout, e.g.
`~/papersieve-data`. This is where projects, `global_config.yaml`, and `.env`
will live:

```bash
mkdir -p ~/papersieve-data
cp .env.example ~/papersieve-data/.env   # only needed if you use the "api" embedding backend
uv run app.py ~/papersieve-data          # or: .venv/bin/python app.py ~/papersieve-data
```

Open `http://localhost:5000/`. `user_dir` is a required argument — the app
checks it exists and is writable before starting, and creates `projects/`
inside it automatically if missing.

By default this serves via a production-grade WSGI server ([waitress](https://docs.pylonsproject.org/projects/waitress/)),
suitable for leaving running. Pass `--debug` (or set `DEBUG=1`) instead for
Flask's dev server with auto-reload while you're changing code — see
"Deploying" below for the difference.

## Usage

1. On the homepage, fill in the "New project" form: pick a **source** and
   enter that source's venue ID (see "Sources" below). The form's hint shows
   the expected format for whichever source is selected.
2. You land on the project page. Click **Run**. A progress bar shows the
   fetch and embedding phases; when it's done the results appear
   automatically.
3. Browse results: tabs switch between queries, the tier sections (good /
   medium / low match) can be collapsed, and you can filter by title/author/
   keyword text or by a meta-score range. Click an abstract to expand it.
4. Re-running later only fetches/embeds what's new (see caching below).
   Use the checkboxes next to Run to force a full refetch or re-embed if you
   want to bust the cache.

## Sources

A project's `source` is picked when you create it and never changes
afterwards (make a second project to index the same venue elsewhere). It
decides which fields the project's config page shows.

### `openreview` — ICLR, NeurIPS, COLM, …

`venue_id` is the venue's OpenReview ID, the `id=` part of its URL:
`openreview.net/group?id=` **`ICLR.cc/2026/Conference`**.

Fetches everything under the venue's `Submission` invitation, so it sees
rejected/withdrawn papers too — `venue_filter` (below) narrows that down, and
defaults to accepted-only. Requires OpenReview credentials in `.env`.

### `cvf` — CVPR, ICCV, WACV

`venue_id` is the conference's path on the CVF Open Access site:
`openaccess.thecvf.com/` **`CVPR2025`**. The full list is at
[openaccess.thecvf.com/menu](https://openaccess.thecvf.com/menu) — e.g.
`CVPR2025`, `ICCV2025`, `WACV2024`, `CVPR2026_findings`. **2021 and later
only**: earlier venues use a different, day-split URL layout, and the run
fails with an explanatory error if you point it at one.

No credentials needed. CVF publishes accepted papers only, so
`venue_filter` and `submission_invitation` don't apply and are hidden on
CVF project pages.

CVF's listing page has no abstracts, and only about half of its entries link
to arXiv — so abstracts are read from each paper's own CVF detail page
instead, 8 at a time. That's the slow part of a first run: roughly two
minutes for a ~2700-paper venue like CVPR2024, and nothing after that, since
the result is cached like any other fetch.

### Adding another source

Each source is one `fetch_*(config, on_progress)` function returning the
shared paper dict, plus one `Source(...)` entry in `SOURCES`, both in
`sources.py`. Nothing outside that file needs to change: the forms, the
cache-invalidation key, and the config fields a project shows are all driven
off the `Source` struct.

## Config

There are two levels of config, both plain YAML files you can edit by hand
or through the Settings/project-config pages in the UI:

### `global_config.yaml` (`<user_dir>/global_config.yaml`)

Written out with the defaults below the first time you start the app, so
it always exists and always matches what the Settings page shows. Projects
inherit every value here unless they override it (see below). Covers:

- `embedding.backend`: `"local"` (runs a `sentence-transformers` model on
  your machine, no API key) or `"api"` (calls an OpenAI-compatible
  embeddings endpoint).
- `embedding.local.model`: the `sentence-transformers` model name.
- `embedding.local.runtime`: `"torch"` (default), `"onnx"` or `"openvino"` —
  which inference runtime loads the model. The latter two need their extra
  installed; see "Two ways to use a GPU" above.
- `embedding.local.device`: `"auto"` (default) detects one. Otherwise
  `cpu`, `cuda`, `mps` or `xpu` for the `torch` runtime; `cpu`, `xpu` or `npu`
  for `openvino`; those plus `rocm` for `onnx`. Two things surprise people:
  ROCm builds of torch identify as **`cuda`**, so AMD users pick `cuda` here
  (`rocm` is an ONNX Runtime provider name only), and an invalid
  runtime/device pair fails at the start of a run with a message listing what
  that runtime accepts.
- `embedding.local.batch_size`: texts encoded per forward pass. Leave empty
  (`null`) to size it from the device — 64 on CUDA/ROCm/XPU/MPS, 16 on CPU.
  Lower it if you run out of GPU memory.
- `embedding.api.base_url` / `.model`: the OpenAI-compatible endpoint and
  the model to request from it.
- `embedding.api.api_key_env`: which `.env` variable holds the key
  (default `OPENAI_API_KEY`). Local servers like Ollama or vLLM usually
  ignore it.
- `embedding.api.batch_size`: texts per request, default 96. Lower it if
  the provider rejects large batches.
- `tiers`: how papers are sorted into good/medium/low. `method: percentile`
  (recommended) ranks papers relative to each other for that query;
  `good_threshold`/`medium_threshold` are the cutoff percentiles (or raw
  cosine-similarity scores if `method: absolute`).

### `projects/<slug>/config.yaml` (`<user_dir>/projects/<slug>/config.yaml`, one per project)

Created automatically when you make a new project. Holds just that
project's specifics: `source`, `venue_id`, `max_papers` (safety cap),
`queries`, and — for OpenReview projects only — `submission_invitation`
(usually `"Submission"`, see below) and `venue_filter`. Anything it doesn't set (e.g. the embedding model) falls
back to `global_config.yaml`.

Everything above appears on the project's config page; rarely-touched
fields (`submission_invitation`, the batch sizes) sit in collapsed
"Advanced" blocks rather than being omitted. `source` and `venue_id` are
shown read-only, since the project folder is named after the venue ID —
index a different venue by creating another project.

The embedding and tiers sections live under **Inherited defaults**, with an
"override these for this project" checkbox. Left unticked, the project
follows `global_config.yaml` and picks up later edits to it; ticked, the
values are copied into the project's own `config.yaml` and stop tracking
the globals. Unticking again drops them and restores inheritance.

`submission_invitation` is the OpenReview invitation type submissions are
fetched under, and `"Submission"` is correct for essentially every venue —
`fetch_openreview()` already retries against older invitation names
(`Blind_Submission`) automatically if a venue turns out to need one, so
there's normally nothing to configure here.

#### `venue_filter` — picking a subset (OpenReview only)

This is what corresponds to the "Accepted (oral)" / "Accepted (poster)" tabs
on OpenReview. By default every submission under the venue's `Submission`
invitation is fetched — accepted, rejected, withdrawn, all of it. OpenReview doesn't
split those into separate invitations; instead, once decisions are out,
each paper gets a `venue` string (e.g. `"ICLR 2024 poster"`, `"ICLR 2024
spotlight"`, `"ICLR 2024 Conference Withdrawn Submission"`) and a
`venueid` (e.g. `"ICLR.cc/2024/Conference"` for accepted papers,
`".../Rejected_Submission"` for rejected ones).

`venue_filter` is a list of case-insensitive substrings checked against
both fields — a paper is kept if *any* term matches. Set it via the
project's config page, one term per line, e.g.:

```yaml
venue_filter:
  - "oral"
  - "poster"
  - "spotlight"
```

The special term `"accepted"` matches any accepted paper regardless of
oral/poster/spotlight, without needing to know the exact wording a venue
uses.

Defaults to `["accepted"]` — new projects only fetch accepted papers unless
you change this. Clear the field (save it empty) to fetch everything
instead. It only has any effect once the venue has released decisions —
before that every paper's `venue` field is blank, so nothing gets filtered
out regardless of what's set here.

Before fetching, PaperSieve checks the venue's own decision-outcome IDs
(e.g. `.../Rejected_Submission`, `.../Withdrawn_Submission`, and the bare
venue ID for accepted papers) and, if a filter term matches one of those,
requests only that subset from OpenReview server-side — so e.g.
`venue_filter: ["accepted"]` or `["reject"]` never downloads
rejected/withdrawn papers at all, rather than fetching everything and
discarding most of it locally. Finer splits like oral vs. poster vs.
spotlight aren't exposed as separate venue IDs by OpenReview, so those
terms still fall back to fetching the accepted set and filtering by the
`venue` text locally — still correct, just not server-side-filtered.

The paper cache (`cache/papers.json`) is tied to the `venue_filter` that
produced it: changing the filter and re-running refetches automatically
(no need to tick "force refresh"), since a different filter may need a
different, server-side-filtered subset. Changing anything else (queries,
embedding model, tiers) still reuses the cache as before.

### `.env` (`<user_dir>/.env`, not committed)

Holds credentials only, never shown or editable from the UI — just checked
for presence on the Settings page. Copy `.env.example` to `<user_dir>/.env`
and fill in:

- `OPENREVIEW_USERNAME` / `OPENREVIEW_PASSWORD` — **required for
  `openreview` projects** (CVF needs no credentials), any registered
  OpenReview account. OpenReview now blocks anonymous `/notes`
  requests with a bot-detection challenge, even for fully public venues, so
  fetching papers needs a logged-in session regardless of what you're
  indexing.
- `OPENAI_API_KEY` (or whatever `embedding.api.api_key_env` points at) —
  only needed if `embedding.backend: "api"`.
- `HF_TOKEN` — optional, only needed if `embedding.local.model` points at a
  gated or private Hugging Face model that requires an access token to
  download (most public `sentence-transformers` models don't).

### Why re-runs are cheap

Fetched papers are cached in `projects/<slug>/cache/papers.json`, alongside
a `cache/fetch_meta.json` recording what that fetch depended on (source,
venue, and for OpenReview the `venue_filter`) — if any of it changes, the
next run refetches by itself. Projects created before sources existed have
a `fetch_meta.json` in the older format; it's read as-is, so they keep
their cache.
Embeddings are cached in `projects/<slug>/cache/embeddings.npz`, keyed by a
hash of (embedding model, text) — so adding a new query or a few new papers
only embeds the new text, and switching embedding models naturally
invalidates the old cache entries without any manual cleanup.

## Deploying

`python app.py <user_dir>` runs [waitress](https://docs.pylonsproject.org/projects/waitress/)
by default — a production-ready, multithreaded, pure-Python WSGI server
(no compiled extensions, works the same on Linux/macOS/Windows). That's
already the "properly deployed" mode; there's no separate gunicorn/uwsgi
setup to wire up, since `app.py` picks the server itself based on
`--debug`.

Do **not** run with `--debug` (or `DEBUG=1`) except while actively changing
code on your own machine: it enables Flask's interactive Werkzeug
debugger, which lets anyone who can reach the port execute arbitrary Python.

Concretely, to leave this running:

- **Bind to localhost only** (the default, `--host 127.0.0.1`) unless you
  specifically need to reach it from another machine — PaperSieve has no
  authentication of its own, so anything it's bound to is fully accessible
  to whoever can reach that address. If you do need remote access, put it
  behind a reverse proxy (nginx/Caddy) with TLS and auth, rather than
  binding `--host 0.0.0.0` directly.
- **Keep it running** with a process manager rather than a background
  shell. On Linux, a systemd user service works well:

  ```ini
  # ~/.config/systemd/user/papersieve.service
  [Unit]
  Description=PaperSieve

  [Service]
  WorkingDirectory=/path/to/PaperSieve
  ExecStart=/path/to/PaperSieve/.venv/bin/python app.py /path/to/user_dir
  Restart=on-failure

  [Install]
  WantedBy=default.target
  ```

  ```bash
  systemctl --user daemon-reload
  systemctl --user enable --now papersieve
  ```
- **Pipeline runs are subprocesses of `app.py`**, so they stop if the
  service is stopped/restarted mid-run — restart the run from the project
  page afterward; the paper/embedding caches mean it won't recompute
  anything already done.
