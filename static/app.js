const PaperSieve = (function () {
  const TIER_LABELS = { good: "Good match", medium: "Medium match", low: "Low match" };
  const TIER_ORDER = ["good", "medium", "low"];

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function debounce(fn, wait) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function initProjectPage({ slug, initialStatus, showResults }) {
    const configSection = document.getElementById("configSection");
    const configForm = document.getElementById("configForm");
    const toggleConfig = document.getElementById("toggleConfig");
    const backendSelect = document.getElementById("backendSelect");
    const runButton = document.getElementById("runButton");
    const stopButton = document.getElementById("stopButton");
    const progressView = document.getElementById("progressView");
    const progressPhase = document.getElementById("progressPhase");
    const progressBar = document.getElementById("progressBar");
    const progressCount = document.getElementById("progressCount");
    const progressError = document.getElementById("progressError");
    const progressLog = document.getElementById("progressLog");
    const resultsView = document.getElementById("resultsView");
    const noResults = document.getElementById("noResults");

    let DATA = null;
    let activeQuery = null;
    let filterText = "";
    let metaMin = 0;
    let metaMax = 100;
    let pollTimer = null;
    const debouncedRenderTiers = debounce(() => renderTiers(), 800);

    function setProgress(status) {
      progressPhase.textContent = status.phase ? `${status.phase}…` : "starting…";
      const current = status.current || 0;
      const total = typeof status.total === "number" && status.total > 0 ? status.total : null;

      if (total) {
        // determinate: known total (paper count from OpenReview, or number
        // of texts left to embed)
        progressBar.max = total;
        progressBar.value = current;
        progressCount.textContent = `${current} / ${total}`;
      } else {
        // indeterminate: total isn't known yet (or couldn't be determined) --
        // show a spinning bar with just the running count, same component
        // for both the fetch and embed phases.
        progressBar.removeAttribute("value");
        progressBar.removeAttribute("max");
        progressCount.textContent = current ? `${current} so far…` : "";
      }

      if (status.state === "error") {
        progressError.hidden = false;
        progressError.textContent = status.error || "Unknown error";
      } else {
        progressError.hidden = true;
      }

      if (status.log) {
        progressLog.textContent = status.log;
        progressLog.scrollTop = progressLog.scrollHeight;
      }
    }

    function startPolling() {
      // poll() first and unconditionally: if anything below throws, polling
      // has already started, so status updates keep flowing instead of
      // silently never starting (this is what caused the "stuck at
      // starting forever" bug -- an exception here used to abort before
      // poll() was ever reached).
      poll();
      progressView.hidden = false;
      resultsView.hidden = true;
      noResults.hidden = true;
      configSection.hidden = true;
      runButton.disabled = true;
      stopButton.hidden = false;
    }

    function poll() {
      fetch(`/project/${slug}/status`)
        .then((r) => r.json())
        .then((status) => {
          setProgress(status);
          if (status.state === "running") {
            pollTimer = setTimeout(poll, 1000);
          } else {
            runButton.disabled = false;
            stopButton.hidden = true;
            if (status.state === "done") {
              progressView.hidden = true;
              loadResults();
            }
          }
        })
        .catch((err) => {
          console.error("status poll failed, retrying", err);
          pollTimer = setTimeout(poll, 2000);
        });
    }

    stopButton.addEventListener("click", () => {
      fetch(`/project/${slug}/stop`, { method: "POST" }).then(poll);
    });

    function loadResults() {
      fetch(`/project/${slug}/data`)
        .then((r) => r.json())
        .then((data) => {
          DATA = data;
          activeQuery = DATA.queries[0];
          resultsView.hidden = false;
          noResults.hidden = true;
          document.getElementById("metaLine").textContent =
            `${DATA.venue_id} · ${DATA.paper_count} papers · model: ${DATA.embedding_model} · generated ${DATA.generated_at}`;
          renderTabs();
          renderTiers();
        })
        .catch(() => {
          noResults.hidden = false;
        });
    }

    function renderTabs() {
      const tabsEl = document.getElementById("tabs");
      tabsEl.innerHTML = "";
      DATA.queries.forEach((q) => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = "tab" + (q === activeQuery ? " active" : "");
        tab.textContent = q.length > 30 ? q.slice(0, 28) + "…" : q;
        tab.title = q;
        tab.addEventListener("click", () => {
          activeQuery = q;
          renderTabs();
          renderTiers();
        });
        tabsEl.appendChild(tab);
      });
    }

    function matchesFilter(paper, ranking) {
      if (ranking.meta_score < metaMin || ranking.meta_score > metaMax) return false;
      if (!filterText) return true;
      const hay = [paper.title, (paper.authors || []).join(" "), (paper.keywords || []).join(" ")]
        .join(" ").toLowerCase();
      return hay.includes(filterText);
    }

    function renderTiers() {
      const container = document.getElementById("tiersContainer");
      container.innerHTML = "";

      const grouped = { good: [], medium: [], low: [] };
      DATA.papers.forEach((p) => {
        const r = p.rankings[activeQuery];
        if (!r) return;
        if (!matchesFilter(p, r)) return;
        grouped[r.tier].push({ paper: p, ranking: r });
      });
      TIER_ORDER.forEach((t) => grouped[t].sort((a, b) => b.ranking.score - a.ranking.score));

      TIER_ORDER.forEach((tier) => {
        const items = grouped[tier];
        const block = document.createElement("div");
        block.className = "tier-block";

        const header = document.createElement("div");
        header.className = `tier-header ${tier}`;
        header.innerHTML = `
          <span class="tier-title">${TIER_LABELS[tier]}</span>
          <span class="tier-count">${items.length}</span>
          <span class="tier-caret">▾</span>
        `;

        const list = document.createElement("div");
        list.className = "tier-list";
        if (items.length === 0) {
          list.innerHTML = '<div class="empty">No papers in this tier for the current filter.</div>';
        } else {
          items.forEach((item) => {
            const p = item.paper, r = item.ranking;
            const row = document.createElement("div");
            row.className = "result-row";
            row.innerHTML = `
              <div class="rank-num">#${r.rank}</div>
              <div class="result-body">
                <a class="result-title" href="${p.forum_url}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a>
                <div class="result-meta">
                  <span class="score-chip ${tier}">${r.score.toFixed(3)} · p${r.meta_score.toFixed(0)}</span>${escapeHtml((p.authors || []).slice(0, 4).join(", "))}${(p.authors || []).length > 4 ? ", et al." : ""}
                </div>
                ${p.tldr ? `<p class="result-tldr">${escapeHtml(p.tldr)}</p>` : ""}
                ${p.abstract ? `<p class="result-abstract">${escapeHtml(p.abstract)}</p>` : ""}
              </div>
            `;
            const abstractEl = row.querySelector(".result-abstract");
            if (abstractEl) {
              abstractEl.addEventListener("click", () => abstractEl.classList.toggle("expanded"));
            }
            list.appendChild(row);
          });
        }

        header.addEventListener("click", () => list.classList.toggle("collapsed"));
        block.appendChild(header);
        block.appendChild(list);
        container.appendChild(block);
      });
    }

    configForm.addEventListener("submit", (e) => e.preventDefault());

    function updateBackendFields() {
      const backend = backendSelect.value;
      document.querySelectorAll("[data-backend]").forEach((el) => {
        el.hidden = el.dataset.backend !== backend;
      });
    }
    backendSelect.addEventListener("change", updateBackendFields);
    updateBackendFields();

    toggleConfig.addEventListener("click", (e) => {
      e.preventDefault();
      configSection.hidden = !configSection.hidden;
    });

    runButton.addEventListener("click", () => {
      const formData = new FormData(configForm);
      fetch(`/project/${slug}/config`, { method: "POST", body: formData })
        .then(() => fetch(`/project/${slug}/run`, { method: "POST", body: formData }))
        .then(startPolling)
        .catch((err) => console.error("failed to start run", err));
    });

    document.getElementById("filterInput").addEventListener("input", (e) => {
      filterText = e.target.value.trim().toLowerCase();
      if (DATA) debouncedRenderTiers();
    });
    document.getElementById("metaMin").addEventListener("input", (e) => {
      metaMin = Number(e.target.value) || 0;
      if (DATA) debouncedRenderTiers();
    });
    document.getElementById("metaMax").addEventListener("input", (e) => {
      metaMax = Number(e.target.value) || 100;
      if (DATA) debouncedRenderTiers();
    });

    if (initialStatus.state === "running") {
      startPolling();
    } else if (showResults) {
      loadResults();
    }
  }

  return { initProjectPage };
})();
