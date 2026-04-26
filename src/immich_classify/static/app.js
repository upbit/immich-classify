// immich-classify WebUI — single-page interaction logic.
//
// Flow:
//   1. Fetch /api/tasks → populate the task <select>.
//   2. On task change → fetch /api/tasks/{id}/schema → render the filter form.
//   3. On form submit → fetch /api/tasks/{id}/results?filter=k=v → render grid.
//   4. Click a thumbnail → open modal with large image + field table + Immich link.
(() => {
  "use strict";

  const IMMICH_API_URL = (window.__IMMICH_API_URL__ || "").replace(/\/+$/, "");

  const $ = (id) => document.getElementById(id);
  const taskSelect = $("task-select");
  const taskMeta = $("task-meta");
  const schemaFieldsEl = $("schema-fields");
  const rawRowsEl = $("raw-filter-rows");
  const addRawBtn = $("add-raw-filter");
  const filterForm = $("filter-form");
  const resetBtn = $("reset-filters");
  const resultCountEl = $("result-count");
  const gridEl = $("grid");
  const emptyStateEl = $("empty-state");
  const modal = $("detail-modal");
  const modalImage = $("modal-image");
  const modalFieldsEl = $("modal-fields");
  const modalImmichLink = $("modal-immich-link");
  const syncBtn = $("sync-status-btn");
  const syncProgressEl = $("sync-progress");

  // Current loaded task schema: [{name, field_type, description, enum, default}].
  let currentSchema = [];
  // Most recent results, keyed by asset_id → fields object, for re-opening modal.
  const resultsByAsset = new Map();

  // ── Boot ────────────────────────────────────────────────────────
  init();

  async function init() {
    // Bind all event listeners BEFORE any async work so they are always
    // attached even if loadTasks() fails or the browser serves a cached page.
    console.log("[init] Binding event listeners");
    taskSelect.addEventListener("change", onTaskChanged);
    addRawBtn.addEventListener("click", () => addRawFilterRow());
    filterForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      loadResults();
    });
    resetBtn.addEventListener("click", () => {
      resetFilterForm();
      loadResults();
    });

    if (syncBtn) {
      console.log("[init] syncBtn found, binding click handler");
      syncBtn.addEventListener("click", onSyncStatus);
    } else {
      console.error("[init] syncBtn (#sync-status-btn) NOT FOUND in DOM!");
    }

    try {
      await loadTasks();
    } catch (err) {
      taskSelect.innerHTML = '<option value="">(failed to load tasks)</option>';
      console.error(err);
      return;
    }

    // Auto-load for the first task.
    if (taskSelect.value) {
      await onTaskChanged();
    }
  }

  // ── Tasks dropdown ──────────────────────────────────────────────
  async function loadTasks() {
    const resp = await fetch("/api/tasks");
    if (!resp.ok) throw new Error(`GET /api/tasks → ${resp.status}`);
    const data = await resp.json();
    const tasks = data.tasks || [];
    if (tasks.length === 0) {
      taskSelect.innerHTML = '<option value="">(no tasks)</option>';
      taskMeta.textContent = "";
      return;
    }
    taskSelect.innerHTML = "";
    for (const t of tasks) {
      const opt = document.createElement("option");
      opt.value = t.task_id;
      const idShort = t.task_id.slice(0, 8);
      const namePart = t.prompt_name ? ` · ${t.prompt_name}` : "";
      opt.textContent = `${idShort}… [${t.status}]${namePart} (${t.completed_count}/${t.total_count})`;
      opt.dataset.meta = JSON.stringify(t);
      taskSelect.appendChild(opt);
    }
    updateTaskMeta();
  }

  function updateTaskMeta() {
    const selectedOpt = taskSelect.selectedOptions[0];
    if (!selectedOpt || !selectedOpt.dataset.meta) {
      taskMeta.textContent = "";
      return;
    }
    const meta = JSON.parse(selectedOpt.dataset.meta);
    taskMeta.textContent =
      `${meta.task_id}  ·  created ${meta.created_at?.slice(0, 19) ?? ""}  ·  ${meta.completed_count}/${meta.total_count} done, ${meta.failed_count} failed`;
  }

  async function onTaskChanged() {
    updateTaskMeta();
    const taskId = taskSelect.value;
    if (!taskId) return;
    await loadSchema(taskId);
    await loadResults();
  }

  // ── Filter form ─────────────────────────────────────────────────
  async function loadSchema(taskId) {
    const resp = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/schema`);
    if (!resp.ok) {
      schemaFieldsEl.innerHTML = `<p class="meta">Failed to load schema (${resp.status})</p>`;
      currentSchema = [];
      return;
    }
    const data = await resp.json();
    currentSchema = data.fields || [];
    renderSchemaFields();
    rawRowsEl.innerHTML = "";
  }

  function renderSchemaFields() {
    schemaFieldsEl.innerHTML = "";
    if (currentSchema.length === 0) {
      schemaFieldsEl.innerHTML = '<p class="meta">(this task has no schema fields)</p>';
      return;
    }
    for (const f of currentSchema) {
      const wrap = document.createElement("div");
      wrap.className = "schema-field";

      const label = document.createElement("label");
      label.className = "schema-field__label";
      label.innerHTML = `${escapeHtml(f.name)}<code>${escapeHtml(f.field_type)}</code>`;
      wrap.appendChild(label);

      if (f.description) {
        const desc = document.createElement("div");
        desc.className = "schema-field__desc";
        desc.textContent = f.description;
        wrap.appendChild(desc);
      }

      const input = buildInputForField(f);
      input.dataset.fieldName = f.name;
      input.dataset.fieldType = f.field_type;
      input.classList.add("schema-field__input");
      label.setAttribute("for", `field-${f.name}`);
      input.id = `field-${f.name}`;
      wrap.appendChild(input);
      schemaFieldsEl.appendChild(wrap);
    }
  }

  function buildInputForField(f) {
    if (Array.isArray(f.enum) && f.enum.length > 0) {
      const sel = document.createElement("select");
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "— any —";
      sel.appendChild(blank);
      for (const opt of f.enum) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        sel.appendChild(o);
      }
      return sel;
    }
    if (f.field_type === "bool") {
      const sel = document.createElement("select");
      for (const [v, label] of [["", "— any —"], ["true", "true"], ["false", "false"]]) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = label;
        sel.appendChild(o);
      }
      return sel;
    }
    if (f.field_type === "int" || f.field_type === "float") {
      const input = document.createElement("input");
      input.type = "number";
      if (f.field_type === "float") input.step = "any";
      input.placeholder = "— any —";
      return input;
    }
    // Default: text input (covers string, list[string], and unknown types).
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "— any —";
    return input;
  }

  function resetFilterForm() {
    schemaFieldsEl.querySelectorAll(".schema-field__input").forEach((el) => {
      if (el.tagName === "SELECT") el.value = "";
      else el.value = "";
    });
    rawRowsEl.innerHTML = "";
  }

  function addRawFilterRow(key = "", value = "") {
    const row = document.createElement("div");
    row.className = "raw-filter-row";
    row.innerHTML = `
      <input type="text" name="key" placeholder="field" value="${escapeAttr(key)}">
      <span>=</span>
      <input type="text" name="value" placeholder="value" value="${escapeAttr(value)}">
      <button type="button" class="btn btn--small" aria-label="Remove">&times;</button>
    `;
    row.querySelector("button").addEventListener("click", () => row.remove());
    rawRowsEl.appendChild(row);
  }

  function collectFilters() {
    const filters = [];
    schemaFieldsEl.querySelectorAll(".schema-field__input").forEach((el) => {
      const key = el.dataset.fieldName;
      const raw = el.value;
      if (raw === "" || raw == null) return;
      filters.push(`${key}=${raw}`);
    });
    rawRowsEl.querySelectorAll(".raw-filter-row").forEach((row) => {
      const key = row.querySelector('input[name="key"]').value.trim();
      const value = row.querySelector('input[name="value"]').value.trim();
      if (!key) return;
      filters.push(`${key}=${value}`);
    });
    return filters;
  }

  // ── Results grid ────────────────────────────────────────────────
  async function loadResults() {
    const taskId = taskSelect.value;
    if (!taskId) return;
    const filters = collectFilters();
    const qs = new URLSearchParams();
    for (const f of filters) qs.append("filter", f);
    const url = `/api/tasks/${encodeURIComponent(taskId)}/results${qs.toString() ? "?" + qs.toString() : ""}`;

    gridEl.innerHTML = '<p class="meta">Loading…</p>';
    emptyStateEl.hidden = true;
    resultCountEl.textContent = "";

    let data;
    try {
      const resp = await fetch(url);
      if (!resp.ok) {
        let detail = resp.statusText;
        try {
          const j = await resp.json();
          if (j.detail) detail = j.detail;
        } catch {}
        gridEl.innerHTML = `<p class="meta">Query failed (${resp.status}): ${escapeHtml(detail)}</p>`;
        return;
      }
      data = await resp.json();
    } catch (err) {
      gridEl.innerHTML = `<p class="meta">Query failed: ${escapeHtml(err.message || String(err))}</p>`;
      return;
    }

    const results = data.results || [];
    resultsByAsset.clear();
    for (const r of results) resultsByAsset.set(r.asset_id, { fields: r.fields || {}, is_archived: r.is_archived, is_trashed: r.is_trashed });

    resultCountEl.textContent = `Matched ${results.length} / ${data.success_total || 0}`
      + (data.archived_total ? ` · ${data.archived_total} archived` : "")
      + (data.trashed_total ? ` · ${data.trashed_total} trashed (hidden)` : "");

    if (results.length === 0) {
      gridEl.innerHTML = "";
      emptyStateEl.hidden = false;
      return;
    }

    gridEl.innerHTML = "";
    const frag = document.createDocumentFragment();
    for (const r of results) {
      const card = document.createElement("div");
      card.className = "grid-card";
      if (r.is_archived) card.classList.add("is-archived");
      card.dataset.assetId = r.asset_id;
      card.title = r.asset_id;
      const img = document.createElement("img");
      img.loading = "lazy";
      img.className = "is-loading";
      img.alt = r.asset_id;

      const spinner = document.createElement("div");
      spinner.className = "grid-card__spinner";
      const errEl = document.createElement("div");
      errEl.className = "grid-card__error";
      errEl.textContent = "⚠ load failed";

      img.addEventListener("load", () => {
        img.classList.remove("is-loading");
        spinner.classList.add("is-hidden");
      });
      img.addEventListener("error", () => {
        spinner.classList.add("is-hidden");
        errEl.classList.add("is-visible");
      });

      img.src = `/thumbnail/${encodeURIComponent(r.asset_id)}`;
      card.appendChild(spinner);
      card.appendChild(errEl);
      card.appendChild(img);
      card.addEventListener("click", () => openModal(r.asset_id));
      frag.appendChild(card);
    }
    gridEl.appendChild(frag);
  }

  // ── Modal ───────────────────────────────────────────────────────
  function openModal(assetId) {
    const data = resultsByAsset.get(assetId) || {};
    const fields = data.fields || {};

    // Reset modal image loading state.
    modalImage.classList.add("is-loading");
    let modalSpinner = modal.querySelector(".modal__spinner");
    if (!modalSpinner) {
      modalSpinner = document.createElement("div");
      modalSpinner.className = "modal__spinner";
      modalImage.parentElement.appendChild(modalSpinner);
    }
    modalSpinner.classList.remove("is-hidden");

    modalImage.onload = () => {
      modalImage.classList.remove("is-loading");
      modalSpinner.classList.add("is-hidden");
    };
    modalImage.onerror = () => {
      modalSpinner.classList.add("is-hidden");
      modalImage.classList.remove("is-loading");
    };

    modalImage.src = `/thumbnail/${encodeURIComponent(assetId)}`;
    modalImage.alt = assetId;
    modalFieldsEl.innerHTML = "";
    // Always show asset_id first.
    appendDt(modalFieldsEl, "asset_id", assetId);
    if (data.is_archived) appendDt(modalFieldsEl, "status", "📦 Archived");
    for (const [k, v] of Object.entries(fields)) {
      appendDt(modalFieldsEl, k, v);
    }
    const immichUrl = IMMICH_API_URL ? `${IMMICH_API_URL}/photos/${encodeURIComponent(assetId)}` : "#";
    modalImmichLink.href = immichUrl;
    if (typeof modal.showModal === "function") {
      modal.showModal();
    } else {
      modal.setAttribute("open", "");
    }
  }

  function appendDt(dl, key, value) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    dl.appendChild(dt);
    const dd = document.createElement("dd");
    if (Array.isArray(value)) {
      dd.className = "is-list";
      for (const item of value) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = String(item);
        dd.appendChild(tag);
      }
    } else if (value === true || value === false) {
      dd.textContent = String(value);
      dd.className = value ? "is-bool-true" : "is-bool-false";
    } else if (value && typeof value === "object") {
      dd.textContent = JSON.stringify(value);
    } else {
      dd.textContent = value == null ? "" : String(value);
    }
    dl.appendChild(dd);
  }

  // ── Sync status ─────────────────────────────────────────────────
  async function onSyncStatus() {
    const taskId = taskSelect.value;
    if (!taskId) {
      console.warn("[sync] No task selected");
      return;
    }
    console.log("[sync] Starting sync for task:", taskId);
    syncBtn.classList.add("is-syncing");
    syncBtn.disabled = true;
    syncProgressEl.textContent = "Starting sync…";

    try {
      const url = `/api/tasks/${encodeURIComponent(taskId)}/sync-status`;
      console.log("[sync] POST", url);
      const resp = await fetch(url, { method: "POST" });
      const body = await resp.json();
      console.log("[sync] POST response:", resp.status, body);
      if (!resp.ok) {
        syncProgressEl.textContent = `Sync failed (${resp.status}): ${body.detail || ""}`;
        syncBtn.classList.remove("is-syncing");
        syncBtn.disabled = false;
        return;
      }
      // Poll progress
      pollSyncProgress(taskId);
    } catch (err) {
      console.error("[sync] Error:", err);
      syncProgressEl.textContent = `Sync error: ${err.message}`;
      syncBtn.classList.remove("is-syncing");
      syncBtn.disabled = false;
    }
  }

  async function pollSyncProgress(taskId) {
    const poll = async () => {
      try {
        const resp = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/sync-status`);
        if (!resp.ok) {
          console.warn("[sync] Poll failed:", resp.status);
          return;
        }
        const data = await resp.json();
        console.log("[sync] Poll:", data);
        syncProgressEl.textContent = `Syncing ${data.completed || 0}/${data.total || 0}` +
          (data.updated ? ` (${data.updated} changed)` : "") +
          (data.errors ? ` · ${data.errors} errors` : "");
        if (data.running) {
          setTimeout(poll, 500);
        } else {
          syncProgressEl.textContent = `Sync done: ${data.updated || 0} changed` +
            (data.errors ? `, ${data.errors} errors` : "");
          syncBtn.classList.remove("is-syncing");
          syncBtn.disabled = false;
          console.log("[sync] Done, reloading results");
          // Reload results to reflect updated flags
          await loadResults();
        }
      } catch (err) {
        console.error("[sync] Poll error:", err);
        syncBtn.classList.remove("is-syncing");
        syncBtn.disabled = false;
      }
    };
    setTimeout(poll, 300);
  }

  // ── Helpers ─────────────────────────────────────────────────────
  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/'/g, "&#39;");
  }
})();
