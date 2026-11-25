"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  samples: { spam: [], legitimate: [] },
  sampleIndex: { spam: 0, legitimate: 0 },
  last: null,
  threshold: 0.5,
  batchUrl: null,
};

const EMPTY_HTML = `<svg class="empty-icon" viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
  <path d="M3 6.5h18v11H3z"/><path d="M3.5 7l8.5 6 8.5-6"/></svg>
  <p>Paste an email and press <strong>Analyze</strong> to score it.</p>`;

/* ------------------------------------------------------------------ tabs */
function activateTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "model") loadMetrics();
}

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

/* --------------------------------------------------------------- health */
async function health() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    $("#model-pill").textContent = j.ok ? "model ready" : "model unavailable";
    $("#model-pill").style.color = j.ok ? "var(--ok)" : "var(--bad)";
  } catch {
    $("#model-pill").textContent = "model unavailable";
  }
}

/* ------------------------------------------------------------- samples */
async function loadSamples() {
  try {
    const r = await fetch("/api/examples");
    const j = await r.json();
    state.samples = { spam: j.spam || [], legitimate: j.legitimate || [] };
  } catch { /* ignore */ }
}

$$("[data-sample]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const kind = btn.dataset.sample;
    const list = state.samples[kind] || [];
    if (!list.length) return;
    const item = list[state.sampleIndex[kind] % list.length];
    state.sampleIndex[kind]++;
    $("#subject").value = item.subject;
    $("#body").value = item.body;
    analyze();
  });
});

/* -------------------------------------------------------------- predict */
function renderTerms(explanation, label) {
  const wanted = label === "spam" ? "spam" : "legitimate";
  const terms = explanation.filter((t) => t.direction === wanted).slice(0, 9);
  $("#terms").innerHTML = terms.length
    ? terms.map((t) => `<span class="term ${t.direction}" title="contribution ${t.weight}">${esc(t.term)}</span>`).join("")
    : '<span class="muted small">No strong terms in the vocabulary for this message.</span>';
}

function applyThreshold() {
  if (!state.last) return;
  const p = state.last.spam_probability;
  const isSpam = p >= state.threshold;
  const label = isSpam ? "spam" : "legitimate";
  const badge = $("#label-badge");
  badge.textContent = label;
  badge.className = "badge " + label;
  $("#meter-mid").style.left = (state.threshold * 100).toFixed(1) + "%";
  renderTerms(state.last.explanation, label);
}

function renderVerdict(j) {
  state.last = j;
  state.threshold = j.threshold ?? 0.5;
  $("#threshold").value = state.threshold;
  $("#threshold-val").textContent = state.threshold.toFixed(4);
  $("#verdict-empty").hidden = true;
  $("#verdict-result").hidden = false;
  $("#spam-prob").textContent = Math.round(j.spam_probability * 1000) / 10;
  $("#meter-fill").style.width = (j.spam_probability * 100).toFixed(1) + "%";
  $("#latency").textContent = `${j.latency_ms} ms · ${j.terms_in_text} known terms`;
  $("#term-count").textContent =
    j.explanation.length ? "Weights are the model's logistic-regression coefficients." : "";
  applyThreshold();
}

async function analyze() {
  const subject = $("#subject").value;
  const body = $("#body").value;
  if (!subject.trim() && !body.trim()) return;
  const btn = $("#analyze");
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, body }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || "request failed");
    renderVerdict(await r.json());
  } catch (e) {
    $("#verdict-empty").hidden = false;
    $("#verdict-result").hidden = true;
    $("#verdict-empty").innerHTML = `<div class="empty-icon">⚠</div><p>${esc(e.message)}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze email";
  }
}

$("#analyze").addEventListener("click", analyze);
$("#clear").addEventListener("click", () => {
  $("#subject").value = "";
  $("#body").value = "";
  state.last = null;
  $("#verdict-result").hidden = true;
  $("#verdict-empty").hidden = false;
  $("#verdict-empty").innerHTML = EMPTY_HTML;
});

$("#threshold").addEventListener("input", (e) => {
  state.threshold = parseFloat(e.target.value);
  $("#threshold-val").textContent = state.threshold.toFixed(2);
  applyThreshold();
});

/* ---------------------------------------------------------------- batch */
const drop = $("#drop");
const fileInput = $("#file");

function setFile(f) {
  $("#file-name").textContent = f ? `${f.name} · ${(f.size / 1024).toFixed(1)} KB` : "no file selected";
  $("#run-batch").disabled = !f;
  if (state.batchUrl) { URL.revokeObjectURL(state.batchUrl); state.batchUrl = null; }
  $("#download").classList.add("hidden");
  $("#batch-result").classList.add("hidden");
  fileInput._file = f || null;
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })
);
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) { fileInput._file = f; setFile(f); }
});

$("#run-batch").addEventListener("click", async () => {
  const f = fileInput._file || fileInput.files[0];
  if (!f) return;
  const btn = $("#run-batch");
  btn.disabled = true;
  btn.textContent = "Scoring…";
  try {
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("/api/batch", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || "request failed");
    const blob = await r.blob();
    const rows = parseInt(r.headers.get("X-Rows") || "0", 10);
    const spam = parseInt(r.headers.get("X-Spam") || "0", 10);
    $("#b-rows").textContent = rows;
    $("#b-spam").textContent = spam;
    $("#b-legit").textContent = rows - spam;
    $("#batch-result").classList.remove("hidden");
    if (state.batchUrl) URL.revokeObjectURL(state.batchUrl);
    state.batchUrl = URL.createObjectURL(blob);
    const a = $("#download");
    a.href = state.batchUrl;
    a.download = f.name.replace(/\.csv$/i, "") + "_predictions.csv";
    a.classList.remove("hidden");
  } catch (e) {
    alert(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Score file";
  }
});

/* ---------------------------------------------------------------- model */
const pct = (v) => (v * 100).toFixed(2) + "%";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function metricCard(label, value, sub) {
  return `<div class="metric"><span>${label}</span><b>${value}</b><small>${sub || ""}</small></div>`;
}

function shortName(entry) {
  return entry === 386 ? "Count + LR" : entry === 387 ? "TF-IDF + LR" : "entry " + entry;
}

async function loadMetrics() {
  const body = $("#model-body");
  try {
    const r = await fetch("/api/metrics");
    if (!r.ok) throw new Error("metrics unavailable");
    const { sweep, terms } = await r.json();
    const best = sweep.best;
    const bestRun = sweep.runs.find((r) => r.run_id === best.run_id) || {};
    const t = best.test;
    const cm = t.confusion_matrix;

    $("#footer-model").textContent = `${shortName(best.config.entry)} (#${best.config.entry})`;

    const header = `
      <div class="metrics-grid">
        ${metricCard("Test accuracy", pct(t.accuracy), `${240} held-out emails`)}
        ${metricCard("Macro F1", pct(t.f1_macro), "balanced across classes")}
        ${metricCard("ROC-AUC", t.roc_auc.toFixed(4), "ranking quality")}
        ${metricCard("PR-AUC", t.pr_auc.toFixed(4), "precision vs recall")}
        ${metricCard("Train time", (bestRun.train_seconds ?? "—") + "s", "Atlas Engine, CPU")}
      </div>`;

    const cmHtml = `
      <div class="card">
        <div class="card-head"><h2>Exploratory synthetic test</h2><span class="pill">240 rows</span></div>
        <div class="cm">
          <div></div><div class="h">pred. legitimate</div><div class="h">pred. spam</div>
          <div class="rh">actual legit.</div>
          <div class="cell good"><b>${cm[0][0]}</b><span>correct</span></div>
          <div class="cell bad"><b>${cm[0][1]}</b><span>false alarm</span></div>
          <div class="rh">actual spam</div>
          <div class="cell bad"><b>${cm[1][0]}</b><span>missed</span></div>
          <div class="cell good"><b>${cm[1][1]}</b><span>caught</span></div>
        </div>
        <div class="explain">
          <h3>Strongest spam terms</h3>
          <div class="term-list">${(terms.spam || []).slice(0, 12).map(x => `<span class="term spam">${esc(x.term)}</span>`).join("")}</div>
        </div>
        <div class="explain">
          <h3>Strongest legitimate terms</h3>
          <div class="term-list">${(terms.legitimate || []).slice(0, 12).map(x => `<span class="term legitimate">${esc(x.term)}</span>`).join("")}</div>
        </div>
      </div>`;

    const rows = sweep.runs.map((r) => {
      const c = r.config;
      const isBest = r.run_id === best.run_id;
      return `<tr class="${isBest ? "best" : ""}">
        <td>${shortName(c.entry)}${isBest ? " ★" : ""}</td>
        <td class="num">${c.ngram}</td>
        <td class="num">${c.C}</td>
        <td class="num">${(c.max_features / 1000)}k</td>
        <td class="num">${pct(r.val.f1_macro)}</td>
        <td class="num">${pct(r.val.accuracy)}</td>
        <td class="num">${r.val.roc_auc.toFixed(3)}</td>
        <td class="num">${r.train_seconds}s</td>
      </tr>`;
    }).join("");

    body.innerHTML = header + `
      <div class="model-grid">
        ${cmHtml}
        <div class="card">
          <div class="card-head"><h2>Validation-selected sweep</h2><span class="pill">${sweep.runs.length} runs · threshold ${(best.threshold ?? 0.5).toFixed(4)}</span></div>
          <table class="sweep">
            <thead><tr><th>model</th><th>ngram</th><th>C</th><th>vocab</th><th>val F1</th><th>val acc</th><th>ROC</th><th>time</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
          <div class="note">
            <b>About the score.</b> ${esc(sweep.dataset.note)}
            Model and threshold were selected on validation accuracy. Independent real-email
            evaluation is still needed. Dataset and metrics: <code>reports/sweep.json</code>.
          </div>
        </div>
      </div>`;
  } catch (e) {
    body.innerHTML = `<div class="card"><p class="muted">${esc(e.message)}</p></div>`;
  }
}

/* ------------------------------------------------------------------ init */
(async () => {
  health();
  await loadSamples();
  loadMetrics();

  const params = new URLSearchParams(location.search);
  if (params.get("tab")) activateTab(params.get("tab"));
  const demo = params.get("demo");
  if (demo && (state.samples[demo] || []).length) {
    const item = state.samples[demo][0];
    $("#subject").value = item.subject;
    $("#body").value = item.body;
    analyze();
  }
})();
