// DOM helpers and model definitions
// ---------------------------------
const $ = (id) => document.getElementById(id);
const modelInfo = [
  {
    id: "naive_bayes",
    name: "Naive Bayes",
    type: "THE PROBABILISTIC APPROACH",
    color: "green",
  },
  {
    id: "logistic_regression",
    name: "Logistic regression",
    type: "THE LINEAR APPROACH",
    color: "blue",
  },
  {
    id: "neural_network",
    name: "Neural network",
    type: "THE NONLINEAR APPROACH",
    color: "purple",
  },
];
// Page state
// ----------
let hasResults = false,
  busy = false,
  submittedEmail = "",
  submittedSubject = "",
  resultVerdict = "neutral";
// Navigation and input controls
// -----------------------------
function route() {
  const r = location.hash === "#results" && hasResults;
  const about = location.hash === "#about";
  $("input-page").hidden = r || about;
  $("about-page").hidden = !about;
  $("results-page").hidden = !r;
  document.body.dataset.page = about ? "about" : r ? "results" : "input";
  document.body.dataset.verdict = r ? resultVerdict : "neutral";
  if ($("project-info")) $("project-info").hidden = r || about;
  for (const [id, active] of [
    ["step1", !r && !about],
    ["step2", r],
  ]) {
    const step = $(id);
    step.classList.toggle("is-active", active);
    if (active) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");
  }
  window.scrollTo(0, 0);
  if (r) {
    $("results-page").focus({ preventScroll: true });
    requestAnimationFrame(fitComparisonPanels);
  }
}
function count() {
  $("count").textContent = $("body").value.length.toLocaleString() + " characters";
  $("error").textContent = "";
}
$("body").addEventListener("input", count);
function reset() {
  $("form").reset();
  count();
  hasResults = false;
}
$("clear").onclick = () => {
  if (!busy) {
    reset();
    $("body").focus();
  }
};
function sample(spam) {
  if (busy) return;
  $("subject").value = spam
    ? "Congratulations! Claim your cash prize"
    : "Project meeting on Thursday";
  $("body").value = spam
    ? "You are our lucky winner! Claim your free cash prize today. Click the link for a guaranteed reward. This limited offer ends tonight."
    : "Hi team,\n\nCan we meet on Thursday at 2pm to review the project report? Please bring your evaluation results and updated slides.\n\nThanks,\nAlex";
  count();
  $("body").focus();
}
$("sample-spam").onclick = () => sample(true);
$("sample-ham").onclick = () => sample(false);
// Prediction cards and model agreement
// ------------------------------------
function show(results) {
  $("cards").replaceChildren();
  for (const [i, m] of modelInfo.entries()) {
    const p = results.find((r) => r.id === m.id) || {
      status: "error",
      message: "No result returned.",
    };
    const ready = p.status === "ready";
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="card-top"><b class="number ${m.color}">0${i + 1}</b><h2>${m.name}</h2><small>${m.type}</small></div><div class="card-body"><div class="prediction"></div><p class="caption"></p><div class="explanation"><small>MODEL STATUS</small><p></p></div></div>`;
    const label = card.querySelector(".prediction");
    label.textContent = ready
      ? p.label === 1
        ? "SPAM"
        : "LEGITIMATE"
      : p.status === "error"
        ? "Prediction failed"
        : "Not connected";
    if (ready) {
      label.classList.add(p.label === 1 ? "spam" : "ham");
      card.dataset.prediction = p.label === 1 ? "spam" : "legitimate";
    }
    card.querySelector(".caption").textContent = ready
      ? `Prediction time: ${p.milliseconds} ms`
      : "No prediction generated";
    card.querySelector(".explanation").remove();
    if (p.out_of_vocabulary)
      node(
        "p",
        "No recognised words: limited evidence.",
        card.querySelector(".card-body"),
        "fine-print",
      );
    addScores(card, p);
    const body = card.querySelector(".card-body");
    const predictionPanel = node(
      "section",
      undefined,
      body,
      "comparison-panel prediction-panel",
    );
    predictionPanel.dataset.view = "prediction";
    for (const child of [...body.children]) {
      if (child !== predictionPanel && !child.classList.contains("performance-block"))
        predictionPanel.append(child);
    }
    const performance = body.querySelector(".performance-block");
    if (performance) {
      performance.classList.add("comparison-panel");
      performance.dataset.view = "results";
    }
    const matrixPanel = node("section", undefined, body, "comparison-panel matrix-panel");
    matrixPanel.dataset.view = "matrix";
    node(
      "p",
      ready ? "Loading saved matrix…" : "No matrix available for this model.",
      matrixPanel,
      "fine-print",
    );
    $("cards").append(card);
  }
  setComparisonView(0);
  const ready = results.filter((p) => p.status === "ready");
  const n = ready.filter((p) => p.label === 1).length;
  resultVerdict =
    !ready.length || n * 2 === ready.length
      ? "neutral"
      : n * 2 > ready.length
        ? "spam"
        : "legitimate";
  document.body.dataset.verdict = resultVerdict;
  $("agreement").textContent =
    ready.length < 2
      ? `Agreement unavailable: ${ready.length} model responded`
      : n * 2 === ready.length
        ? `${ready.length} available models disagree`
        : `${Math.max(n, ready.length - n)} of ${ready.length} available models predict ${n > ready.length / 2 ? "spam" : "legitimate"}`;
}
// Prediction request and form events
// ----------------------------------
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  if (!$("body").value.trim()) {
    $("error").textContent = "Please paste email content.";
    return;
  }
  const subject = $("subject").value,
    body = $("body").value;
  busy = true;
  const button = $("form").querySelector("[type=submit]");
  button.disabled = true;
  button.textContent = "Analysing saved models…";
  $("error").textContent = "";
  try {
    if (location.protocol === "file:")
      throw new Error(
        "Start the local application with python app/app.py, then open http://127.0.0.1:8000.",
      );
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);
    let response;
    try {
      response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject, body }),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeout);
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    if (!Array.isArray(data.results)) throw new Error("Invalid prediction response.");
    submittedSubject = subject.trim();
    submittedEmail = (subject.trim() ? subject + "\n\n" : "") + body;
    show(data.results);
    hasResults = true;
    location.hash = "#results";
    route();
  } catch (error) {
    $("error").textContent =
      error.name === "AbortError"
        ? "Prediction timed out. Please try again."
        : error.message;
  } finally {
    busy = false;
    button.disabled = false;
    button.innerHTML = "Analyse email <span>↗</span>";
  }
});
$("edit").onclick = () => {
  location.hash = "#input";
  route();
  $("body").focus();
};
$("another").onclick = () => {
  reset();
  location.hash = "#input";
  route();
  $("body").focus();
};
window.addEventListener("hashchange", route);
route();
if (location.protocol !== "file:")
  fetch("/api/status")
    .then((r) => r.json())
    .then((data) => {
      const n = data.models.filter((m) => m.status === "ready").length;
      document.querySelector(".footnote").textContent = `${n} of 3 models ready`;
    })
    .catch(() => {
      $("error").textContent =
        "Cannot reach the prediction server. Start python app/app.py.";
    });

// Reusable DOM builders and saved evidence
// ----------------------------------------
let evidenceRequest;
function node(tag, text, parent, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  if (parent) parent.append(e);
  return e;
}
function accordion(parent, title) {
  const d = node("details", undefined, parent, "card-detail");
  node("summary", title, d);
  return node("div", undefined, d, "detail-content");
}
function getEvidence() {
  if (!evidenceRequest)
    evidenceRequest = fetch("/api/experiments")
      .then((r) => {
        if (!r.ok) throw new Error("Saved metrics unavailable.");
        return r.json();
      })
      .catch((e) => {
        evidenceRequest = null;
        throw e;
      });
  return evidenceRequest;
}
// Model scores and performance
// ----------------------------
function addScores(card, result) {
  const body = card.querySelector(".card-body");
  const p = result.spam_probability;
  const score = node("section", undefined, body, "score-area");
  node("span", "SPAM SCORE", score, "section-label");
  if (result.status !== "ready" || typeof p !== "number") {
    node("p", "Model not connected.", score, "fine-print");
    return;
  }
  const value = node("strong", (p * 100).toFixed(2) + "%", score, "score-value");
  const scoreHue = 120 * (1 - Math.max(0, Math.min(1, p)));
  const scoreColour = `hsl(${scoreHue.toFixed(2)} 85% 65%)`;
  value.style.color = scoreColour;
  value.style.textShadow = `0 0 20px ${scoreColour.replace("hsl(", "hsl(").replace(")", " / 0.15)")}`;
  const track = node("div", undefined, score, "prob-track");
  track.setAttribute("role", "img");
  track.setAttribute("aria-label", `Spam score ${(p * 100).toFixed(2)} percent`);
  const fill = node("div", undefined, track);
  fill.style.width = p * 100 + "%";
  fill.style.background = scoreColour;
  const scale = node("div", undefined, score, "score-colour-scale");
  ["0%", "50%", "100%"].forEach((label) => node("span", label, scale));
  node("p", "Model estimate · not calibrated confidence.", score, "fine-print");
  const explanation = result.explanation;
  const threshold =
    explanation?.threshold ?? (result.id === "neural_network" ? 0.7 : 0.5);
  const border = node("div", undefined, score, "borderline-status");
  const distance = Math.abs(p - threshold);
  node(
    "span",
    distance <= 0.05 ? "BORDERLINE" : "CLEAR OF BOUNDARY",
    border,
    distance <= 0.05 ? "border-badge" : "clear-badge",
  );
  node(
    "small",
    `Boundary ${(threshold * 100).toFixed(0)}% · ${(distance * 100).toFixed(2)} pp away`,
    border,
  );
  const area = node("section", undefined, body, "sentence-section");
  node("h3", "Your email, explained", area);
  if (explanation?.status === "ready") {
    const legend = node("div", undefined, area, "sentence-legend");
    node("span", "↑ Spam", legend, "legend-spam");
    node("span", "↓ Legitimate", legend, "legend-ham");
    node("span", "Little change", legend, "legend-neutral");
    renderExplainedEmail(explanation, area, p);
    renderImpactSummary(explanation, area);
    if (explanation.tested_sentences < explanation.total_sentences)
      node(
        "p",
        `Tested ${explanation.tested_sentences} of ${explanation.total_sentences} sentences; remaining text is uncoloured.`,
        area,
        "fine-print",
      );
  } else
    node(
      "p",
      "Restart the updated server to enable sentence highlights.",
      area,
      "fine-print",
    );
  const perf = node("section", undefined, body, "performance-block");
  node("p", "Loading saved metrics…", perf, "fine-print");
  getEvidence()
    .then((data) => {
      if (!card.isConnected) return;
      perf.replaceChildren();
      const model = data.models.find((m) => m.id === result.id);
      if (!model || model.status !== "ready") {
        node("p", "No saved scores.", perf);
        return;
      }
      renderMergedResults(model, perf);
      const matrixPanel = card.querySelector(".matrix-panel");
      matrixPanel.replaceChildren(perf.querySelector(".fixed-matrix"));
      const matrixControls = perf.querySelector(".split-switch");
      if (matrixControls) matrixPanel.prepend(matrixControls);
      node(
        "p",
        "Saved results · compare the same split. Different text features were used.",
        perf,
        "fine-print",
      );
    })
    .catch((error) => {
      perf.replaceChildren();
      node("p", error.message, perf, "fine-print");
    });
}
function renderPerformance(e, parent, showTitle = true) {
  if (showTitle) node("h3", e.split + " results", parent);
  renderDoughnut(e, parent);
  for (const [key, label] of [
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1", "F1"],
  ]) {
    const row = node("div", undefined, parent, "metric-row");
    const heading = node("div", undefined, row, "metric-heading");
    node("span", label, heading);
    node("strong", (e.metrics[key] * 100).toFixed(2) + "%", heading);
    const track = node("div", undefined, row, "metric-track");
    track.setAttribute("role", "img");
    track.setAttribute(
      "aria-label",
      `${e.split} ${label} ${(e.metrics[key] * 100).toFixed(2)} percent`,
    );
    node("div", undefined, track, "metric-fill").style.width = e.metrics[key] * 100 + "%";
  }
  const axis = node("div", undefined, parent, "chart-axis");
  ["0", "25", "50", "75", "100%"].forEach((x) => node("span", x, axis));
  if (typeof e.metrics.false_positive_rate === "number")
    node(
      "p",
      "False alarms: " + (e.metrics.false_positive_rate * 100).toFixed(2) + "%",
      parent,
      "fine-print",
    );
  const area = accordion(parent, "Confusion matrix");
  const table = node("table", undefined, area, "heatmap");
  table.setAttribute("aria-label", e.split + " confusion matrix");
  let row = node("tr", undefined, table);
  ["Actual / Pred.", "Legitimate", "Spam"].forEach((x) => node("th", x, row));
  e.matrix.forEach((values, i) => {
    row = node("tr", undefined, table);
    node("th", i ? "Spam" : "Legitimate", row);
    const total = values.reduce((a, b) => a + b, 0);
    values.forEach((n) => {
      const ratio = total ? n / total : 0;
      const cell = node("td", undefined, row);
      cell.style.background = `rgba(97,97,255,${0.08 + 0.85 * ratio})`;
      cell.style.color = ratio > 0.45 ? "white" : "#323338";
      node("strong", String(n), cell);
      node("small", (ratio * 100).toFixed(1) + "%", cell);
    });
  });
  node(
    "p",
    "Actual rows · predicted columns · shading by row percentage",
    area,
    "fine-print",
  );
}

let impactTooltip;
let tooltipOwner;
// Sentence tooltips and influence
// -------------------------------
function hideImpactTooltip() {
  if (impactTooltip) impactTooltip.hidden = true;
  if (tooltipOwner) tooltipOwner.removeAttribute("aria-describedby");
  tooltipOwner = null;
}
function attachSentenceTooltip(mark, sentence, score) {
  function open() {
    if (!impactTooltip) {
      impactTooltip = node("div", undefined, document.body, "impact-tooltip");
      impactTooltip.id = "sentence-impact-tooltip";
      impactTooltip.setAttribute("role", "tooltip");
    }
    if (tooltipOwner && tooltipOwner !== mark)
      tooltipOwner.removeAttribute("aria-describedby");
    tooltipOwner = mark;
    impactTooltip.replaceChildren();
    const delta = sentence.delta * 100;
    const direction =
      Math.abs(delta) <= 0.05
        ? "Little change"
        : delta > 0
          ? "↑ Toward spam"
          : "↓ Toward legitimate";
    node(
      "strong",
      direction + " · " + (delta >= 0 ? "+" : "") + delta.toFixed(2) + " pp",
      impactTooltip,
    );
    node(
      "span",
      `Spam score: ${(score * 100).toFixed(2)}% → ${(sentence.without_score * 100).toFixed(2)}% without this sentence`,
      impactTooltip,
    );
    mark.setAttribute("aria-describedby", impactTooltip.id);
    impactTooltip.hidden = false;
    const rect = mark.getBoundingClientRect();
    const width = Math.min(300, window.innerWidth - 24);
    impactTooltip.style.width = width + "px";
    impactTooltip.style.left =
      Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)) + "px";
    const h = impactTooltip.getBoundingClientRect().height;
    impactTooltip.style.top =
      (rect.top > h + 12
        ? rect.top - h - 8
        : Math.min(rect.bottom + 8, window.innerHeight - h - 12)) + "px";
  }
  mark.addEventListener("mouseenter", open);
  mark.addEventListener("mouseleave", () => {
    if (document.activeElement !== mark) hideImpactTooltip();
  });
  mark.addEventListener("focus", open);
  mark.addEventListener("blur", hideImpactTooltip);
  mark.addEventListener("click", open);
  mark.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hideImpactTooltip();
      e.stopPropagation();
    }
  });
}
window.addEventListener("scroll", hideImpactTooltip, true);
window.addEventListener("resize", hideImpactTooltip);
window.addEventListener("hashchange", hideImpactTooltip);
function renderImpactSummary(explanation, parent) {
  const values = explanation.sentences.map((s) => s.delta * 100);
  const positive = values.reduce((sum, v) => sum + Math.max(0, v), 0);
  const negative = values.reduce((sum, v) => sum + Math.min(0, v), 0);
  const net = positive + negative;
  const box = node("div", undefined, parent, "total-impact");
  const heading = node("div", undefined, box, "total-heading");
  const netLabel = node("span", "Net tested impact", heading);
  netLabel.title =
    "Sum of tested sentence-removal effects; effects can overlap and do not add up to the spam score.";
  node("strong", (net >= 0 ? "+" : "") + net.toFixed(2) + " pp", heading);
  const scale = Math.max(positive, Math.abs(negative), 0.01);
  for (const [label, value, cls] of [
    ["↑ Spam", positive, "impact-red"],
    ["↓ Legitimate", negative, "impact-green"],
  ]) {
    const row = node("div", undefined, box, "total-row");
    node("span", label, row);
    const track = node("div", undefined, row, "impact-track");
    track.setAttribute("role", "img");
    track.setAttribute(
      "aria-label",
      `${label}: ${value.toFixed(2)} percentage points, sum of tested removal effects`,
    );
    node("div", undefined, track, "impact-bar " + cls).style.width =
      (Math.abs(value) / scale) * 100 + "%";
    node("b", (value >= 0 ? "+" : "") + value.toFixed(2) + " pp", row);
  }
}

// Charts and dataset overview
// ---------------------------
function renderDoughnut(e, parent) {
  const [[tn, fp], [fn, tp]] = e.matrix;
  const total = tn + fp + fn + tp;
  if (!total) return;
  const outcomes = [
    ["TN", "Legitimate → legitimate", tn, "#44e7c6"],
    ["TP", "Spam → spam", tp, "#ad89ff"],
    ["FP", "Legitimate → spam", fp, "#ff6cae"],
    ["FN", "Spam → legitimate", fn, "#ffd166"],
  ];
  const wrap = node("section", undefined, parent, "four-outcome-chart");
  const ns = "http://www.w3.org/2000/svg";
  function s(tag, attrs, text) {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
    if (text !== undefined) e.textContent = text;
    return e;
  }
  const svg = s("svg", {
    viewBox: "0 0 180 180",
    role: "img",
    "aria-label": `${e.split} confusion chart`,
  });
  wrap.append(svg);
  let offset = 0;
  for (const [label, description, count, color] of outcomes) {
    const pct = (count / total) * 100;
    const circle = s("circle", {
      cx: 90,
      cy: 90,
      r: 63,
      fill: "none",
      stroke: color,
      "stroke-width": 22,
      pathLength: 100,
      "stroke-dasharray": `${pct} ${100 - pct}`,
      "stroke-dashoffset": -offset,
      transform: "rotate(-90 90 90)",
      tabindex: 0,
    });
    const title = `${label}: ${description}, ${count.toLocaleString()} (${pct.toFixed(2)}%)`;
    circle.setAttribute("aria-label", title);
    circle.append(s("title", {}, title));
    svg.append(circle);
    circle.dataset.chartPercent = String(pct);
    comparisonAnimationObserver.observe(circle);
    offset += pct;
  }
  svg.append(
    s(
      "text",
      {
        x: 90,
        y: 90,
        "text-anchor": "middle",
        fill: "#edf1ff",
        "font-size": 24,
        "font-weight": 600,
      },
      total.toLocaleString(),
    ),
  );
  svg.append(
    s(
      "text",
      {
        x: 90,
        y: 111,
        "text-anchor": "middle",
        fill: "#a6b0d4",
        "font-size": 10,
      },
      e.split === "Validation" ? "Emails" : e.split + " emails",
    ),
  );
  const legend = node("div", undefined, wrap, "four-legend");
  for (const [label, description, count, color] of outcomes) {
    const row = node("div", undefined, legend);
    const dot = node("i", undefined, row);
    dot.style.background = color;
    node("b", label, row);
    node("span", description, row);
    node("strong", count.toLocaleString(), row);
    node("small", ((count / total) * 100).toFixed(2) + "%", row);
  }
}
function renderProjectInfo(data) {
  if (document.getElementById("project-info")) return;
  const area = node("section", undefined, undefined, "project-info");
  area.id = "project-info";
  area.hidden = document.body.dataset.page !== "input";
  $("input-page").append(area);
  const d = data.dataset;
  const heading = node("div", undefined, area, "project-heading");
  node("h2", "DATA / MODEL NOTES", heading);
  node(
    "span",
    data.comparison_rows.toLocaleString() + " SAME TEST EMAILS",
    heading,
    "section-label",
  );
  const totals = Object.values(d.splits).reduce(
    (a, s) => ({
      spam: a.spam + s.spam,
      legitimate: a.legitimate + s.legitimate,
    }),
    { spam: 0, legitimate: 0 },
  );
  const stats = node("div", undefined, area, "dataset-stats");
  for (const [label, value] of [
    ["TOTAL EMAILS", d.total],
    ["LEGITIMATE", totals.legitimate],
    ["SPAM", totals.spam],
  ]) {
    const box = node("div", undefined, stats);
    node("strong", value.toLocaleString(), box);
    node("span", label, box);
  }
  const graph = node("div", undefined, area, "split-chart");
  for (const [key, color] of [
    ["train", "#44e7c6"],
    ["validation", "#ad89ff"],
    ["test", "#ff6cae"],
  ]) {
    const split = d.splits[key];
    const bar = node(
      "div",
      `${key.toUpperCase()} ${((split.rows / d.total) * 100).toFixed(1)}%`,
      graph,
    );
    bar.style.width = (split.rows / d.total) * 100 + "%";
    bar.style.background = color;
    bar.title = split.rows.toLocaleString() + " emails";
  }
  const aboutLink = node("a", "About the data & models ↗", area, "about-page-link");
  aboutLink.href = "#about";
  const about = $("about-content");
  about.replaceChildren();
  node("p", d.name, about);
  const link = node("a", "Dataset source ↗", about);
  link.href = d.source_url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  node("p", d.version_note, about);
  node("p", d.audit_note, about);
  node("p", data.comparison_note, about);
  for (const model of data.models) {
    node("h4", model.name, about);
    for (const [key, value] of Object.entries(model.about))
      node("p", key + ": " + value, about);
  }
}
getEvidence()
  .then(renderProjectInfo)
  .catch(() => {});

// Detailed model results and matrices
// -----------------------------------
function renderMergedResults(model, parent) {
  node("h3", "RESULTS", parent);
  const evaluations = model.evaluations
    .slice()
    .sort((a, b) => (a.split === b.split ? 0 : a.split === "Test" ? -1 : 1));
  const legend = node("div", undefined, parent, "comparison-legend");
  for (const e of evaluations) {
    if (e.split !== "Validation")
      node("span", e.split, legend, "split-legend " + e.split.toLowerCase());
  }
  for (const [key, label] of [
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1", "F1"],
  ]) {
    const group = node("div", undefined, parent, "paired-metric");
    node("h4", label, group);
    for (const e of evaluations) {
      const value = e.metrics[key];
      const row = node("div", undefined, group, "paired-row");
      node("span", e.split === "Validation" ? "Val." : e.split, row);
      const track = node("div", undefined, row, "metric-track");
      track.setAttribute("role", "img");
      track.setAttribute(
        "aria-label",
        `${e.split} ${label}: ${(value * 100).toFixed(2)} percent`,
      );
      node("div", undefined, track, "metric-fill " + e.split.toLowerCase()).style.width =
        value * 100 + "%";
      node("strong", (value * 100).toFixed(2) + "%", row);
    }
  }
  const axis = node("div", undefined, parent, "chart-axis");
  ["0", "25", "50", "75", "100%"].forEach((x) => node("span", x, axis));
  const title = node("div", undefined, parent, "confusion-title");
  node("h4", "CONFUSION CHART", title);
  const controls = node("div", undefined, parent, "split-switch");
  controls.setAttribute("role", "tablist");
  controls.setAttribute("aria-label", model.name + " confusion chart split");
  const chart = node("div", undefined, parent, "selected-confusion");
  chart.id = "confusion-" + model.id;
  chart.setAttribute("role", "tabpanel");
  const matrixSection = node("section", undefined, parent, "fixed-matrix");
  node("h4", "Matrix", matrixSection);
  const matrix = node("div", undefined, matrixSection);
  const buttons = [];
  function select(e, index) {
    buttons.forEach((b, i) => {
      b.classList.toggle("selected", i === index);
      b.setAttribute("aria-selected", i === index ? "true" : "false");
      b.tabIndex = i === index ? 0 : -1;
    });
    chart.setAttribute("aria-labelledby", buttons[index].id);
    chart.replaceChildren();
    renderDoughnut(e, chart);
    node(
      "p",
      "False alarms: " + (e.metrics.false_positive_rate * 100).toFixed(2) + "%",
      chart,
      "fine-print",
    );
    matrix.replaceChildren();
    drawSelectedMatrix(e, matrix);
  }
  evaluations.forEach((e, i) => {
    const button = node("button", e.split, controls);
    if (evaluations.length === 1 && e.split === "Validation") controls.hidden = true;
    button.type = "button";
    button.id = model.id + "-split-" + i;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", chart.id);
    button.onclick = () => select(e, i);
    button.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        const next =
          (i + (event.key === "ArrowRight" ? 1 : -1) + evaluations.length) %
          evaluations.length;
        select(evaluations[next], next);
        buttons[next].focus();
      }
    });
    buttons.push(button);
  });
  if (evaluations.length) select(evaluations[0], 0);
  node("p", "Test and validation shown together, never pooled.", parent, "fine-print");
}

function drawSelectedMatrix(e, parent) {
  const axis = node("div", undefined, parent, "matrix-axis");
  node("span", e.split === "Validation" ? "Evaluation set" : e.split + " set", axis);
  node("span", "Predicted →", axis);
  const table = node("table", undefined, parent, "matrix-table");
  table.setAttribute(
    "aria-label",
    e.split + " confusion matrix, actual classes in rows, predictions in columns",
  );
  const head = node("tr", undefined, node("thead", undefined, table));
  for (const label of ["Actual ↓", "Legit", "SPAM"]) {
    const th = node("th", label, head);
    th.scope = "col";
  }
  const tbody = node("tbody", undefined, table);
  const cells = [
    [
      ["TN", "Correct legit", "correct"],
      ["FP", "False alarm", "false-alarm"],
    ],
    [
      ["FN", "Missed spam", "missed-spam"],
      ["TP", "Correct spam", "correct"],
    ],
  ];
  e.matrix.forEach((values, i) => {
    const row = node("tr", undefined, tbody);
    const rowHead = node("th", i ? "SPAM" : "Legit", row);
    rowHead.scope = "row";
    const total = values.reduce((a, b) => a + b, 0);
    values.forEach((count, j) => {
      const [code, label, tone] = cells[i][j];
      const cell = node("td", undefined, row);
      const tile = node("div", undefined, cell, "matrix-tile " + tone);
      const ratio = total ? count / total : 0;
      tile.style.setProperty("--cell-weight", String(0.08 + ratio * 0.18));
      node("span", code, tile, "matrix-code");
      const countLabel = node("strong", count.toLocaleString(), tile, "matrix-count");
      countLabel.dataset.targetCount = String(count);
      countLabel.setAttribute("aria-label", count.toLocaleString());
      comparisonAnimationObserver.observe(countLabel);
      node("span", label, tile, "matrix-name");
      node("small", (ratio * 100).toFixed(2) + "%", tile, "matrix-share");
      tile.title = `${code}: ${label}, ${count} emails, ${(ratio * 100).toFixed(2)}% of actual ${i ? "spam" : "legitimate"} emails.`;
    });
  });
  // Balance the row-label column so the four data cells sit in the card center.
  for (const [index, row] of [...table.rows].entries()) {
    const spacer = node(index === 0 ? "th" : "td", undefined, row, "matrix-spacer");
    spacer.setAttribute("aria-hidden", "true");
  }
  const legend = node("div", undefined, parent, "matrix-key");
  for (const [label, cls] of [
    ["Correct", "correct"],
    ["False alarm", "false-alarm"],
    ["Missed spam", "missed-spam"],
  ])
    node("span", label, legend, cls);
  node("p", "Percentages are within each actual class.", parent, "fine-print");
}

// Highlighted email text
// ----------------------
function renderExplainedEmail(explanation, parent, score) {
  const chars = Array.from(explanation.text);
  const subjectChars = Array.from(submittedSubject);
  const hasSubject =
    submittedSubject.length > 0 && explanation.text.startsWith(submittedSubject + "\n");
  const email = node("div", undefined, parent, "sentence-email explained-email");
  const title = hasSubject ? node("p", undefined, email, "explained-subject") : null;
  function range(target, start, end) {
    let cursor = start;
    for (const sentence of explanation.sentences) {
      const from = Math.max(start, sentence.start),
        to = Math.min(end, sentence.end);
      if (to <= from) continue;
      if (from > cursor)
        target.append(document.createTextNode(chars.slice(cursor, from).join("")));
      const tone =
        sentence.delta > 0.0005
          ? "toward-spam"
          : sentence.delta < -0.0005
            ? "toward-ham"
            : "low-impact";
      const mark = node(
        "mark",
        chars.slice(from, to).join(""),
        target,
        "sentence-mark " + tone,
      );
      mark.tabIndex = 0;
      mark.style.setProperty(
        "--influence",
        String(0.2 + Math.min(1, Math.abs(sentence.delta) / 0.15) * 0.2),
      );
      const description = `Removing this sentence changes the spam score from ${(score * 100).toFixed(2)}% to ${(sentence.without_score * 100).toFixed(2)}%.`;
      mark.setAttribute("aria-label", mark.textContent + " " + description);
      attachSentenceTooltip(mark, sentence, score);
      cursor = to;
    }
    if (cursor < end)
      target.append(document.createTextNode(chars.slice(cursor, end).join("")));
  }
  if (hasSubject) range(title, 0, subjectChars.length);
  const body = node("div", undefined, email, "explained-message");
  range(body, hasSubject ? subjectChars.length + 1 : 0, chars.length);
}

// Give each visible title its own irregular glitch rhythm.
// Optional title animation
// ------------------------
function startTitleGlitches() {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  for (const title of document.querySelectorAll("#input-title, #results-title")) {
    let timer;
    function schedule() {
      timer = window.setTimeout(
        () => {
          if (
            !reducedMotion.matches &&
            !document.hidden &&
            title.getClientRects().length
          ) {
            const strength = 0.6 + Math.random() * 1.1;
            title.style.setProperty("--glitch-strength", String(strength));
            title.style.setProperty(
              "--glitch-duration",
              `${180 + Math.random() * 260}ms`,
            );
            title.classList.add("is-glitching");
          }
          schedule();
        },
        1000 + Math.random() * 3000,
      );
    }
    title.addEventListener("animationend", (event) => {
      if (event.animationName === "random-title-glitch") {
        title.classList.remove("is-glitching");
      }
    });
    reducedMotion.addEventListener("change", () => {
      window.clearTimeout(timer);
      title.classList.remove("is-glitching");
      if (!reducedMotion.matches) schedule();
    });
    if (!reducedMotion.matches) schedule();
  }
}
startTitleGlitches();

// Comparison buttons switch all three models to the selected view.
const comparisonViews = ["prediction", "results", "matrix"];
let comparisonViewIndex = 0;
function setComparisonView(index) {
  comparisonViewIndex = (index + comparisonViews.length) % comparisonViews.length;
  const view = comparisonViews[comparisonViewIndex];
  $("results-page").dataset.view = view;
  document.querySelectorAll(".comparison-panel").forEach((panel) => {
    panel.hidden = panel.dataset.view !== view;
  });
  document.querySelectorAll("[data-comparison-view]").forEach((button, i) => {
    button.setAttribute("aria-pressed", String(i === comparisonViewIndex));
  });
  requestAnimationFrame(fitComparisonPanels);
}
function usePageScroll() {
  return matchMedia("(max-width: 900px), (max-height: 700px)").matches;
}

function fitComparisonPanels() {
  document.querySelectorAll(".comparison-panel").forEach((panel) => {
    panel.style.removeProperty("zoom");
  });
}

document.querySelectorAll("[data-comparison-view]").forEach((button, index) => {
  button.addEventListener("click", () => setComparisonView(index));
});
new MutationObserver(() => requestAnimationFrame(fitComparisonPanels)).observe(
  $("cards"),
  { childList: true, subtree: true },
);
window.addEventListener("resize", fitComparisonPanels);

// Outside the model cards, a deliberate wheel gesture cycles comparison views.
let outsideWheelLastEvent = 0;
let outsideWheelAccumulated = 0;
let outsideWheelLastSwitch = 0;
window.addEventListener("wheel", (event) => {
  if (document.body.dataset.page !== "results" || event.ctrlKey) return;
  if (event.target.closest("#cards, .impact-tooltip")) return;
  if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
  event.preventDefault();
  const now = performance.now();
  const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? innerHeight : 1);
  if (now - outsideWheelLastEvent > 180 || Math.sign(delta) !== Math.sign(outsideWheelAccumulated)) outsideWheelAccumulated = 0;
  outsideWheelLastEvent = now;
  if (now - outsideWheelLastSwitch < 650) return;
  outsideWheelAccumulated += delta;
  if (Math.abs(outsideWheelAccumulated) < 40) return;
  setComparisonView(comparisonViewIndex + Math.sign(outsideWheelAccumulated));
  outsideWheelLastSwitch = now;
  outsideWheelAccumulated = 0;
}, { passive: false });

// Animate charts and counts only when their comparison panel becomes visible.
const comparisonAnimationFrames = new WeakMap();
const comparisonAnimationObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      const element = entry.target;
      cancelAnimationFrame(comparisonAnimationFrames.get(element));
      if (!element.isConnected) {
        comparisonAnimationObserver.unobserve(element);
        continue;
      }
      const chart = element.dataset.chartPercent !== undefined;
      const target = Number(
        chart ? element.dataset.chartPercent : element.dataset.targetCount,
      );
      const render = (progress) => {
        const value = target * progress;
        if (chart) element.setAttribute("stroke-dasharray", `${value} ${100 - value}`);
        else element.textContent = Math.round(value).toLocaleString();
      };
      if (
        !entry.isIntersecting ||
        matchMedia("(prefers-reduced-motion: reduce)").matches
      ) {
        render(1);
        continue;
      }
      const start = performance.now();
      render(0);
      function frame(now) {
        const progress = Math.min(1, (now - start) / 850);
        render(1 - Math.pow(1 - progress, 3));
        if (progress < 1)
          comparisonAnimationFrames.set(element, requestAnimationFrame(frame));
      }
      comparisonAnimationFrames.set(element, requestAnimationFrame(frame));
    }
  },
  { threshold: 0.1 },
);
