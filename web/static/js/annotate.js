// Video annotator: label local videos in the browser and export the official ground-truth JSON.
import { chrome, el, CLASSES, CLASS_ORDER, famColor, className, fmtTime } from "./common.js";

chrome();

const STORE = "rw-labels";
// "e" is the end-of-segment key, so the last four classes use q, w, r, t.
const KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "q", "w", "r", "t"];
const RATES = [0.5, 1, 2];
const SVGNS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);
const video = $("video");

let labels = {};            // file name -> { duration, fps, events: [{ id, s, e, label }] }
const urls = new Map();     // file name -> object URL (videos opened this session)
let current = null;         // file name of the video on screen
let selected = CLASS_ORDER[0];
let pending = {};           // class -> start time set with S (current video only)
let preds = null;           // { team, videos: { name: { events } } }
let rateIdx = 1;
let seq = 1;
let raf = 0;
let tableDirty = false;
let geom = null;            // timeline geometry for seeking / playhead

// ------------------------------------------------------------------ helpers
const round2 = (x) => Math.round(x * 100) / 100;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const basename = (p) => String(p).split(/[\\/]/).pop();
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

// JSON number with `d` decimals that always keeps a decimal point (60 -> 60.0).
function jnum(x, d = 2) {
  const f = 10 ** d;
  let r = Math.round(x * f) / f;
  if (Object.is(r, -0)) r = 0;
  return Number.isInteger(r) ? r.toFixed(1) : String(r);
}

function status(msg, bad = false) {
  const s = $("status");
  s.textContent = msg;
  s.classList.toggle("bad", bad);
}

function entry(name) {
  return (labels[name] ||= { duration: null, fps: 25, events: [] });
}
const cur = () => (current ? labels[current] : null);
const sortEvents = (v) => v.events.sort((a, b) => a.s - b.s || a.e - b.e);
const fps = () => cur()?.fps || 25;

function duration() {
  if (isFinite(video.duration) && video.duration > 0 && video.src) return video.duration;
  const v = cur();
  if (v?.duration > 0) return v.duration;
  const ends = (v?.events || []).map((e) => e.e);
  return Math.max(10, ...ends);
}

// ------------------------------------------------------------------ storage / import / export
function mergeLabels(obj) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("expected an object keyed by file name");
  let added = 0;
  let skipped = 0;
  for (const [key, v] of Object.entries(obj)) {
    if (!v || typeof v !== "object") { skipped++; continue; }
    const name = basename(key);
    const isNew = !labels[name];
    const t = entry(name);
    const dur = Number(v.duration);
    const f = Number(v.fps);
    if (dur > 0 && !(t.duration > 0)) t.duration = dur;
    if (f > 0 && (isNew || !t.events.length)) t.fps = f;
    for (const ev of Array.isArray(v.events) ? v.events : []) {
      const [s, e, label] = Array.isArray(ev) ? ev : [];
      if (typeof s !== "number" || typeof e !== "number" || !CLASSES[label]) { skipped++; continue; }
      if (t.events.some((x) => x.label === label && Math.abs(x.s - s) < 0.005 && Math.abs(x.e - e) < 0.005)) continue;
      t.events.push({ id: seq++, s, e, label });
      added++;
    }
    sortEvents(t);
  }
  return { added, skipped };
}

function exportText() {
  const names = Object.keys(labels).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  if (!names.length) return "{}\n";
  const out = ["{"];
  names.forEach((name, i) => {
    const v = labels[name];
    sortEvents(v);
    const dur = v.duration > 0 ? v.duration : Math.max(0, ...v.events.map((e) => e.e));
    const evs = v.events.map((e) => `      [${jnum(e.s)}, ${jnum(e.e)}, ${JSON.stringify(e.label)}]`);
    out.push(`  ${JSON.stringify(name)}: {`);
    out.push(`    "duration": ${jnum(dur)},`);
    out.push(`    "fps": ${jnum(v.fps || 25)},`);
    out.push(evs.length ? `    "events": [\n${evs.join(",\n")}\n    ]` : `    "events": []`);
    out.push(`  }${i < names.length - 1 ? "," : ""}`);
  });
  out.push("}");
  return out.join("\n") + "\n";
}

function save() {
  try { localStorage.setItem(STORE, exportText()); } catch { /* storage may be unavailable */ }
}

function restore() {
  let raw = null;
  try { raw = localStorage.getItem(STORE); } catch { /* storage may be unavailable */ }
  if (!raw) return;
  try { mergeLabels(JSON.parse(raw)); } catch { /* ignore a corrupt autosave */ }
}

function download(text, name) {
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "application/json" })), download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ------------------------------------------------------------------ videos
function addFiles(list) {
  let first = null;
  for (const f of list) {
    if (!/^video\//.test(f.type) && !/\.(mp4|m4v|mov|webm|mkv|avi)$/i.test(f.name)) continue;
    if (urls.has(f.name)) URL.revokeObjectURL(urls.get(f.name));
    urls.set(f.name, URL.createObjectURL(f));
    entry(f.name);
    first = first || f.name;
  }
  if (!first) { status("That doesn't look like a video file.", true); return; }
  openVideo(first);
  save();
}

function openVideo(name) {
  current = name;
  pending = {};
  video.defaultPlaybackRate = RATES[rateIdx];
  video.src = urls.get(name);
  $("fps").value = entry(name).fps;
  $("player").hidden = false;
  $("controls").hidden = false;
  $("noVideo").hidden = true;
  $("drop").classList.add("compact");
  status(`Opened ${name}.`);
  renderAll();
}

video.addEventListener("loadedmetadata", () => {
  video.playbackRate = RATES[rateIdx];
  if (current && isFinite(video.duration)) {
    entry(current).duration = round2(video.duration);
    save();
  }
  renderAll();
});
video.addEventListener("error", () => {
  if (current) status(`This browser can't play ${current}. Try an H.264 .mp4 or a .webm file.`, true);
});

function togglePlay() {
  if (!current) return;
  if (video.paused || video.ended) video.play().catch(() => {});
  else video.pause();
}

function seek(t) {
  if (!current) return;
  video.currentTime = clamp(t, 0, duration());
  updateNow();
}

function step(dt) {
  if (!current) return;
  video.pause();
  seek(video.currentTime + dt);
}

function cycleRate(dir) {
  rateIdx = (rateIdx + dir + RATES.length) % RATES.length;
  video.playbackRate = video.defaultPlaybackRate = RATES[rateIdx];
  $("rate").textContent = `${RATES[rateIdx]}×`;
}

function tick() {
  updateNow();
  raf = !video.paused && !video.ended ? requestAnimationFrame(tick) : 0;
}
video.addEventListener("play", () => { if (!raf) raf = requestAnimationFrame(tick); updatePlayBtn(); });
video.addEventListener("pause", () => { updateNow(); updatePlayBtn(); });
video.addEventListener("ended", updatePlayBtn);
video.addEventListener("timeupdate", updateNow);
video.addEventListener("seeked", updateNow);

function updatePlayBtn() {
  $("play").firstChild.textContent = video.paused || video.ended ? "Play " : "Pause ";
}

function updateNow() {
  const t = current ? video.currentTime || 0 : 0;
  $("clock").textContent = `${fmtTime(t)} · f ${Math.round(t * fps())}`;
  if (geom) {
    const x = geom.X(t);
    geom.playhead.setAttribute("x1", x);
    geom.playhead.setAttribute("x2", x);
  }
}

// ------------------------------------------------------------------ marking
function selectClass(label) {
  selected = label;
  renderClasses();
}

function markStart() {
  if (!current) { status("Open a video first.", true); return; }
  pending[selected] = video.currentTime;
  status(`Start of ${className(selected)} at ${fmtTime(video.currentTime)}. Press E at the end.`);
  renderClasses();
}

function markEnd() {
  if (!current) { status("Open a video first.", true); return; }
  const t = video.currentTime;
  const had = selected in pending;
  const s0 = had ? pending[selected] : Math.max(0, t - 2);
  delete pending[selected];
  const [s, e] = s0 <= t ? [s0, t] : [t, s0];
  const v = cur();
  v.events.push({ id: seq++, s: round2(s), e: round2(e), label: selected });
  sortEvents(v);
  save();
  status(`Added ${className(selected)} ${fmtTime(s)}–${fmtTime(e)}${had ? "" : " (no start set: used now − 2 s)"}.`);
  renderAll();
}

// ------------------------------------------------------------------ validation
function validate(v) {
  const issues = new Map();
  const add = (id, msg) => { if (!issues.has(id)) issues.set(id, []); issues.get(id).push(msg); };
  const dur = v.duration > 0 ? v.duration : null;
  for (const ev of v.events) {
    if (ev.s < 0) add(ev.id, "start is negative");
    if (ev.s >= ev.e) add(ev.id, "start ≥ end");
    if (dur && ev.e > dur + 0.5) add(ev.id, `ends after the video (${dur.toFixed(2)} s)`);
  }
  const groups = [];
  for (const label of CLASS_ORDER) {
    const evs = v.events.filter((e) => e.label === label && e.e > e.s).sort((a, b) => a.s - b.s);
    let g = [];
    let end = -Infinity;
    for (const ev of evs) {
      if (g.length && ev.s <= end) { g.push(ev); end = Math.max(end, ev.e); continue; }
      if (g.length > 1) groups.push({ label, evs: g });
      g = [ev];
      end = ev.e;
    }
    if (g.length > 1) groups.push({ label, evs: g });
  }
  for (const g of groups) for (const ev of g.evs) add(ev.id, `overlaps another ${className(g.label)} segment`);
  return { issues, groups };
}

function mergeGroup(g) {
  const v = cur();
  const keep = g.evs[0];
  keep.s = Math.min(...g.evs.map((e) => e.s));
  keep.e = Math.max(...g.evs.map((e) => e.e));
  const drop = new Set(g.evs.slice(1).map((e) => e.id));
  v.events = v.events.filter((e) => !drop.has(e.id));
  sortEvents(v);
  save();
  status(`Merged ${g.evs.length} ${className(g.label)} segments into ${fmtTime(keep.s)}–${fmtTime(keep.e)}.`);
  renderAll();
}

// ------------------------------------------------------------------ rendering
function renderAll() {
  renderPicker();
  renderClasses();
  renderTable();
  renderTimeline();
  renderPredInfo();
  $("dur").textContent = current && cur().duration > 0 ? `${cur().duration.toFixed(2)} s` : "–";
  $("rate").textContent = `${RATES[rateIdx]}×`;
  updatePlayBtn();
  updateNow();
}

function renderPicker() {
  const box = $("picker");
  box.replaceChildren();
  for (const name of urls.keys()) {
    const n = labels[name]?.events.length || 0;
    box.append(el("button", {
      type: "button", role: "tab", "aria-selected": String(name === current), title: name,
      onclick: () => { if (name !== current) openVideo(name); },
    }, name, el("span", { class: "pill num" }, n)));
  }
  box.hidden = !urls.size;
  const others = Object.keys(labels).filter((n) => !urls.has(n) && labels[n].events.length);
  const stored = $("stored");
  stored.hidden = !others.length;
  stored.textContent = others.length
    ? `Also saved in this browser: ${others.map((n) => `${n} (${labels[n].events.length})`).join(", ")}. Open the file again to edit its labels; they are included in the export.`
    : "";
}

function renderClasses() {
  for (const b of $("classes").children) {
    const label = b.dataset.label;
    b.setAttribute("aria-pressed", String(label === selected));
    const p = b.querySelector(".pend");
    p.textContent = label in pending ? `start ${fmtTime(pending[label])}` : "";
    p.hidden = !(label in pending);
  }
}

function buildClasses() {
  const box = $("classes");
  CLASS_ORDER.forEach((label, i) => {
    box.append(el("button", {
      class: "cls", type: "button", "data-label": label, "aria-pressed": "false",
      title: `${className(label)} (${label}) — key ${KEYS[i].toUpperCase()}`,
      onclick: () => selectClass(label),
    }, el("kbd", {}, KEYS[i].toUpperCase()),
    el("span", { class: `swatch fam-${CLASSES[label].family}`, "aria-hidden": "true" }),
    el("span", { class: "cls-text" }, className(label), el("span", { class: "pend", hidden: "" }))));
  });
}

function timeInput(ev, field) {
  const input = el("input", {
    type: "number", step: "0.1", min: "0", value: ev[field].toFixed(2), "aria-label": `${field === "s" ? "Start" : "End"} (seconds)`,
    onchange: () => {
      const x = parseFloat(input.value);
      if (!isFinite(x)) { input.value = ev[field].toFixed(2); return; }
      ev[field] = round2(x);
      tableDirty = true;
      save();
      refreshChecks();
      renderTimeline();
    },
  });
  const now = el("button", {
    class: "mini", type: "button", title: "Set to the current video time",
    onclick: () => {
      if (!current) return;
      ev[field] = round2(video.currentTime);
      input.value = ev[field].toFixed(2);
      sortEvents(cur());
      save();
      renderAll();
    },
  }, "⟲ now");
  return el("div", { class: "tcell" }, input, now);
}

function renderTable() {
  tableDirty = false;
  const v = cur();
  const body = $("segBody");
  body.replaceChildren();
  $("segCount").textContent = v ? plural(v.events.length, "segment") : "no video";
  if (!v || !v.events.length) {
    $("warnings").replaceChildren();
    body.append(el("tr", {}, el("td", { colspan: "6", class: "muted" },
      v ? "No segments yet. Pick a class, press S at the start and E at the end." : "Open a video to see its segments.")));
    return;
  }
  sortEvents(v);
  for (const ev of v.events) {
    const tr = el("tr", { class: "clickable", "data-id": ev.id,
      onclick: (e) => { if (!e.target.closest("input, button")) seek(ev.s); } },
    el("td", { class: "c-class" }, el("span", { class: `swatch fam-${CLASSES[ev.label].family}`, "aria-hidden": "true" }), className(ev.label)),
    el("td", { "data-k": "Start (s)" }, timeInput(ev, "s")),
    el("td", { "data-k": "End (s)" }, timeInput(ev, "e")),
    el("td", { class: "num c-dur", "data-k": "Duration" }),
    el("td", { class: "c-iss", "data-k": "Check" }),
    el("td", {}, el("button", { class: "mini", type: "button", "aria-label": `Delete ${className(ev.label)} segment`,
      onclick: () => {
        v.events = v.events.filter((x) => x.id !== ev.id);
        save();
        status(`Deleted ${className(ev.label)} ${fmtTime(ev.s)}–${fmtTime(ev.e)}.`);
        renderAll();
      } }, "Delete")));
    body.append(tr);
  }
  refreshChecks();
}

// Updates durations, per-row checks and overlap warnings without rebuilding the rows (keeps input focus).
function refreshChecks() {
  const v = cur();
  if (!v) return;
  const { issues, groups } = validate(v);
  const byId = new Map(v.events.map((e) => [String(e.id), e]));
  for (const tr of $("segBody").querySelectorAll("tr[data-id]")) {
    const ev = byId.get(tr.dataset.id);
    if (!ev) continue;
    tr.querySelector(".c-dur").textContent = `${(ev.e - ev.s).toFixed(2)} s`;
    const msgs = issues.get(ev.id);
    const cell = tr.querySelector(".c-iss");
    cell.replaceChildren(msgs ? el("span", { class: "iss" }, `⚠ ${msgs.join("; ")}`) : el("span", { class: "ok" }, "ok"));
  }
  $("warnings").replaceChildren(...groups.map((g) => el("div", { class: "warn", role: "alert" },
    el("span", {}, `${className(g.label)}: ${plural(g.evs.length, "segment")} overlap (${g.evs.map((e) => `${fmtTime(e.s)}–${fmtTime(e.e)}`).join(", ")}). `,
      "The organisers merge simultaneous same-class events into one segment — merge?"),
    el("button", { class: "btn sm", type: "button", onclick: () => mergeGroup(g) }, "Merge"))));
}

$("segBody").addEventListener("focusout", (e) => {
  if (tableDirty && !$("segBody").contains(e.relatedTarget)) {
    sortEvents(cur());
    renderAll();
  }
});

// ------------------------------------------------------------------ timeline
function svg(tag, attrs = {}, parent = null, text = null) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== null) n.textContent = text;
  if (parent) parent.append(n);
  return n;
}

function niceStep(dur, maxTicks) {
  for (const s of [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600]) if (dur / s <= maxTicks) return s;
  return 7200;
}
const tickFmt = (t) => `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}`;

function predEvents() {
  if (!preds || !current) return [];
  const evs = preds.videos[current]?.events;
  return (Array.isArray(evs) ? evs : []).filter((e) => Array.isArray(e) && typeof e[0] === "number" && typeof e[1] === "number" && CLASSES[e[2]]);
}

function renderTimeline() {
  const box = $("timeline");
  box.querySelector("svg")?.remove();
  geom = null;
  const v = cur();
  box.hidden = !v;
  $("tlHint").hidden = !v;
  if (!v) return;
  const W = Math.max(200, Math.floor(box.clientWidth || 600));
  const dur = duration();
  const pe = predEvents();
  const hasPred = !!preds;
  const rows = CLASS_ORDER.filter((l) => v.events.some((e) => e.label === l) || pe.some((p) => p[2] === l));
  const narrow = W < 560;
  const BAR = 12;
  const SUB = BAR + 2;
  const labelW = narrow ? (hasPred ? 34 : 0) : 184;
  const x0 = labelW;
  const x1 = W - 4;
  const X = (t) => x0 + (clamp(t, 0, dur) / dur) * (x1 - x0);
  const AXIS = 18;
  const root = svg("svg", { viewBox: `0 0 ${W} 100`, width: W, role: "img", "aria-label": "Timeline of labelled segments" });
  const grid = svg("g", {}, root);
  const bars = svg("g", {}, root);
  let y = AXIS + 6;

  const bar = (s, e, label, isPred) => {
    const w = Math.max(2, X(e) - X(s));
    svg("rect", {
      class: "bar", x: X(Math.min(s, e)), y, width: w, height: BAR, rx: 4, fill: famColor(label),
      opacity: isPred ? 0.45 : 1, "data-label": label, "data-s": s, "data-e": e, "data-pred": isPred ? "1" : "",
    }, bars);
  };

  for (const label of rows) {
    if (narrow) { svg("text", { class: "tl-label", x: 0, y: y + 10 }, bars, className(label)); y += 15; }
    else svg("text", { class: "tl-label", x: 0, y: y + BAR - 2 }, bars, className(label));
    if (narrow && hasPred) svg("text", { class: "tl-sub", x: 0, y: y + BAR - 3 }, bars, "label");
    for (const ev of v.events) if (ev.label === label) bar(ev.s, ev.e, label, false);
    if (hasPred) {
      y += SUB;
      svg("text", { class: "tl-sub", x: narrow ? 0 : labelW - 8, y: y + BAR - 3, "text-anchor": narrow ? "start" : "end" }, bars, "pred");
      for (const p of pe) if (p[2] === label) bar(p[0], p[1], label, true);
    }
    y += SUB + 6;
  }
  if (!rows.length) {
    svg("text", { class: "tl-empty", x: x0 + 4, y: y + 12 }, bars, "No segments yet — press S at the start and E at the end.");
    y += 24;
  }
  const H = y;
  // ticks + gridlines
  const stepS = niceStep(dur, Math.max(2, Math.floor((x1 - x0) / 70)));
  for (let t = 0; t <= dur + 1e-6; t += stepS) {
    const x = X(t);
    svg("line", { class: "gridline", x1: x, x2: x, y1: AXIS, y2: H }, grid);
    const anchor = x - x0 < 18 ? "start" : x1 - x < 18 ? "end" : "middle";
    svg("text", { class: "tick", x, y: 12, "text-anchor": anchor }, grid, tickFmt(t));
  }
  svg("line", { class: "axis", x1: x0, x2: x1, y1: AXIS, y2: AXIS }, grid);
  const playhead = svg("line", { class: "playhead", x1: x0, x2: x0, y1: AXIS - 4, y2: H }, root);
  root.setAttribute("viewBox", `0 0 ${W} ${H}`);
  root.setAttribute("height", H);
  box.prepend(root);
  geom = { X, x0, x1, W, dur, playhead, root };

  const toTime = (e) => {
    const r = root.getBoundingClientRect();
    const x = ((e.clientX - r.left) * W) / r.width;
    return ((clamp(x, x0, x1) - x0) / (x1 - x0)) * dur;
  };
  root.addEventListener("pointerdown", (e) => {
    if (!current) return;
    root.setPointerCapture(e.pointerId);
    seek(toTime(e));
  });
  root.addEventListener("pointermove", (e) => {
    if (e.buttons & 1 && root.hasPointerCapture(e.pointerId)) seek(toTime(e));
    showTip(e);
  });
  root.addEventListener("pointerleave", hideTip);
  updateNow();
}

function tip() {
  let t = $("timeline").querySelector(".tooltip");
  if (!t) { t = el("div", { class: "tooltip", hidden: "" }); $("timeline").append(t); }
  return t;
}
function hideTip() { tip().hidden = true; }
function showTip(e) {
  const b = e.target.closest?.("rect.bar");
  const t = tip();
  if (!b) { t.hidden = true; return; }
  const s = +b.dataset.s;
  const en = +b.dataset.e;
  t.replaceChildren(
    el("div", { class: "tt-title" }, className(b.dataset.label)),
    el("div", { class: "tt-row" }, `${fmtTime(s)} – ${fmtTime(en)}`),
    el("div", { class: "tt-row" }, `duration ${(en - s).toFixed(2)} s`),
    el("div", { class: "tt-row" }, b.dataset.pred ? "prediction" : "your label"));
  t.hidden = false;
  const box = $("timeline").getBoundingClientRect();
  const left = clamp(e.clientX - box.left + 12, 0, box.width - t.offsetWidth);
  t.style.left = `${left}px`;
  t.style.top = `${e.clientY - box.top + 14}px`;
}

// ------------------------------------------------------------------ predictions
function renderPredInfo() {
  const info = $("predInfo");
  $("clearPred").hidden = !preds;
  $("adoptPred").hidden = !preds || !predEvents().length;
  if (!preds) return;
  const n = Object.keys(preds.videos).length;
  const team = preds.team ? `team “${preds.team}”` : "unnamed team";
  if (!current) { info.textContent = `Loaded predictions from ${team} for ${plural(n, "video")}. Open a video to compare.`; return; }
  const k = predEvents().length;
  info.textContent = current in preds.videos
    ? `Predictions from ${team}: ${plural(k, "event")} for ${current}, shown as faded bars in the “pred” rows.`
    : `Predictions from ${team} have no entry for ${current} (${plural(n, "video")} in the file).`;
}

// ------------------------------------------------------------------ wiring
buildClasses();
restore();

const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("file").click(); } });
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drag");
  addFiles(e.dataTransfer.files);
});
$("file").addEventListener("click", (e) => e.stopPropagation());
$("file").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });

$("play").addEventListener("click", togglePlay);
$("rate").addEventListener("click", () => cycleRate(1));
$("fps").addEventListener("change", () => {
  const f = parseFloat($("fps").value);
  if (!current) return;
  if (f > 0 && f <= 240) { cur().fps = f; save(); updateNow(); }
  else $("fps").value = cur().fps;
});
$("btnStart").addEventListener("click", markStart);
$("btnEnd").addEventListener("click", markEnd);

$("export").addEventListener("click", () => {
  if (!Object.keys(labels).length) { status("Nothing to export yet.", true); return; }
  download(exportText(), "ground_truth.json");
  status("Downloaded ground_truth.json (use it with evaluate.py --gt).");
});
$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(exportText());
    status("Copied the ground-truth JSON.");
  } catch {
    status("The clipboard is not available here; use Download instead.", true);
  }
});
$("import").addEventListener("click", () => $("importFile").click());
$("importFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  try {
    const { added, skipped } = mergeLabels(JSON.parse(await f.text()));
    save();
    renderAll();
    status(`Imported ${plural(added, "segment")} from ${f.name}${skipped ? ` (${skipped} skipped)` : ""}.`);
  } catch (err) {
    status(`Could not import ${f.name}: ${err.message}`, true);
  }
});
$("clearVideo").addEventListener("click", () => {
  const v = cur();
  if (!v || !v.events.length) { status("No labels to clear for this video."); return; }
  if (!confirm(`Delete all ${v.events.length} segments of ${current}?`)) return;
  v.events = [];
  pending = {};
  save();
  renderAll();
  status(`Cleared the labels of ${current}.`);
});
$("loadPred").addEventListener("click", () => $("predFile").click());
$("predFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  try {
    const d = JSON.parse(await f.text());
    if (!d || typeof d.videos !== "object" || Array.isArray(d.videos)) throw new Error("no \"videos\" object");
    preds = { team: d.team, videos: {} };
    for (const [k, v] of Object.entries(d.videos)) preds.videos[basename(k)] = v;
    renderAll();
  } catch (err) {
    status(`Could not read predictions from ${f.name}: ${err.message}`, true);
  }
});
$("clearPred").addEventListener("click", () => { preds = null; renderAll(); $("predInfo").textContent = "Predictions hidden."; });
// Copy the predicted events of this video into its labels, as a draft to correct.
$("adoptPred").addEventListener("click", () => {
  const pe = predEvents();
  const v = cur();
  if (!v || !pe.length) return;
  if (v.events.length && !confirm(`${current} already has ${plural(v.events.length, "segment")}. Add the ${plural(pe.length, "predicted event")} to them?`)) return;
  for (const [s, e, label] of pe) v.events.push({ id: seq++, s: round2(s), e: round2(e), label });
  sortEvents(v);
  save();
  renderAll();
  status(`Added ${plural(pe.length, "predicted event")} to the labels of ${current}: check each start, end and class.`);
});

// Sample videos published with the site (tools/build_site.py): browser-playable previews plus
// the pipeline's events for each, so the team can label without the multi-GB camera originals.
async function loadSamples() {
  let index;
  try {
    const r = await fetch("data/index.json", { cache: "no-cache" });
    if (!r.ok) return;
    index = await r.json();
  } catch { return; }
  const vids = (index.videos || []).filter((v) => v.id && v.video);
  if (!vids.length) return;
  const box = $("sampleBtns");
  box.replaceChildren(...vids.map((v) => el("button", {
    class: "btn sm", type: "button",
    title: `${fmtTime(v.duration)} · ${plural(v.n_events ?? 0, "predicted event")}`,
    onclick: () => openSample(v),
  }, v.video)));
  $("samples").hidden = false;
}

async function openSample(v) {
  urls.set(v.video, `data/samples/${encodeURIComponent(v.id)}/preview.mp4`);
  const e = entry(v.video);
  if (v.fps) e.fps = round2(v.fps);
  try {
    const r = await fetch(`data/samples/${encodeURIComponent(v.id)}/result.json`);
    if (r.ok) {
      const res = await r.json();
      preds ||= { team: "roadwatch (this site)", videos: {} };
      preds.videos[v.video] = { events: res.events || [] };
    }
  } catch { /* predictions are optional */ }
  openVideo(v.video);
  save();
}
loadSamples();

window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.defaultPrevented) return;
  if (e.target.closest?.("input, textarea, select, [contenteditable='true']")) return;
  const k = e.key;
  const lower = k.length === 1 ? k.toLowerCase() : k;
  const ci = KEYS.indexOf(lower);
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "ArrowLeft" || k === "ArrowRight") {
    if (!current) return;
    e.preventDefault();
    step((k === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 1 : 1 / fps()));
  } else if (k === "[") cycleRate(-1);
  else if (k === "]") cycleRate(1);
  else if (lower === "s") { e.preventDefault(); markStart(); }
  else if (lower === "e") { e.preventDefault(); markEnd(); }
  else if (ci >= 0 && !e.shiftKey) { e.preventDefault(); selectClass(CLASS_ORDER[ci]); }
});

new ResizeObserver(() => { if (geom && Math.floor($("timeline").clientWidth) !== geom.W) renderTimeline(); }).observe($("timeline"));
window.addEventListener("themechange", renderTimeline);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderTimeline);

renderAll();
status("Open a video, pick a class, then mark with S and E.");
