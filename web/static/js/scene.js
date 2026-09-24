// Scene editor: draw the fixed camera's road layout and export weights/zones.json (normalised coordinates).
import { chrome, el, fmtTime } from "./common.js";

chrome();

const STORE = "rw-zones";
const KINDS = ["crosswalks", "intersection", "u_turn_allowed", "ignore", "solid_lines", "stop_lines", "no_turn"];
const POLY = new Set(["crosswalks", "intersection", "u_turn_allowed", "ignore"]);
const META = {
  crosswalks: { label: "crosswalk", title: "Crosswalks", tool: "Crosswalk", token: "--ink", fill: 0.25, lw: 2 },
  intersection: { label: "intersection", title: "Intersection", tool: "Intersection", token: "--axis", fill: 0.25, lw: 2 },
  u_turn_allowed: { label: "u-turn allowed", title: "U-turn allowed", tool: "U-turn allowed", token: "--fam-manoeuvre", fill: 0.22, lw: 2 },
  ignore: { label: "ignore", title: "Ignore", tool: "Ignore", token: "--muted", fill: 0.3, lw: 2 },
  solid_lines: { label: "solid line", title: "Solid lines", tool: "Solid line", token: "--fam-pedestrian", lw: 3 },
  stop_lines: { label: "stop line", title: "Stop lines", tool: "Stop line", token: "--critical", lw: 3 },
  no_turn: { label: "no-turn", title: "No-turn rules", tool: "No-turn rule", token: "--fam-signal", token2: "--fam-collision", fill: 0.22, lw: 2 },
};
const HINTS = {
  select: "Click an item to select it, then drag its square handles to move vertices. Delete or Backspace removes it.",
  poly: "Click to add vertices; double-click, Enter or a click on the first vertex closes the polygon. Esc cancels, Backspace removes the last point.",
  solid_lines: "Click to add points along the solid line; double-click or Enter finishes. Esc cancels.",
  stop_lines: "Click the two ends of the stop line. The arrow shows the travel direction of the approaching traffic — use Flip direction if it points the wrong way.",
  no_turn_from: "No-turn rule, step 1 of 2: draw the FROM polygon (where the prohibited turn starts). Name the rule above.",
  no_turn_to: "No-turn rule, step 2 of 2: draw the TO polygon (where the prohibited turn ends).",
};

const $ = (id) => document.getElementById(id);
const cv = $("cv");
const ctx = cv.getContext("2d");

let zones = emptyZones();
let tool = "select";
let draft = null;         // { kind, pts, stage: "from" | "to" | null, from }
let sel = null;           // { kind, i }
let drag = null;          // { kind, i, part, vi, id }
let cursor = null;        // normalised pointer position (rubber band)
let size = { w: 640, h: 360 };
let baseBg = null;        // { src, w, h, label }
let dirsBg = null;        // learned lane directions image, or false if missing
let showDirs = false;
let lastDown = { t: 0, x: 0, y: 0 };
let T = {};               // colour tokens
let frameReq = 0;

// ------------------------------------------------------------------ helpers
function emptyZones() { return Object.fromEntries(KINDS.map((k) => [k, []])); }
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const isNum = (x) => typeof x === "number" && isFinite(x);
const bg = () => (showDirs && dirsBg ? dirsBg : baseBg);
const imgSize = () => { const b = bg(); return b ? [b.w, b.h] : [16, 9]; };
const P = (p) => [p[0] * size.w, p[1] * size.h];

function jnum(x, d) {
  const f = 10 ** d;
  let r = Math.round(x * f) / f;
  if (Object.is(r, -0)) r = 0;
  return Number.isInteger(r) ? r.toFixed(1) : String(r);
}

function normDeg(d) {
  let r = Math.round((((d % 360) + 360) % 360) * 10) / 10;
  if (r >= 360) r = 0;
  return r;
}

// Default direction as in scene.py: perpendicular to the line, pointing up the image.
function perpDirs(pts) {
  const [W, H] = imgSize();
  const vx = (pts[1][0] - pts[0][0]) * W;
  const vy = (pts[1][1] - pts[0][1]) * H;
  let n = [-vy, vx];
  if (n[1] > 0) n = [vy, -vx];
  const up = normDeg((Math.atan2(n[1], n[0]) * 180) / Math.PI);
  return [up, normDeg(up + 180)];
}
const angDiff = (a, b) => Math.abs(((a - b + 540) % 360) - 180);

function readTokens() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["--ink", "--ink-2", "--muted", "--axis", "--grid", "--surface", "--surface-2", "--page", "--accent",
    "--fam-manoeuvre", "--fam-pedestrian", "--fam-signal", "--fam-collision", "--critical"]) T[k] = cs.getPropertyValue(k).trim();
}

function msg(text, bad = false) {
  $("msg").textContent = text;
  $("msg").classList.toggle("bad", bad);
}

function itemName(kind, i) { return `${META[kind].label} ${i + 1}`; }

// Parts of an item: [{ part, pts, closed, color }]
function parts(kind, item) {
  if (POLY.has(kind)) return [{ part: null, pts: item, closed: true, color: T[META[kind].token] }];
  if (kind === "solid_lines") return [{ part: null, pts: item, closed: false, color: T[META[kind].token] }];
  if (kind === "stop_lines") return [{ part: null, pts: item.points, closed: false, color: T[META[kind].token] }];
  return [
    { part: "from", pts: item.from, closed: true, color: T["--fam-signal"] },
    { part: "to", pts: item.to, closed: true, color: T["--fam-collision"] },
  ];
}

// ------------------------------------------------------------------ persistence
function exportText() {
  const f = (x) => jnum(x, 4);
  const pt = (p) => `[${f(p[0])}, ${f(p[1])}]`;
  const pts = (a) => `[${a.map(pt).join(", ")}]`;
  const block = (key, rows, last) => {
    const comma = last ? "" : ",";
    return rows.length ? `  "${key}": [\n${rows.map((r) => `    ${r}`).join(",\n")}\n  ]${comma}` : `  "${key}": []${comma}`;
  };
  const out = ["{"];
  KINDS.forEach((k, idx) => {
    const last = idx === KINDS.length - 1;
    let rows;
    if (k === "stop_lines") rows = zones[k].map((s) => `{"points": ${pts(s.points)}, "dir": ${jnum(s.dir, 1)}}`);
    else if (k === "no_turn") rows = zones[k].map((r) => `{"name": ${JSON.stringify(r.name)}, "from": ${pts(r.from)}, "to": ${pts(r.to)}}`);
    else rows = zones[k].map(pts);
    out.push(block(k, rows, last));
  });
  out.push("}");
  return out.join("\n") + "\n";
}

function parseZones(obj) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("expected an object with zone kinds");
  let outOfRange = false;
  const pt = (p) => {
    if (!Array.isArray(p) || !isNum(p[0]) || !isNum(p[1])) return null;
    if (p[0] < -0.01 || p[0] > 1.01 || p[1] < -0.01 || p[1] > 1.01) outOfRange = true;
    return [clamp01(p[0]), clamp01(p[1])];
  };
  const list = (a, min) => {
    if (!Array.isArray(a)) return null;
    const r = a.map(pt);
    return r.length >= min && r.every(Boolean) ? r : null;
  };
  const arr = (k) => (Array.isArray(obj[k]) ? obj[k] : []);
  const z = emptyZones();
  let skipped = 0;
  for (const k of POLY) for (const p of arr(k)) { const r = list(p, 3); if (r) z[k].push(r); else skipped++; }
  for (const l of arr("solid_lines")) { const r = list(l, 2); if (r) z.solid_lines.push(r); else skipped++; }
  for (const s of arr("stop_lines")) {
    const r = list(s?.points, 2);
    if (!r) { skipped++; continue; }
    const two = r.slice(0, 2);
    z.stop_lines.push({ points: two, dir: isNum(s.dir) ? normDeg(s.dir) : perpDirs(two)[0] });
  }
  for (const n of arr("no_turn")) {
    const a = list(n?.from, 3);
    const b = list(n?.to, 3);
    if (a && b) z.no_turn.push({ name: String(n.name ?? ""), from: a, to: b }); else skipped++;
  }
  if (outOfRange) throw new Error("coordinates must be normalised to [0, 1]");
  return { z, skipped };
}

function save() {
  try { localStorage.setItem(STORE, exportText()); } catch { /* storage may be unavailable */ }
}

function restore() {
  let raw = null;
  try { raw = localStorage.getItem(STORE); } catch { /* storage may be unavailable */ }
  if (!raw) return;
  try { zones = parseZones(JSON.parse(raw)).z; } catch { /* ignore a corrupt autosave */ }
}

function commit() {
  save();
  renderList();
  renderTools();
  $("preview").textContent = exportText();
  draw();
}

function download(text, name) {
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "application/json" })), download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ------------------------------------------------------------------ drawing
function layout() {
  const cssW = Math.max(100, $("stage").clientWidth);
  const [iw, ih] = imgSize();
  const cssH = Math.round((cssW * ih) / iw);
  const dpr = window.devicePixelRatio || 1;
  cv.style.height = `${cssH}px`;
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  size = { w: cssW, h: cssH };
  draw();
}

function requestDraw() {
  if (!frameReq) frameReq = requestAnimationFrame(() => { frameReq = 0; draw(); });
}

function path(pts, closed) {
  ctx.beginPath();
  pts.forEach((p, j) => { const [x, y] = P(p); if (j) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
  if (closed) ctx.closePath();
}

function strokeHalo(color, lw, dashed = false) {
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = lw + 2.5;
  ctx.strokeStyle = T["--surface"];
  ctx.globalAlpha = 0.85;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.lineWidth = lw;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.setLineDash([]);
}

function arrow(from, to, color, lw) {
  const [x0, y0] = from;
  const [x1, y1] = to;
  const a = Math.atan2(y1 - y0, x1 - x0);
  const head = 9 + lw;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1 - Math.cos(a) * head * 0.6, y1 - Math.sin(a) * head * 0.6);
  strokeHalo(color, lw);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(a - 0.45), y1 - head * Math.sin(a - 0.45));
  ctx.lineTo(x1 - head * Math.cos(a + 0.45), y1 - head * Math.sin(a + 0.45));
  ctx.closePath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = T["--surface"];
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fill();
}

function stopArrow(sl) {
  const [a, b] = sl.points.map(P);
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const len = Math.min(60, Math.max(26, size.w * 0.05));
  const r = (sl.dir * Math.PI) / 180;
  return [mid, [mid[0] + Math.cos(r) * len, mid[1] + Math.sin(r) * len]];
}

function centroid(pts) {
  const n = pts.length;
  return [pts.reduce((s, p) => s + p[0], 0) / n, pts.reduce((s, p) => s + p[1], 0) / n];
}

function drawLabel(text, p, bold) {
  ctx.font = `${bold ? "600 " : ""}12px ${getComputedStyle(document.body).fontFamily}`;
  const w = ctx.measureText(text).width + 8;
  let [x, y] = P(p);
  x = Math.min(Math.max(2, x + 6), size.w - w - 2);
  y = Math.min(Math.max(2, y - 22), size.h - 20);
  ctx.globalAlpha = 0.88;
  ctx.fillStyle = T["--surface"];
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, 18, 4) : ctx.rect(x, y, w, 18);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.fillStyle = T["--ink"];
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 4, y + 9.5);
}

function handles(pts, color) {
  for (const p of pts) {
    const [x, y] = P(p);
    ctx.fillStyle = T["--surface"];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.fillRect(x - 5, y - 5, 10, 10);
    ctx.strokeRect(x - 5, y - 5, 10, 10);
  }
}

function drawBackground() {
  const b = bg();
  if (b) { ctx.drawImage(b.src, 0, 0, size.w, size.h); return; }
  ctx.fillStyle = T["--surface-2"];
  ctx.fillRect(0, 0, size.w, size.h);
  ctx.strokeStyle = T["--grid"];
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < 10; i++) {
    const x = Math.round((i * size.w) / 10) + 0.5;
    const y = Math.round((i * size.h) / 10) + 0.5;
    ctx.moveTo(x, 0); ctx.lineTo(x, size.h);
    ctx.moveTo(0, y); ctx.lineTo(size.w, y);
  }
  ctx.stroke();
  ctx.fillStyle = T["--muted"];
  ctx.font = `13px ${getComputedStyle(document.body).fontFamily}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const long = "No background image — open one or grab a video frame";
  ctx.fillText(ctx.measureText(long).width < size.w - 24 ? long : "No background image", size.w / 2, size.h / 2);
  ctx.textAlign = "start";
}

function draw() {
  ctx.clearRect(0, 0, size.w, size.h);
  drawBackground();
  const labels = [];
  for (const kind of KINDS) {
    zones[kind].forEach((item, i) => {
      const isSel = sel && sel.kind === kind && sel.i === i;
      const m = META[kind];
      const lw = m.lw + (isSel ? 1.5 : 0);
      for (const pr of parts(kind, item)) {
        path(pr.pts, pr.closed);
        if (pr.closed) {
          ctx.globalAlpha = (m.fill || 0.2) + (isSel ? 0.1 : 0);
          ctx.fillStyle = pr.color;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        strokeHalo(pr.color, lw);
        const suffix = pr.part ? ` ${pr.part}` : "";
        labels.push([`${itemName(kind, i)}${suffix}`, pr.pts[0], isSel]);
      }
      if (kind === "stop_lines") {
        const [a, b] = stopArrow(item);
        arrow(a, b, T["--critical"], 2.5);
      }
      if (kind === "no_turn") {
        const a = P(centroid(item.from));
        const b = P(centroid(item.to));
        ctx.globalAlpha = 0.9;
        arrow(a, b, T["--ink-2"], 1.5);
        ctx.globalAlpha = 1;
      }
    });
  }
  if (sel && zones[sel.kind]?.[sel.i]) {
    for (const pr of parts(sel.kind, zones[sel.kind][sel.i])) handles(pr.pts, pr.color);
  }
  drawDraft();
  for (const [text, p, bold] of labels) drawLabel(text, p, bold);
  if (draft) {
    const k = draft.kind;
    const title = k === "no_turn" ? `new no-turn rule: ${draft.stage}` : `new ${META[k].label}`;
    drawLabel(title, draft.pts[0] || cursor || [0.02, 0.08], true);
  }
}

function drawDraft() {
  if (!draft) return;
  const k = draft.kind;
  const color = k === "no_turn" ? T[draft.stage === "from" ? "--fam-signal" : "--fam-collision"] : T[META[k].token];
  if (k === "no_turn" && draft.from) {
    path(draft.from, true);
    ctx.globalAlpha = 0.22; ctx.fillStyle = T["--fam-signal"]; ctx.fill(); ctx.globalAlpha = 1;
    strokeHalo(T["--fam-signal"], 2);
  }
  const pts = cursor ? [...draft.pts, cursor] : draft.pts;
  if (pts.length >= 2) {
    path(pts, false);
    strokeHalo(color, META[k].lw, true);
  }
  draft.pts.forEach((p, j) => {
    const [x, y] = P(p);
    ctx.beginPath();
    ctx.arc(x, y, j === 0 ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = j === 0 ? T["--surface"] : color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.stroke();
  });
}

// ------------------------------------------------------------------ hit testing
function distSeg(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2)) : 0;
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

function inPoly(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function area(pts) {
  let s = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) s += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
  return Math.abs(s / 2);
}

function vertexHit(px) {
  if (!sel || !zones[sel.kind]?.[sel.i]) return null;
  for (const pr of parts(sel.kind, zones[sel.kind][sel.i])) {
    for (let vi = 0; vi < pr.pts.length; vi++) {
      const q = P(pr.pts[vi]);
      if (Math.hypot(q[0] - px[0], q[1] - px[1]) <= 11) return { ...sel, part: pr.part, vi };
    }
  }
  return null;
}

function itemHit(px) {
  // Thin lines first, then the smallest polygon containing the point (crosswalk inside the intersection box).
  for (const kind of ["stop_lines", "solid_lines"]) {
    for (let i = zones[kind].length - 1; i >= 0; i--) {
      const pts = parts(kind, zones[kind][i])[0].pts.map(P);
      for (let j = 1; j < pts.length; j++) if (distSeg(px, pts[j - 1], pts[j]) <= 8) return { kind, i };
    }
  }
  let best = null;
  for (const kind of KINDS) {
    if (!POLY.has(kind) && kind !== "no_turn") continue;
    zones[kind].forEach((item, i) => {
      for (const pr of parts(kind, item)) {
        const pts = pr.pts.map(P);
        let hit = inPoly(px, pts);
        for (let j = 0; !hit && j < pts.length; j++) hit = distSeg(px, pts[j], pts[(j + 1) % pts.length]) <= 6;
        if (hit) {
          const a = area(pts);
          if (!best || a < best.a) best = { kind, i, a };
        }
      }
    });
  }
  return best && { kind: best.kind, i: best.i };
}

// ------------------------------------------------------------------ editing
function toNorm(e) {
  const r = cv.getBoundingClientRect();
  return [clamp01((e.clientX - r.left) / r.width), clamp01((e.clientY - r.top) / r.height)];
}

function dedupe(pts) {
  return pts.filter((p, j) => j === 0 || Math.hypot(p[0] - pts[j - 1][0], p[1] - pts[j - 1][1]) > 1e-4);
}

function finishDraft() {
  if (!draft) return;
  const k = draft.kind;
  const pts = dedupe(draft.pts);
  if (POLY.has(k) || k === "no_turn") {
    if (pts.length < 3) { hint("A polygon needs at least 3 points."); return; }
    if (k === "no_turn" && draft.stage === "from") {
      draft = { kind: k, pts: [], stage: "to", from: pts };
      hint(HINTS.no_turn_to);
      renderTools();
      draw();
      return;
    }
    if (k === "no_turn") {
      const name = $("ntName").value.trim() || `no turn ${zones.no_turn.length + 1}`;
      zones.no_turn.push({ name, from: draft.from, to: pts });
      $("ntName").value = "";
    } else zones[k].push(pts);
  } else if (k === "solid_lines") {
    if (pts.length < 2) { hint("A solid line needs at least 2 points."); return; }
    zones.solid_lines.push(pts);
  } else if (k === "stop_lines") {
    if (pts.length < 2) return;
    const two = pts.slice(0, 2);
    zones.stop_lines.push({ points: two, dir: perpDirs(two)[0] });
  }
  sel = { kind: k, i: zones[k].length - 1 };
  draft = null;
  msg(`Added ${itemName(k, sel.i)}.`);
  setHint();
  commit();
}

function cancelDraft() {
  draft = null;
  setHint();
  renderTools();
  draw();
}

function removeItem(kind, i) {
  const name = itemName(kind, i);
  zones[kind].splice(i, 1);
  if (sel && sel.kind === kind) sel = sel.i === i ? null : sel.i > i ? { kind, i: sel.i - 1 } : sel;
  msg(`Deleted ${name}.`);
  commit();
}

function select(kind, i) {
  sel = kind ? { kind, i } : null;
  renderList();
  renderTools();
  draw();
}

function flip() {
  if (sel?.kind !== "stop_lines") return;
  const sl = zones.stop_lines[sel.i];
  sl.dir = normDeg(sl.dir + 180);
  msg(`Flipped ${itemName("stop_lines", sel.i)}: dir ${jnum(sl.dir, 1)}°.`);
  commit();
}

cv.addEventListener("pointerdown", (e) => {
  if (e.pointerType === "mouse" && e.button !== 0) return;
  const pt = toNorm(e);
  const px = P(pt);
  const now = performance.now();
  const dbl = now - lastDown.t < 400 && Math.hypot(e.clientX - lastDown.x, e.clientY - lastDown.y) < 10;
  lastDown = { t: dbl ? 0 : now, x: e.clientX, y: e.clientY };
  e.preventDefault();
  cv.focus({ preventScroll: true }); // so Enter / Esc / Delete reach the editor after typing a rule name
  if (tool === "select") {
    const vh = vertexHit(px);
    if (vh) {
      drag = { ...vh, id: e.pointerId };
      cv.setPointerCapture(e.pointerId);
      return;
    }
    const hit = itemHit(px);
    select(hit?.kind, hit?.i);
    return;
  }
  cursor = pt;
  if (!draft) draft = { kind: tool, pts: [], stage: tool === "no_turn" ? "from" : null };
  const closed = POLY.has(tool) || tool === "no_turn";
  if (dbl && tool !== "stop_lines") {
    if (draft.pts.length >= (closed ? 3 : 2)) finishDraft();
    return;
  }
  if (closed && draft.pts.length >= 3) {
    const f = P(draft.pts[0]);
    if (Math.hypot(f[0] - px[0], f[1] - px[1]) <= 10) { finishDraft(); return; }
  }
  draft.pts.push(pt);
  renderTools();
  if (tool === "stop_lines" && draft.pts.length === 2) { finishDraft(); return; }
  draw();
});

cv.addEventListener("pointermove", (e) => {
  const pt = toNorm(e);
  if (drag && e.pointerId === drag.id) {
    const item = zones[drag.kind][drag.i];
    const pts = drag.kind === "stop_lines" ? item.points : drag.kind === "no_turn" ? item[drag.part] : item;
    pts[drag.vi] = pt;
    if (drag.kind === "stop_lines") {
      const [a, b] = perpDirs(item.points);
      item.dir = angDiff(a, item.dir) <= angDiff(b, item.dir) ? a : b;
    }
    requestDraw();
    return;
  }
  if (draft) { cursor = pt; requestDraw(); }
  if (tool === "select") cv.style.cursor = vertexHit(P(pt)) ? "move" : itemHit(P(pt)) ? "pointer" : "default";
});

function endDrag(e) {
  if (!drag || e.pointerId !== drag.id) return;
  drag = null;
  commit();
}
cv.addEventListener("pointerup", endDrag);
cv.addEventListener("pointercancel", endDrag);
cv.addEventListener("pointerleave", () => { if (draft && !drag) { cursor = null; requestDraw(); } });

window.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.closest?.("input, textarea, select, [contenteditable='true']")) return;
  if (e.key === "Escape") {
    if (draft) cancelDraft();
    else if (sel) select(null);
  } else if (e.key === "Enter" && draft) {
    e.preventDefault();
    finishDraft();
  } else if (e.key === "Delete" || e.key === "Backspace") {
    if (draft && draft.pts.length) { e.preventDefault(); draft.pts.pop(); renderTools(); draw(); }
    else if (sel && !draft) { e.preventDefault(); removeItem(sel.kind, sel.i); }
  }
});

// ------------------------------------------------------------------ panels
function hint(text) { $("hint").textContent = text; }
function setHint() {
  if (tool === "select") hint(HINTS.select);
  else if (tool === "no_turn") hint(draft?.stage === "to" ? HINTS.no_turn_to : HINTS.no_turn_from);
  else hint(POLY.has(tool) ? HINTS.poly : HINTS[tool]);
}

function setTool(t) {
  if (draft) draft = null;
  tool = t;
  cursor = null;
  cv.classList.toggle("select", t === "select");
  cv.style.cursor = "";
  setHint();
  renderTools();
  draw();
}

function renderTools() {
  const box = $("tools");
  box.replaceChildren(
    el("button", { type: "button", "aria-pressed": String(tool === "select"), onclick: () => setTool("select") }, "Select / edit"),
    ...KINDS.map((k) => el("button", {
      type: "button", "aria-pressed": String(tool === k), onclick: () => setTool(k),
    }, el("span", { class: `ksw k-${k === "no_turn" ? "no_turn_from" : k}${k === "solid_lines" || k === "stop_lines" ? " line" : ""}`, "aria-hidden": "true" }), META[k].tool)));
  $("ntWrap").hidden = tool !== "no_turn";
  $("finish").disabled = !(draft && draft.kind !== "stop_lines" && draft.pts.length >= (draft.kind === "solid_lines" ? 2 : 3));
  $("cancel").disabled = !draft;
  $("flip").disabled = sel?.kind !== "stop_lines";
  $("del").disabled = !sel;
}

function renderList() {
  const box = $("list");
  box.replaceChildren();
  for (const k of KINDS) {
    const items = zones[k];
    const isLine = k === "solid_lines" || k === "stop_lines";
    const group = el("div", { class: "kgroup" }, el("div", { class: "khead" },
      el("span", { class: `ksw k-${k === "no_turn" ? "no_turn_from" : k}${isLine ? " line" : ""}`, "aria-hidden": "true" }),
      META[k].title, el("span", { class: "pill num", title: "items" }, items.length)));
    items.forEach((item, i) => {
      const isSel = sel && sel.kind === k && sel.i === i;
      let detail = "";
      if (POLY.has(k)) detail = `${item.length} pts`;
      else if (k === "solid_lines") detail = `${item.length} pts`;
      else if (k === "stop_lines") detail = `dir ${jnum(item.dir, 1)}°`;
      const row = el("div", { class: `krow${isSel ? " sel" : ""}` },
        el("button", { class: "kname", type: "button", "aria-pressed": String(!!isSel), onclick: () => select(k, i) },
          itemName(k, i), el("span", { class: "muted small" }, detail)));
      if (k === "no_turn") {
        row.append(el("input", {
          type: "text", value: item.name, "aria-label": `Name of ${itemName(k, i)}`,
          onfocus: () => { if (!isSel) select(k, i); },
          onchange: (e) => { item.name = e.target.value.trim(); save(); $("preview").textContent = exportText(); },
        }));
      }
      if (k === "stop_lines") row.append(el("button", { class: "mini", type: "button", onclick: () => { select(k, i); flip(); } }, "Flip"));
      row.append(el("button", { class: "mini", type: "button", "aria-label": `Delete ${itemName(k, i)}`, onclick: () => removeItem(k, i) }, "Delete"));
      group.append(row);
    });
    if (!items.length) group.append(el("div", { class: "none" }, "none"));
    box.append(group);
  }
}

function renderLegend() {
  const item = (cls, text, line = false) => el("span", { class: "item", role: "listitem" },
    el("span", { class: `ksw ${cls}${line ? " line" : ""}`, "aria-hidden": "true" }), text);
  $("legend").replaceChildren(
    item("k-crosswalks", "crosswalk"), item("k-intersection", "intersection"), item("k-u_turn_allowed", "u-turn allowed"),
    item("k-ignore", "ignore"), item("k-solid_lines", "solid line", true), item("k-stop_lines", "stop line + travel direction", true),
    item("k-no_turn_from", "no-turn from"), item("k-no_turn_to", "no-turn to"));
}

// ------------------------------------------------------------------ backgrounds
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(`could not load ${url}`));
    im.src = url;
  });
}

function setBase(src, w, h, label) {
  baseBg = { src, w, h, label };
  showDirs = false;
  $("dirs").checked = false;
  updateBgNote();
  layout();
}

function updateBgNote() {
  const b = bg();
  $("bgMsg").hidden = !!b;
  $("bgNote").textContent = b ? `Background: ${b.label} (${b.w}×${b.h}).` : "";
}

async function tryDefaultBackground() {
  try {
    const im = await loadImage("data/scene/background.jpg");
    if (!baseBg) setBase(im, im.naturalWidth, im.naturalHeight, "data/scene/background.jpg");
  } catch {
    updateBgNote();
  }
}

$("dirs").addEventListener("change", async (e) => {
  if (!e.target.checked) { showDirs = false; updateBgNote(); layout(); return; }
  if (dirsBg === null) {
    try {
      const im = await loadImage("data/scene/directions.jpg");
      dirsBg = { src: im, w: im.naturalWidth, h: im.naturalHeight, label: "data/scene/directions.jpg (learned lane directions)" };
    } catch {
      dirsBg = false;
    }
  }
  if (!dirsBg) {
    e.target.checked = false;
    e.target.disabled = true;
    msg("No learned lane directions image at data/scene/directions.jpg.", true);
    return;
  }
  showDirs = true;
  updateBgNote();
  layout();
});

$("openImg").addEventListener("click", () => $("imgFile").click());
$("imgFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  try {
    const im = await loadImage(URL.createObjectURL(f));
    setBase(im, im.naturalWidth, im.naturalHeight, f.name);
    msg(`Background set to ${f.name}.`);
  } catch {
    msg(`Could not open ${f.name} as an image.`, true);
  }
});

const gv = $("gv");
$("openVid").addEventListener("click", () => $("vidFile").click());
$("vidFile").addEventListener("change", (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  if (gv.src) URL.revokeObjectURL(gv.src);
  gv.dataset.name = f.name;
  gv.src = URL.createObjectURL(f);
  $("grab").hidden = false;
});
gv.addEventListener("loadedmetadata", () => {
  $("gs").max = String(Math.max(0, gv.duration - 0.05));
  $("gs").value = "0";
  gv.currentTime = 0;
});
gv.addEventListener("error", () => msg(`This browser can't play ${gv.dataset.name || "that video"}.`, true));
$("gs").addEventListener("input", () => { gv.currentTime = +$("gs").value; $("gt").textContent = fmtTime(+$("gs").value); });
$("useFrame").addEventListener("click", () => {
  if (!gv.videoWidth) { msg("The video is not ready yet.", true); return; }
  const off = document.createElement("canvas");
  off.width = gv.videoWidth;
  off.height = gv.videoHeight;
  off.getContext("2d").drawImage(gv, 0, 0);
  setBase(off, off.width, off.height, `frame at ${fmtTime(gv.currentTime)} of ${gv.dataset.name}`);
  $("grab").hidden = true;
  msg("Background set from the video frame.");
});
$("closeGrab").addEventListener("click", () => { $("grab").hidden = true; gv.pause(); });

// ------------------------------------------------------------------ save / load
$("export").addEventListener("click", () => { download(exportText(), "zones.json"); msg("Downloaded zones.json — save it as weights/zones.json."); });
$("copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(exportText()); msg("Copied zones.json."); }
  catch { msg("The clipboard is not available here; use Download instead.", true); }
});
$("import").addEventListener("click", () => $("zonesFile").click());
$("zonesFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  try {
    const { z, skipped } = parseZones(JSON.parse(await f.text()));
    zones = z;
    sel = null;
    draft = null;
    const n = KINDS.reduce((s, k) => s + zones[k].length, 0);
    msg(`Imported ${n} items from ${f.name}${skipped ? ` (${skipped} invalid skipped)` : ""}.`);
    commit();
  } catch (err) {
    msg(`Could not import ${f.name}: ${err.message}`, true);
  }
});
$("clear").addEventListener("click", () => {
  const n = KINDS.reduce((s, k) => s + zones[k].length, 0);
  if (!n || !confirm(`Delete all ${n} zones?`)) return;
  zones = emptyZones();
  sel = null;
  msg("Cleared all zones.");
  commit();
});
$("finish").addEventListener("click", finishDraft);
$("cancel").addEventListener("click", cancelDraft);
$("flip").addEventListener("click", flip);
$("del").addEventListener("click", () => { if (sel) removeItem(sel.kind, sel.i); });

// ------------------------------------------------------------------ init
readTokens();
restore();
renderLegend();
setTool("select");
renderList();
$("preview").textContent = exportText();
layout();
tryDefaultBackground();

const retheme = () => { readTokens(); draw(); };
window.addEventListener("themechange", retheme);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
new ResizeObserver(() => { if (Math.max(100, $("stage").clientWidth) !== size.w) layout(); }).observe($("stage"));
