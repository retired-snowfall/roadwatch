// Small SVG chart components: swimlane timeline, line chart, bar chart.
// Thin marks, hairline grid, hover tooltips, click-to-seek, playhead; redrawn on resize and theme change.
import { CLASS_ORDER, className, famColor, fmtTime } from "./common.js";

const NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function niceStep(range, target) {
  const raw = range / Math.max(1, target);
  const mag = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

class Base {
  constructor(container) {
    container._chart?.destroy();      // one live chart per container
    container._chart = this;
    this.container = container;
    this.container.classList.add("chart");
    this.container.replaceChildren();
    this.tip = document.createElement("div");
    this.tip.className = "tooltip";
    this.tip.hidden = true;
    this.container.append(this.tip);
    this.time = null;
    this.redraw = () => this.render();
    this.ro = new ResizeObserver(this.redraw);
    this.ro.observe(container);
    this.mq = matchMedia("(prefers-color-scheme: dark)");
    window.addEventListener("themechange", this.redraw);
    this.mq.addEventListener("change", this.redraw);
  }

  destroy() {
    this.ro.disconnect();
    window.removeEventListener("themechange", this.redraw);
    this.mq.removeEventListener("change", this.redraw);
    this.container.replaceChildren();
  }

  width() {
    return Math.max(280, this.container.clientWidth);
  }

  showTip(html, x, y) {
    this.tip.innerHTML = html;
    this.tip.hidden = false;
    const w = this.tip.offsetWidth;
    const cw = this.container.clientWidth;
    this.tip.style.left = `${Math.min(Math.max(0, x + 12), cw - w)}px`;
    this.tip.style.top = `${Math.max(0, y - 10)}px`;
  }

  hideTip() {
    this.tip.hidden = true;
  }

  mount(svg) {
    this.svg?.remove();
    this.svg = svg;
    this.container.prepend(svg);
  }
}

// ---------------------------------------------------------------- swimlane timeline
export class Timeline extends Base {
  /** events: [[s, e, label]], duration: seconds, onSeek(t), preds: optional second set drawn faded */
  constructor(container, { events, duration, onSeek, preds = null, rows = null }) {
    super(container);
    Object.assign(this, { events, duration, onSeek, preds });
    this.rows = rows || CLASS_ORDER.filter((c) => events.some((e) => e[2] === c) || (preds || []).some((e) => e[2] === c));
    this.render();
  }

  render() {
    const W = this.width();
    const narrow = W < 560;
    const labelW = narrow ? 118 : 178;
    const rowH = this.preds ? 30 : 24;
    const top = 6;
    const H = top + Math.max(1, this.rows.length) * rowH + 24;
    const x0 = labelW, x1 = W - 8;
    const sx = (t) => x0 + (t / Math.max(this.duration, 1e-6)) * (x1 - x0);
    this.sx = sx; this.x0 = x0; this.x1 = x1;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Event timeline, ${this.events.length} events over ${fmtTime(this.duration)}` });
    const step = niceStep(this.duration, narrow ? 4 : 8);
    for (let t = 0; t <= this.duration + 1e-6; t += step) {
      if (this.rows.length) svg.append(svgEl("line", { class: "gridline", x1: sx(t), x2: sx(t), y1: top, y2: H - 22 }));
      const lab = svgEl("text", { class: "tick", x: sx(t), y: H - 6, "text-anchor": "middle" });
      lab.textContent = fmtTime(t).replace(/\.\d$/, "");
      svg.append(lab);
    }
    if (!this.rows.length) {
      const t = svgEl("text", { class: "tick", x: (x0 + x1) / 2, y: top + 16, "text-anchor": "middle" });
      t.textContent = "no events detected";
      svg.append(t);
    }
    this.bars = [];
    this.rows.forEach((label, i) => {
      const y = top + i * rowH;
      const name = svgEl("text", { class: "rowlabel", x: 0, y: y + 15 });
      name.textContent = narrow ? className(label).replace("Pedestrian", "Ped.") : className(label);
      svg.append(name);
      svg.append(svgEl("line", { class: "gridline", x1: x0, x2: x1, y1: y + rowH - 1, y2: y + rowH - 1 }));
      const color = famColor(label);
      const draw = (list, yy, h, opacity, kind) => {
        for (const ev of list.filter((e) => e[2] === label)) {
          const bx = sx(ev[0]);
          const bw = Math.max(3, sx(ev[1]) - bx);
          svg.append(svgEl("rect", { x: bx, y: yy, width: bw, height: h, rx: Math.min(4, h / 2),
            fill: color, "fill-opacity": opacity }));
          this.bars.push({ x: bx, w: bw, y: yy, h, ev, kind });
        }
      };
      if (this.preds) {
        draw(this.events, y + 3, 11, 1, "truth");
        draw(this.preds, y + 16, 9, 0.45, "pred");
      } else {
        draw(this.events, y + 5, 13, 1, "event");
      }
    });
    this.playhead = svgEl("line", { class: "playhead", x1: x0, x2: x0, y1: top, y2: H - 22, visibility: "hidden" });
    svg.append(this.playhead);
    const hit = svgEl("rect", { x: x0, y: 0, width: x1 - x0, height: H, fill: "transparent", style: "cursor:pointer" });
    svg.append(hit);
    const locate = (evt) => {
      const r = svg.getBoundingClientRect();
      const px = ((evt.clientX - r.left) / r.width) * W;
      const py = ((evt.clientY - r.top) / r.height) * H;
      return { px, py, bar: this.bars.find((b) => px >= b.x - 3 && px <= b.x + b.w + 3 && py >= b.y - 3 && py <= b.y + b.h + 3),
        cx: evt.clientX - this.container.getBoundingClientRect().left, cy: evt.clientY - this.container.getBoundingClientRect().top };
    };
    hit.addEventListener("pointermove", (evt) => {
      const { bar, cx, cy } = locate(evt);
      if (!bar) return this.hideTip();
      const [s, e, lab] = bar.ev;
      this.showTip(`<div class="tt-title">${className(lab)}${bar.kind === "pred" ? " (prediction)" : ""}</div>
        <div class="tt-row">${fmtTime(s)} – ${fmtTime(e)} · ${(e - s).toFixed(1)} s</div>`, cx, cy);
    });
    hit.addEventListener("pointerleave", () => this.hideTip());
    hit.addEventListener("click", (evt) => {
      const { px, bar } = locate(evt);
      const t = bar ? bar.ev[0] : ((px - x0) / (x1 - x0)) * this.duration;
      this.onSeek?.(Math.max(0, Math.min(this.duration, t)));
    });
    this.mount(svg);
    if (this.time !== null) this.setTime(this.time);
  }

  setTime(t) {
    this.time = t;
    if (!this.playhead) return;
    const x = this.sx(t);
    this.playhead.setAttribute("x1", x);
    this.playhead.setAttribute("x2", x);
    this.playhead.setAttribute("visibility", "visible");
  }
}

// ---------------------------------------------------------------- line chart
export class LineChart extends Base {
  /** series: [{name, points: [[x, y]], color}] ; opts: yMax, yLabel, threshold, xFormat, onSeek, height */
  constructor(container, { series, yMax = null, yLabel = "", threshold = null, xFormat = fmtTime, onSeek = null,
    height = 180, yFormat = (v) => v.toFixed(2), area = true }) {
    super(container);
    Object.assign(this, { series, yMax, yLabel, threshold, xFormat, onSeek, height, yFormat, area });
    this.render();
  }

  render() {
    const W = this.width();
    const H = this.height;
    const pad = { l: 40, r: 12, t: 10, b: 24 };
    const xs = this.series.flatMap((s) => s.points.map((p) => p[0]));
    const ys = this.series.flatMap((s) => s.points.map((p) => p[1]));
    const xMin = xs.length ? Math.min(...xs) : 0, xMax = xs.length ? Math.max(...xs) : 1;
    let yMax = this.yMax ?? Math.max(1e-6, ...ys);
    const yStep = niceStep(yMax, 4);
    if (this.yMax === null) yMax = Math.ceil(yMax / yStep) * yStep;
    const sx = (x) => pad.l + ((x - xMin) / Math.max(xMax - xMin, 1e-6)) * (W - pad.l - pad.r);
    const sy = (y) => H - pad.b - (y / yMax) * (H - pad.t - pad.b);
    this.sx = sx; this.H = H; this.pad = pad;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": this.yLabel || "line chart" });
    for (let y = 0; y <= yMax + 1e-9; y += yStep) {
      svg.append(svgEl("line", { class: y === 0 ? "axis" : "gridline", x1: pad.l, x2: W - pad.r, y1: sy(y), y2: sy(y) }));
      const t = svgEl("text", { class: "tick", x: pad.l - 6, y: sy(y) + 4, "text-anchor": "end" });
      t.textContent = this.yFormat(y);
      svg.append(t);
    }
    const xStep = niceStep(xMax - xMin, W < 560 ? 4 : 8);
    for (let x = Math.ceil(xMin / xStep) * xStep; x <= xMax + 1e-9; x += xStep) {
      const t = svgEl("text", { class: "tick", x: sx(x), y: H - 6, "text-anchor": "middle" });
      t.textContent = this.xFormat(x).replace(/\.\d$/, "");
      svg.append(t);
    }
    if (this.threshold !== null) {
      svg.append(svgEl("line", { class: "threshold", x1: pad.l, x2: W - pad.r, y1: sy(this.threshold), y2: sy(this.threshold) }));
      const t = svgEl("text", { class: "tick", x: W - pad.r, y: sy(this.threshold) - 4, "text-anchor": "end" });
      t.textContent = `alarm threshold ${this.threshold}`;
      svg.append(t);
    }
    for (const s of this.series) {
      if (!s.points.length) continue;
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
      if (this.area && this.series.length === 1) {
        const last = s.points[s.points.length - 1], first = s.points[0];
        svg.append(svgEl("path", { d: `${d}L${sx(last[0])},${sy(0)}L${sx(first[0])},${sy(0)}Z`, fill: s.color, "fill-opacity": 0.1 }));
      }
      svg.append(svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }
    this.cross = svgEl("line", { class: "crosshair", y1: pad.t, y2: H - pad.b, visibility: "hidden" });
    this.dots = this.series.map((s) => {
      const c = svgEl("circle", { r: 4, fill: s.color, stroke: token("--surface"), "stroke-width": 2, visibility: "hidden" });
      svg.append(c);
      return c;
    });
    svg.append(this.cross);
    this.playhead = svgEl("line", { class: "playhead", y1: pad.t, y2: H - pad.b, visibility: "hidden" });
    svg.append(this.playhead);
    const hit = svgEl("rect", { x: pad.l, y: 0, width: W - pad.l - pad.r, height: H, fill: "transparent",
      style: this.onSeek ? "cursor:pointer" : "" });
    svg.append(hit);
    const xAt = (evt) => {
      const r = svg.getBoundingClientRect();
      const px = ((evt.clientX - r.left) / r.width) * W;
      return xMin + ((px - pad.l) / (W - pad.l - pad.r)) * (xMax - xMin);
    };
    hit.addEventListener("pointermove", (evt) => {
      const x = xAt(evt);
      const rows = [];
      this.series.forEach((s, i) => {
        if (!s.points.length) return;
        let k = s.points.findIndex((p) => p[0] >= x);
        if (k < 0) k = s.points.length - 1;
        const p = s.points[k];
        this.dots[i].setAttribute("cx", sx(p[0]));
        this.dots[i].setAttribute("cy", sy(p[1]));
        this.dots[i].setAttribute("visibility", "visible");
        rows.push(`<div class="tt-row"><span class="swatch" style="background:${s.color}"></span>${s.name}: <b>${this.yFormat(p[1])}</b></div>`);
      });
      this.cross.setAttribute("x1", sx(x));
      this.cross.setAttribute("x2", sx(x));
      this.cross.setAttribute("visibility", "visible");
      const cr = this.container.getBoundingClientRect();
      this.showTip(`<div class="tt-title">${this.xFormat(x)}</div>${rows.join("")}`, evt.clientX - cr.left, evt.clientY - cr.top);
    });
    hit.addEventListener("pointerleave", () => {
      this.hideTip();
      this.cross.setAttribute("visibility", "hidden");
      this.dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    });
    if (this.onSeek) hit.addEventListener("click", (evt) => this.onSeek(Math.max(xMin, Math.min(xMax, xAt(evt)))));
    this.mount(svg);
    if (this.time !== null) this.setTime(this.time);
  }

  setTime(t) {
    this.time = t;
    if (!this.playhead) return;
    this.playhead.setAttribute("x1", this.sx(t));
    this.playhead.setAttribute("x2", this.sx(t));
    this.playhead.setAttribute("visibility", "visible");
  }
}

// ---------------------------------------------------------------- bar chart (horizontal, one series)
export class BarChart extends Base {
  /** bars: [{label, value, color?, note?}] sorted by the caller */
  constructor(container, { bars, valueFormat = (v) => String(v), color = null }) {
    super(container);
    Object.assign(this, { bars, valueFormat, color });
    this.render();
  }

  render() {
    const W = this.width();
    const rowH = 26;
    const labelW = Math.min(190, W * 0.4);
    const H = this.bars.length * rowH + 4;
    const vMax = Math.max(1e-6, ...this.bars.map((b) => b.value));
    const x0 = labelW, x1 = W - 48;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${Math.max(H, 24)}`, role: "img", "aria-label": "bar chart" });
    this.bars.forEach((b, i) => {
      const y = i * rowH;
      const name = svgEl("text", { class: "rowlabel", x: 0, y: y + 17 });
      name.textContent = b.label;
      svg.append(name);
      const w = Math.max(2, ((x1 - x0) * b.value) / vMax);
      const r = svgEl("path", { d: roundedRight(x0, y + 6, w, 14, 4), fill: b.color || this.color || token("--accent") });
      svg.append(r);
      const v = svgEl("text", { class: "tick", x: x0 + w + 6, y: y + 17 });
      v.textContent = this.valueFormat(b.value);
      svg.append(v);
      const hit = svgEl("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" });
      hit.addEventListener("pointermove", (evt) => {
        const cr = this.container.getBoundingClientRect();
        this.showTip(`<div class="tt-title">${b.label}</div><div class="tt-row">${this.valueFormat(b.value)}${b.note ? ` · ${b.note}` : ""}</div>`,
          evt.clientX - cr.left, evt.clientY - cr.top);
      });
      hit.addEventListener("pointerleave", () => this.hideTip());
      svg.append(hit);
    });
    this.mount(svg);
  }
}

// vertical histogram (one series): bars grow from the baseline, 4px rounded tops, 2px gaps
export class Histogram extends Base {
  constructor(container, { edges, counts, xLabel = "", height = 170, color = null }) {
    super(container);
    Object.assign(this, { edges, counts, xLabel, height, color });
    this.render();
  }

  render() {
    const W = this.width(), H = this.height;
    const pad = { l: 40, r: 8, t: 8, b: 34 };
    const n = this.counts.length;
    const cMax = Math.max(1, ...this.counts);
    const step = niceStep(cMax, 4);
    const yMax = Math.ceil(cMax / step) * step;
    const bw = (W - pad.l - pad.r) / Math.max(n, 1);
    const sy = (v) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": this.xLabel });
    for (let y = 0; y <= yMax; y += step) {
      svg.append(svgEl("line", { class: y === 0 ? "axis" : "gridline", x1: pad.l, x2: W - pad.r, y1: sy(y), y2: sy(y) }));
      const t = svgEl("text", { class: "tick", x: pad.l - 6, y: sy(y) + 4, "text-anchor": "end" });
      t.textContent = y.toLocaleString();
      svg.append(t);
    }
    const color = this.color || token("--accent");
    this.counts.forEach((c, i) => {
      const x = pad.l + i * bw + 1;
      const w = Math.min(24, bw - 2);
      const h = Math.max(0, sy(0) - sy(c));
      if (h > 0) svg.append(svgEl("path", { d: roundedTop(x + (bw - 2 - w) / 2, sy(c), w, h, 4), fill: color }));
      const hit = svgEl("rect", { x: pad.l + i * bw, y: pad.t, width: bw, height: H - pad.t - pad.b, fill: "transparent" });
      hit.addEventListener("pointermove", (evt) => {
        const cr = this.container.getBoundingClientRect();
        this.showTip(`<div class="tt-title">${this.edges[i]}–${this.edges[i + 1]}</div><div class="tt-row">${c.toLocaleString()} samples</div>`,
          evt.clientX - cr.left, evt.clientY - cr.top);
      });
      hit.addEventListener("pointerleave", () => this.hideTip());
      svg.append(hit);
    });
    const every = Math.ceil(n / (W < 560 ? 4 : 8));
    for (let i = 0; i <= n; i += every) {
      const t = svgEl("text", { class: "tick", x: pad.l + i * bw, y: H - pad.b + 14, "text-anchor": "middle" });
      t.textContent = this.edges[i];
      svg.append(t);
    }
    const xl = svgEl("text", { class: "tick", x: (W + pad.l) / 2, y: H - 4, "text-anchor": "middle" });
    xl.textContent = this.xLabel;
    svg.append(xl);
    this.mount(svg);
  }
}

function roundedRight(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

function roundedTop(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
