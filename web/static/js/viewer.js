// Result viewer: video + canvas overlays + synced timeline, risk curve and events table.
import { LineChart, Timeline } from "./charts.js";
import { CLASSES, FAMILIES, className, el, famColor, fmtTime, legend } from "./common.js";

function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return { vehicle: "#e8e6df", two_wheeler: "#9ec5f4", person: "#b8f5c9", animal: "#e2b6ff", ink: g("--ink"),
    accent: g("--accent"), critical: g("--critical") };
}

class TrackIndex {
  constructor(tracks) {
    this.tracks = tracks.filter((tr) => tr.t.length);
  }

  // boxes at time t, linearly interpolated between analysed frames (gaps > 0.6 s are not bridged)
  *at(t) {
    for (const tr of this.tracks) {
      const ts = tr.t;
      if (t < ts[0] - 0.05 || t > ts[ts.length - 1] + 0.05) continue;
      let lo = 0, hi = ts.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (ts[mid] < t) lo = mid + 1; else hi = mid;
      }
      const k = lo;
      if (k === 0 || ts[k] === t) { yield [tr, tr.box[k], k]; continue; }
      const t0 = ts[k - 1], t1 = ts[k];
      if (t1 - t0 > 0.6) continue;
      const a = (t - t0) / (t1 - t0);
      const b0 = tr.box[k - 1], b1 = tr.box[k];
      yield [tr, b0.map((v, i) => v * (1 - a) + b1[i] * a), k];
    }
  }
}

export class ResultViewer {
  constructor(root, { result, videoUrl, annotatedUrl = null, title = "" }) {
    this.root = root;
    this.result = result;
    this.index = new TrackIndex(result.tracks || []);
    this.hot = new Map();   // track id -> [label, start, end]
    for (const c of result.candidates || []) for (const tid of c.tracks) {
      (this.hot.get(tid) || this.hot.set(tid, []).get(tid)).push([c.label, c.start, c.end]);
    }
    this.show = { boxes: true, trails: true, scene: false };
    this.build(videoUrl, annotatedUrl, title);
  }

  build(videoUrl, annotatedUrl, title) {
    const r = this.result;
    this.video = el("video", { src: videoUrl, controls: "", playsinline: "", preload: "metadata", muted: "" });
    this.canvas = el("canvas");
    const player = el("div", { class: "player" }, this.video, this.canvas);
    player.style.setProperty("--ar", `${r.width} / ${r.height}`);   // stable layout before the video loads
    const toggles = ["boxes", "trails", "scene"].map((k) => {
      const box = el("input", { type: "checkbox" });
      box.checked = this.show[k];
      box.addEventListener("change", () => { this.show[k] = box.checked; this.draw(); });
      return el("label", {}, box, { boxes: "Tracked objects", trails: "Trails", scene: "Scene layers" }[k]);
    });
    this.nowEvents = el("div", { class: "now-events", "aria-live": "polite" });
    const downloads = el("div", { class: "player-controls" },
      el("a", { href: "#", onclick: (e) => { e.preventDefault(); this.downloadJSON(); } }, "Download events JSON"),
      annotatedUrl ? el("a", { href: annotatedUrl, download: "" }, "Annotated video (mp4)") : "");
    this.timelineBox = el("div");
    this.riskBox = el("div");
    this.tableBox = el("div", { class: "table-wrap" });
    const present = new Set(r.events.map((e) => CLASSES[e[2]]?.family));
    const families = Object.keys(FAMILIES).filter((f) => present.has(f));   // fixed order, never by appearance
    this.root.replaceChildren(
      el("div", { class: "viewer" },
        title ? el("h3", {}, title) : "",
        player,
        el("div", { class: "player-controls" }, toggles),
        this.nowEvents,
        el("div", { class: "card" }, el("h3", {}, "Event timeline"),
          el("p", { class: "muted small" }, "One row per class; click a bar to jump to it."),
          families.length ? legend(families) : "", this.timelineBox),
        el("div", { class: "card" }, el("h3", {}, "Accident risk (Part B, causal)"),
          el("p", { class: "muted small" }, "P(accident starts within 5 s) computed frame by frame from past frames only."),
          this.riskBox),
        el("div", { class: "card" }, el("h3", {}, `Events (${r.events.length})`), this.tableBox, downloads)));
    const seek = (t) => { this.video.currentTime = t; this.video.play?.().catch(() => {}); };
    this.timeline = new Timeline(this.timelineBox, { events: r.events, duration: r.duration, onSeek: seek });
    this.risk = new LineChart(this.riskBox, { series: [{ name: "risk", points: r.risk || [], color: getComputedStyle(document.documentElement).getPropertyValue("--fam-collision").trim() }],
      yMax: 1, threshold: 0.5, onSeek: seek, height: 160 });
    this.renderTable(seek);
    const tick = () => {
      this.sync();
      if (!this.video.paused && !this.video.ended) this.raf = requestAnimationFrame(tick);
    };
    this.video.addEventListener("play", () => { cancelAnimationFrame(this.raf); this.raf = requestAnimationFrame(tick); });
    for (const ev of ["seeked", "loadeddata", "timeupdate"]) this.video.addEventListener(ev, () => this.sync());
    new ResizeObserver(() => this.draw()).observe(player);
    window.addEventListener("themechange", () => { this.renderTable(seek); this.draw(); });
  }

  renderTable(seek) {
    const byEvent = (e) => (this.result.candidates || []).filter((c) => c.label === e[2] && c.start < e[1] + 0.5 && c.end > e[0] - 0.5);
    const rows = this.result.events.map((e) => {
      const cands = byEvent(e);
      const tids = [...new Set(cands.flatMap((c) => c.tracks))];
      const tr = el("tr", { class: "clickable", tabindex: "0", onclick: () => seek(e[0]),
        onkeydown: (k) => { if (k.key === "Enter") seek(e[0]); } },
        el("td", {}, el("span", { class: `swatch fam-${CLASSES[e[2]]?.family}` }), className(e[2])),
        el("td", { class: "num" }, fmtTime(e[0])), el("td", { class: "num" }, fmtTime(e[1])),
        el("td", { class: "num" }, `${(e[1] - e[0]).toFixed(1)} s`),
        el("td", { class: "small muted" }, tids.length ? `objects ${tids.join(", ")}` : ""));
      return tr;
    });
    this.tableBox.replaceChildren(rows.length
      ? el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Class"), el("th", { class: "num" }, "Start"),
          el("th", { class: "num" }, "End"), el("th", { class: "num" }, "Length"), el("th", {}, "Evidence"))),
        el("tbody", {}, rows))
      : el("div", { class: "empty" }, "No events in this video."));
  }

  sync() {
    const t = this.video.currentTime || 0;
    this.timeline.setTime(t);
    this.risk.setTime(t);
    const active = this.result.events.filter((e) => e[0] <= t && t <= e[1]);
    const key = active.map((e) => e.join()).join("|");
    if (key !== this.activeKey) {
      this.activeKey = key;
      this.nowEvents.replaceChildren(...active.map((e) => el("span", { class: "pill" },
        el("span", { class: `swatch fam-${CLASSES[e[2]]?.family}` }), className(e[2]))));
    }
    this.draw();
  }

  draw() {
    const v = this.video, c = this.canvas, r = this.result;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const ctx = c.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    // the video is letterboxed inside the player box when aspect ratios differ
    const vw = v.videoWidth || r.width, vh = v.videoHeight || r.height;
    const scale = Math.min(c.width / vw, c.height / vh);
    const ox = (c.width - vw * scale) / 2, oy = (c.height - vh * scale) / 2;
    const k = scale * (vw / r.width);
    ctx.setTransform(k, 0, 0, k, ox, oy);
    const T = tokens();
    const t = v.currentTime || 0;
    if (this.show.scene && r.scene) this.drawScene(ctx, T, k);
    if (!this.show.boxes && !this.show.trails) return;
    ctx.lineJoin = "round";
    for (const [tr, b, idx] of this.index.at(t)) {
      const hot = (this.hot.get(tr.id) || []).find(([, s, e]) => s - 0.2 <= t && t <= e + 0.2);
      const color = hot ? famColor(hot[0]) : T[tr.group] || T.vehicle;
      if (this.show.trails) {
        ctx.beginPath();
        for (let i = Math.max(0, idx - 15); i <= idx && i < tr.box.length; i++) {
          const bb = tr.box[i];
          const x = (bb[0] + bb[2]) / 2, y = bb[3];
          if (i === Math.max(0, idx - 15)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 / k;
        ctx.globalAlpha = 0.7;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      if (this.show.boxes) {
        ctx.strokeStyle = color;
        ctx.lineWidth = (hot ? 3 : 1.5) / k;
        ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        const label = hot ? `${className(hot[0])} · ${tr.id}` : `${tr.label} ${tr.id}`;
        ctx.font = `${(hot ? 13 : 11) / k}px system-ui, sans-serif`;
        const tw = ctx.measureText(label).width;
        const th = (hot ? 16 : 14) / k;
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(b[0], b[1] - th, tw + 6 / k, th);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, b[0] + 3 / k, b[1] - 4 / k);
      }
    }
  }

  drawScene(ctx, T, k) {
    const r = this.result, s = r.scene;
    const [gw, gh] = s.grid;
    const cw = r.width / gw, ch = r.height / gh;
    ctx.fillStyle = "rgba(57,135,229,0.18)";
    s.road.forEach((row, y) => row.forEach((v, x) => { if (v) ctx.fillRect(x * cw, y * ch, cw, ch); }));
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = 1.5 / k;
    s.direction.forEach((row, y) => row.forEach((d, x) => {
      if (!s.oriented[y][x]) return;
      const a = (d * Math.PI) / 180, cx = (x + 0.5) * cw, cy = (y + 0.5) * ch, L = cw * 0.4;
      ctx.beginPath();
      ctx.moveTo(cx - Math.cos(a) * L, cy - Math.sin(a) * L);
      ctx.lineTo(cx + Math.cos(a) * L, cy + Math.sin(a) * L);
      ctx.lineTo(cx + Math.cos(a + 2.6) * L * 0.5, cy + Math.sin(a + 2.6) * L * 0.5);
      ctx.stroke();
    }));
    ctx.strokeStyle = T.critical;
    ctx.lineWidth = 3 / k;
    for (const sl of s.stop_lines || []) {
      ctx.beginPath();
      ctx.moveTo(sl.a[0], sl.a[1]);
      ctx.lineTo(sl.b[0], sl.b[1]);
      ctx.stroke();
    }
    const poly = (pts, close) => {
      const P = (pts.points || pts).map(([x, y]) => [x * r.width, y * r.height]);
      if (!P.length) return;
      ctx.beginPath();
      P.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (close) ctx.closePath();
      ctx.stroke();
    };
    ctx.lineWidth = 2 / k;
    ctx.strokeStyle = "#ffffff";
    for (const z of s.zones?.crosswalks || []) poly(z, true);
    ctx.strokeStyle = "#f5c518";
    for (const z of s.zones?.solid_lines || []) poly(z, false);
  }

  downloadJSON() {
    const r = this.result;
    const blob = new Blob([JSON.stringify({ video: r.video, duration: r.duration, events: r.events }, null, 1)],
      { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `${(r.video || "video").replace(/\.[^.]+$/, "")}_events.json` });
    document.body.append(a);
    a.click();
    a.remove();
  }
}
