import { CLASS_ORDER, CLASSES, chrome, className, el, fmtTime, getJSON } from "./common.js";
import { ResultViewer } from "./viewer.js";

chrome();
const $ = (id) => document.getElementById(id);
let index = null;
let viewer = null;

function tile(label, value, note = "") {
  return el("div", { class: "card stat" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value),
    note ? el("div", { class: "note" }, note) : "");
}

// How this recording's framing differs from the reference view (the camera is re-aimed between recordings).
function viewTile(reg) {
  if (!reg) return tile("Camera view", "own view", "not registered: analysed without the learned junction");
  const [dx, dy] = reg.shift;
  const same = Math.hypot(dx, dy) < 3 && Math.abs(reg.scale - 1) < 0.005;
  return tile("Camera view", same ? "reference" : `${dx.toFixed(0)}, ${dy.toFixed(0)} px`,
    same ? "the view the junction was learned in" : `shift from the reference view · zoom ${(reg.scale * 100).toFixed(1)} %`);
}

async function show(id, seekTo = null) {
  const v = index.videos.find((x) => x.id === id);
  for (const b of $("tabs").children) b.setAttribute("aria-selected", b.dataset.id === id);
  const base = `data/samples/${id}`;
  const result = await getJSON(`${base}/result.json`);
  $("tiles").replaceChildren(
    tile("Duration", fmtTime(v.duration), `${v.width}×${v.height} @ ${v.fps.toFixed(0)} fps · ${v.lighting}`),
    tile("Events", String(result.events.length), [...new Set(result.events.map((e) => className(e[2])))].slice(0, 3).join(", ") || "none"),
    tile("Peak accident risk", result.risk.length ? Math.max(...result.risk.map((r) => r[1])).toFixed(2) : "–", "alarm at ≥ 0.50"),
    viewTile(result.registration));
  viewer = new ResultViewer($("viewer"), { result, videoUrl: `${base}/preview.mp4`, annotatedUrl: `${base}/annotated.mp4` });
  if (seekTo !== null) {
    viewer.video.addEventListener("loadedmetadata", () => { viewer.video.currentTime = seekTo; }, { once: true });
    $("viewer").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  history.replaceState(null, "", `#${id}`);
}

async function main() {
  try {
    index = await getJSON("data/index.json");
  } catch {
    $("viewer").replaceChildren(el("div", { class: "empty" },
      "Sample results have not been built yet (python tools/build_site.py --videos samples)."));
    return;
  }
  $("tabs").replaceChildren(...index.videos.map((v) => el("button", { role: "tab", "data-id": v.id,
    onclick: () => show(v.id) }, `${v.video} · ${v.n_events} events`)));
  const first = decodeURIComponent(location.hash.slice(1)) || index.videos[0]?.id;
  if (first) show(index.videos.some((v) => v.id === first) ? first : index.videos[0].id);

  const examples = [];
  for (const label of CLASS_ORDER) {
    for (const v of index.videos) {
      const ev = v.events.find((e) => e[2] === label);
      if (ev) { examples.push([label, v, ev]); break; }
    }
  }
  $("examples").replaceChildren(...(examples.length ? examples.map(([label, v, ev]) => el("button", {
    class: "card", style: "text-align:left;cursor:pointer;font:inherit;color:inherit",
    onclick: () => show(v.id, Math.max(0, ev[0] - 1)) },
    el("div", {}, el("span", { class: `swatch fam-${CLASSES[label].family}` }), el("strong", {}, className(label))),
    el("div", { class: "small muted" }, `${v.video} · ${fmtTime(ev[0])}–${fmtTime(ev[1])}`),
    el("img", { src: `data/samples/${v.id}/poster.jpg`, alt: "", loading: "lazy", style: "margin-top:8px;border-radius:6px" })))
    : [el("div", { class: "empty" }, "No events detected on the samples.")]));

  try {
    const failures = await getJSON("data/failures.json");
    $("failures").replaceChildren(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Video"), el("th", {}, "Time"), el("th", {}, "What happened"), el("th", {}, "Why"))),
      el("tbody", {}, failures.map((f) => el("tr", { class: "clickable", onclick: () => show(f.video_id, f.t) },
        el("td", {}, f.video), el("td", { class: "num" }, fmtTime(f.t)), el("td", {}, f.what), el("td", { class: "muted" }, f.why)))))));
  } catch {
    $("failures").replaceChildren(el("div", { class: "empty" }, "Failure analysis is added once the dev labels are complete."));
  }
}
main();
