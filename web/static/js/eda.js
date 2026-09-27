import { Histogram, LineChart } from "./charts.js";
import { chrome, el, fmtTime, getJSON } from "./common.js";

chrome();
const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
// first three categorical slots (validated all-pairs in light and dark)
const SERIES = [["vehicle", "Vehicles", "--fam-collision"], ["two_wheeler", "Two-wheelers", "--fam-signal"],
  ["person", "Pedestrians", "--fam-manoeuvre"]];
const PICTURES = [["heat_vehicles", "Motion heatmap — moving vehicles"], ["heat_stops", "Where vehicles stand still"],
  ["heat_people", "Pedestrian heatmap"], ["trajectories", "Vehicle trajectories, coloured by direction"],
  ["directions", "Learned lane directions (this video)"], ["background", "Median background (empty road)"]];

const DEFAULT_FINDINGS = [
  ["The camera never moves", "A per-pixel median of frames gives a clean empty-road background, and every road-layout "
    + "fact (carriageway, lane directions, stop lines, crossings) can be learned once from the samples and reused on the test videos."],
  ["Perspective changes sizes several-fold", "Near vehicles are many times larger than far ones, so speeds and distances are "
    + "measured in object sizes, and objects under 1.8 % of the frame width are excluded from acceleration-based rules (their boxes jitter)."],
  ["Most frames have no event", "Events are short and rare against minutes of normal traffic, and a class predicted but absent "
    + "from the test set costs a zero in the macro F1, so every rule is tuned for precision."],
  ["Vehicles move several grid cells between analysed frames", "At 5–8 analysed frames per second a car near the camera "
    + "jumps 2–3 cells, so trajectories are rasterised between samples before learning the lane-direction field."],
];

async function main() {
  let index;
  try {
    index = await getJSON("data/index.json");
  } catch {
    $("summary").replaceChildren(el("div", { class: "empty" }, "EDA data has not been built yet (tools/build_site.py)."));
    return;
  }
  let findings = DEFAULT_FINDINGS;
  try { findings = (await getJSON("data/eda_findings.json")).map((f) => [f.title, f.text]); } catch { /* defaults */ }
  $("findings").replaceChildren(...findings.map(([t, x]) => el("div", { class: "card" }, el("h3", {}, t), el("p", { class: "muted" }, x))));

  const results = await Promise.all(index.videos.map((v) => getJSON(`data/samples/${v.id}/result.json`)));
  const rows = results.map((r) => {
    const e = r.eda;
    return el("tr", {}, el("td", {}, r.video), el("td", { class: "num" }, `${e.width}×${e.height}`),
      el("td", { class: "num" }, e.fps.toFixed(2)), el("td", { class: "num" }, fmtTime(e.duration)),
      el("td", { class: "num" }, `${e.size_mb} MB`), el("td", { class: "num" }, `${e.bitrate_mbps} Mb/s`),
      el("td", {}, e.lighting), el("td", { class: "num" }, e.unique_tracks.vehicle),
      el("td", { class: "num" }, e.unique_tracks.two_wheeler), el("td", { class: "num" }, e.unique_tracks.person),
      el("td", { class: "num" }, e.mean_simultaneous.vehicle.toFixed(1)),
      el("td", { class: "num" }, `${Math.round(100 * e.vehicle_stationary_share)} %`));
  });
  const heads = ["Video", "Resolution", "fps", "Length", "Size", "Bitrate", "Lighting", "Vehicles", "Two-wheelers",
    "Pedestrians", "Vehicles in frame (mean)", "Vehicle time standing"];
  $("summary").replaceChildren(el("table", {}, el("thead", {}, el("tr", {}, heads.map((h, i) =>
    el("th", { class: i && i !== 6 ? "num" : "" }, h)))), el("tbody", {}, rows)));

  $("tabs").replaceChildren(...results.map((r, i) => el("button", { role: "tab", "data-i": i,
    onclick: () => showVideo(results, i) }, r.video)));
  $("counts-legend").replaceChildren(...SERIES.map(([, name, v]) => el("span", {},
    el("span", { class: "swatch", style: `background:var(${v})` }), name)));
  showVideo(results, 0);

  const scene = index.scene;
  $("scene").replaceChildren(scene
    ? el("figure", { class: "figure" }, el("img", { src: "data/scene/directions.jpg", alt: "Lane direction field learned from all samples", loading: "lazy" }),
      el("figcaption", {}, `${scene.videos} videos, ${Math.round(scene.seconds / 60)} minutes of traffic, ${scene.stop_lines} stop lines.`))
    : el("div", { class: "empty" }, "Run tools/calibrate.py to build the scene prior."));
}

function showVideo(results, i) {
  const r = results[i];
  const e = r.eda;
  for (const b of $("tabs").children) b.setAttribute("aria-selected", String(Number(b.dataset.i) === i));
  const bins = e.bins;
  new LineChart($("counts"), { series: SERIES.map(([k, name, v]) => ({ name, color: css(v),
    points: bins.map((b, j) => [b, e.tracks_per_bin[k][j]]) })), yFormat: (v) => v.toFixed(0), area: false });
  new LineChart($("light"), { series: [{ name: "brightness", color: css("--fam-pedestrian"), points: e.brightness }],
    yMax: 255, yFormat: (v) => v.toFixed(0) });
  new Histogram($("speed"), { edges: e.speed_hist.edges, counts: e.speed_hist.counts, xLabel: "sizes per second" });
  $("glance").replaceChildren(el("div", { class: "table-wrap" }, el("table", { class: "kv" }, el("tbody", {},
    [["Lighting", `${e.lighting} (mean ${e.mean_brightness ?? "–"})`],
      ["Max vehicles in one frame", e.max_simultaneous.vehicle], ["Max pedestrians in one frame", e.max_simultaneous.person],
      ["Vehicle time standing still", `${Math.round(100 * e.vehicle_stationary_share)} %`],
      ["Events detected", Object.entries(e.events_per_class).map(([k, n]) => `${k} ×${n}`).join(", ") || "none"],
      ["Part A processing", `${e.timings.total_sec} s (${e.timings.realtime_factor}× realtime, ${e.timings.device})`]]
      .map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, String(v))))))));
  const base = `data/samples/${r.video.replace(/\.[^.]+$/, "")}`;
  $("pictures").replaceChildren(...PICTURES.filter(([k]) => r.pictures?.[k]).map(([k, cap]) =>
    el("figure", { class: "figure card" }, el("img", { src: `${base}/${r.pictures[k]}`, alt: cap, loading: "lazy" }),
      el("figcaption", {}, cap))));
}
main();
