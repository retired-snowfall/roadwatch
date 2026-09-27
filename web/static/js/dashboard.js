import { BarChart } from "./charts.js";
import { CLASSES, FAMILIES, chrome, className, el, famColor, fmtTime, getJSON, legend } from "./common.js";

chrome();
const $ = (id) => document.getElementById(id);

function tile(label, value, note = "") {
  return el("div", { class: "card stat" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value),
    note ? el("div", { class: "note" }, note) : "");
}

async function main() {
  let index;
  try {
    index = await getJSON("data/index.json");
  } catch {
    $("tiles").replaceChildren(el("div", { class: "empty" }, "No processed videos yet (tools/build_site.py)."));
    return;
  }
  const vids = index.videos;
  const hours = vids.reduce((a, v) => a + v.duration, 0) / 3600;
  const all = vids.flatMap((v) => v.events.map((e) => ({ v, e })));
  const accidents = all.filter(({ e }) => e[2] === "accident").length;
  const peak = Math.max(0, ...vids.map((v) => v.max_risk));
  $("tiles").replaceChildren(
    tile("Video analysed", `${(hours * 60).toFixed(1)} min`, `${vids.length} clips`),
    tile("Events", String(all.length), `${(all.length / Math.max(hours, 1e-6)).toFixed(0)} per hour`),
    tile("Accidents", String(accidents), accidents ? "see the feed below" : "none detected"),
    tile("Peak accident risk", peak.toFixed(2), peak >= 0.5 ? "an alarm was raised" : "no alarm raised"));

  const counts = {};
  for (const { e } of all) counts[e[2]] = (counts[e[2]] || 0) + 1;
  const bars = Object.entries(counts).sort((a, b) => b[1] - a[1])
    .map(([label, n]) => ({ label: className(label), value: n, color: famColor(label) }));
  const present = new Set(Object.keys(counts).map((c) => CLASSES[c]?.family));
  const fams = Object.keys(FAMILIES).filter((f) => present.has(f));
  $("fam-legend").replaceChildren(...legend(fams).childNodes);
  if (bars.length) new BarChart($("by-class"), { bars });
  else $("by-class").replaceChildren(el("div", { class: "empty" }, "No events."));

  const famCounts = Object.keys(FAMILIES).map((f) => ({ f, n: all.filter(({ e }) => CLASSES[e[2]]?.family === f).length }));
  new BarChart($("rate"), { bars: famCounts.map(({ f, n }) => ({ label: FAMILIES[f], value: n / Math.max(hours, 1e-6),
    color: getComputedStyle(document.documentElement).getPropertyValue(`--fam-${f}`).trim(), note: `${n} events` })),
    valueFormat: (v) => `${v.toFixed(1)}/h` });

  const rows = all.sort((a, b) => a.v.video.localeCompare(b.v.video) || a.e[0] - b.e[0]).map(({ v, e }) =>
    el("tr", { class: "clickable", onclick: () => { location.href = `results.html#${v.id}`; } },
      el("td", {}, v.video), el("td", { class: "num" }, fmtTime(e[0])), el("td", { class: "num" }, `${(e[1] - e[0]).toFixed(1)} s`),
      el("td", {}, el("span", { class: `swatch fam-${CLASSES[e[2]]?.family}` }), className(e[2]))));
  $("feed").replaceChildren(rows.length ? el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Video"),
    el("th", { class: "num" }, "Time"), el("th", { class: "num" }, "Length"), el("th", {}, "Event"))), el("tbody", {}, rows))
    : el("div", { class: "empty" }, "No events."));
}
main();
