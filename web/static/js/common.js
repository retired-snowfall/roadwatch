// Shared page chrome (nav, theme), formatting and the event-class catalogue.
export const CLASSES = {
  accident: { name: "Accident", family: "collision" },
  near_miss: { name: "Near miss", family: "collision" },
  red_light: { name: "Red-light running", family: "signal" },
  stop_line: { name: "Stop-line violation", family: "signal" },
  solid_line_crossing: { name: "Solid line crossing", family: "signal" },
  wrong_way: { name: "Wrong-way driving", family: "manoeuvre" },
  illegal_u_turn: { name: "Illegal U-turn", family: "manoeuvre" },
  illegal_turn: { name: "Illegal turn", family: "manoeuvre" },
  jaywalking: { name: "Pedestrian on roadway", family: "pedestrian" },
  failure_to_yield: { name: "Not yielding to pedestrian", family: "pedestrian" },
  stopped_vehicle: { name: "Stopped vehicle", family: "state" },
  congestion: { name: "Congestion", family: "state" },
  road_obstacle: { name: "Obstacle on road", family: "state" },
  fire_smoke: { name: "Fire or smoke", family: "state" },
};
export const CLASS_ORDER = Object.keys(CLASSES);
export const FAMILIES = {
  collision: "Collisions",
  signal: "Signals & markings",
  manoeuvre: "Manoeuvres",
  pedestrian: "Pedestrians",
  state: "Traffic state & hazards",
};

export function famColor(label) {
  const fam = CLASSES[label]?.family || "state";
  return getComputedStyle(document.documentElement).getPropertyValue(`--fam-${fam}`).trim();
}

export function className(label) {
  return CLASSES[label]?.name || label;
}

export function fmtTime(sec) {
  if (!isFinite(sec)) return "–";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

export async function getJSON(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

// API base: same origin by default; a static host (e.g. GitHub Pages) can point at a hosted backend
// with <meta name="roadwatch-api" content="https://...">.
export function apiBase() {
  const m = document.querySelector('meta[name="roadwatch-api"]');
  return (m && m.content) || "";
}

const PAGES = [
  ["index.html", "Overview"],
  ["demo.html", "Live demo"],
  ["results.html", "Results"],
  ["dashboard.html", "Dashboard"],
  ["eda.html", "EDA"],
  ["approach.html", "Approach"],
  ["report.html", "Report"],
  ["team.html", "Team"],
  ["tools.html", "Tools"],
];

const LOGO = `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="2" width="18" height="20" rx="5" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="7.5" r="2.2" fill="var(--critical)"/><circle cx="12" cy="12.5" r="2.2" fill="var(--warning)"/><circle cx="12" cy="17.5" r="2.2" fill="var(--good)"/></svg>`;

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

export function chrome() {
  let saved = null;
  try { saved = localStorage.getItem("rw-theme"); } catch { /* storage may be unavailable */ }
  applyTheme(saved);
  const here = location.pathname.split("/").pop() || "index.html";
  const header = el("header", { class: "site-header" });
  const nav = el("nav", { class: "nav", "aria-label": "Main" });
  const brand = el("a", { class: "brand", href: "index.html" });
  brand.innerHTML = `${LOGO}<span>roadwatch</span>`;
  const links = el("div", { class: "nav-links", id: "nav-links" },
    PAGES.map(([href, label]) => el("a", { href, "aria-current": href === here ? "page" : null }, label)));
  const menu = el("button", { class: "menu-toggle", "aria-controls": "nav-links", "aria-expanded": "false",
    onclick: () => { links.classList.toggle("open"); menu.setAttribute("aria-expanded", links.classList.contains("open")); } }, "Menu");
  const toggle = el("button", { class: "theme-toggle", title: "Toggle light / dark", "aria-label": "Toggle theme",
    onclick: () => {
      const dark = document.documentElement.dataset.theme === "dark" ||
        (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem("rw-theme", next); } catch { /* ignore */ }
      window.dispatchEvent(new Event("themechange"));
    } }, "◐");
  nav.append(brand, menu, links, toggle);
  header.append(nav);
  document.body.prepend(header);
  const footer = el("footer", { class: "site-footer" });
  footer.innerHTML = `<div class="inner"><span>roadwatch — traffic events from a fixed CCTV camera</span>
    <a href="https://github.com/retired-snowfall/some_shi">Repository</a>
    <a href="data/predictions_samples.json">predictions_samples.json</a>
    <a href="report.html">Technical report</a></div>`;
  document.body.append(footer);
}

export function legend(families = Object.keys(FAMILIES)) {
  return el("div", { class: "legend", role: "list" },
    families.map((f) => el("span", { role: "listitem" }, el("span", { class: `swatch fam-${f}` }), FAMILIES[f])));
}
