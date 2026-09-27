import { apiBase, chrome, el, getJSON } from "./common.js";
import { ResultViewer } from "./viewer.js";

chrome();
const API = apiBase();
const $ = (id) => document.getElementById(id);
const STAGES = [["upload", "Upload"], ["queued", "Queue"], ["preview", "Preview"], ["detect", "Detect & track"],
  ["rules", "Event rules"], ["risk", "Risk curve"], ["done", "Done"]];
let file = null;

function setStages(active) {
  const idx = STAGES.findIndex(([k]) => k === active);
  $("stages").replaceChildren(...STAGES.map(([k, name], i) =>
    el("span", { class: i < idx ? "done" : i === idx ? "active" : "" }, `${i < idx ? "✓ " : ""}${name}`)));
}

function setProgress(frac, text) {
  const pct = Math.round(frac * 100);
  $("bar").firstElementChild.style.width = `${pct}%`;
  $("bar").setAttribute("aria-valuenow", pct);
  if (text) $("status").textContent = text;
}

function pick(f) {
  if (!f) return;
  file = f;
  const mb = f.size / 1e6;
  const probe = el("video", { preload: "metadata", muted: "" });
  probe.src = URL.createObjectURL(f);
  $("picked").textContent = `${f.name} · ${mb.toFixed(1)} MB`;
  $("go").disabled = mb > 200;
  if (mb > 200) $("picked").append(el("div", { class: "error" }, "Larger than 200 MB — please trim the video."));
  probe.onloadedmetadata = () => {
    $("picked").textContent = `${f.name} · ${mb.toFixed(1)} MB · ${probe.duration.toFixed(1)} s · ${probe.videoWidth}×${probe.videoHeight}`;
    if (probe.duration > 180) {
      $("go").disabled = true;
      $("picked").append(el("div", { class: "error" }, "Longer than 3 minutes — please trim the video."));
    }
  };
}

const drop = $("drop");
$("file").addEventListener("change", (e) => pick(e.target.files[0]));
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); });
drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));

function upload(f) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/api/jobs`);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) setProgress(0.05 * e.loaded / e.total, `Uploading… ${Math.round(100 * e.loaded / e.total)} %`); };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body.id);
      else reject(new Error(body.detail || `upload failed (HTTP ${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("network error — is the demo server running?"));
    const form = new FormData();
    form.append("file", f);
    xhr.send(form);
  });
}

async function poll(id) {
  for (;;) {
    const s = await getJSON(`${API}/api/jobs/${id}`);
    if (s.status === "error") throw new Error(s.error || "processing failed");
    if (s.status === "done") return s;
    const stage = s.status === "queued" ? "queued" : s.stage === "tracks" ? "rules" : s.stage;
    setStages(stage);
    const eta = s.eta_sec != null ? ` · about ${Math.max(1, Math.round(s.eta_sec))} s left` : "";
    const txt = s.status === "queued" ? `Waiting in queue (position ${s.queue_position ?? "?"})…`
      : `${STAGES.find(([k]) => k === stage)?.[1] || s.stage} · ${Math.round(s.progress * 100)} %${eta}`;
    setProgress(0.05 + 0.95 * s.progress, txt);
    await new Promise((r) => setTimeout(r, 1000));
  }
}

$("go").addEventListener("click", async () => {
  if (!file) return;
  $("go").disabled = true;
  setStages("upload");
  try {
    const id = await upload(file);
    const s = await poll(id);
    setStages("done");
    setProgress(1, `Done in ${s.elapsed_sec} s.`);
    const result = await getJSON(`${API}/api/jobs/${id}/result`);
    new ResultViewer($("result"), { result, videoUrl: `${API}/api/jobs/${id}/video`, title: result.video });
  } catch (err) {
    setProgress(0, "");
    $("status").replaceChildren(el("span", { class: "error" }, `Error: ${err.message}`));
  } finally {
    $("go").disabled = false;
  }
});
setStages("upload");
getJSON(`${API}/api/health`).then((h) => {
  $("limits").textContent = `Accepted: ${h.limits.formats.join(", ")} up to ${h.limits.max_mb} MB and ${Math.round(h.limits.max_sec / 60)} minutes.
    Two minutes is plenty. Files are deleted after 3 hours. Demo server: ${h.device.toUpperCase()}.`;
}).catch(() => {
  $("status").replaceChildren(el("span", { class: "error" }, "The demo server is not reachable right now."));
});
