// SayCut front end: upload → describe → watch the edit render.
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  project: null, busy: false, view: "edited",
  // push-to-talk
  recorder: null, recTimer: null, chunks: [], holding: false, recStarting: false,
  recStartedAt: 0, stream: null, streamTimer: null,
};

// ------------------------------------------------------------------ icons (Lucide, MIT)
const ICONS = {
  scissors: '<circle cx="6" cy="6" r="3"/><path d="M8.12 8.12 12 12"/><path d="M20 4 8.12 15.88"/><circle cx="6" cy="18" r="3"/><path d="M14.8 14.8 20 20"/>',
  github: '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>',
  sparkles: '<path d="M9.94 15.5A2 2 0 0 0 8.5 14.06l-6.14-1.58a.5.5 0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.14a.5.5 0 0 1 .96 0l1.58 6.14a2 2 0 0 0 1.44 1.44l6.14 1.58a.5.5 0 0 1 0 .96l-6.14 1.58a2 2 0 0 0-1.44 1.44l-1.58 6.14a.5.5 0 0 1-.96 0z"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
  alert: '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  shield: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
  mountain: '<path d="m8 3 4 8 5-5 5 15H2L8 3z"/>',
  captions: '<rect width="18" height="14" x="3" y="5" rx="2" ry="2"/><path d="M7 15h4M15 15h2M7 11h2M13 11h4"/>',
  film: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>',
  image: '<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
  layers: '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
  undo: '<path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>',
  plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>',
  "arrow-up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
  droplet: '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>',
  grid: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>',
  palette: '<circle cx="13.5" cy="6.5" r="1"/><circle cx="17.5" cy="10.5" r="1"/><circle cx="8.5" cy="7.5" r="1"/><circle cx="6.5" cy="12.5" r="1"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.93 0 1.65-.75 1.65-1.69 0-.44-.18-.84-.44-1.13-.29-.29-.44-.65-.44-1.13a1.64 1.64 0 0 1 1.67-1.67h2c3.05 0 5.55-2.5 5.55-5.55C21.97 6.01 17.46 2 12 2z"/>',
  focus: '<circle cx="12" cy="12" r="3"/><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>',
  aperture: '<circle cx="12" cy="12" r="10"/><path d="m14.31 8 5.74 9.94"/><path d="M9.69 8h11.48"/><path d="m7.38 12 5.74-9.94"/><path d="M9.69 16 3.95 6.06"/><path d="M14.31 16H2.83"/><path d="m16.62 12-5.74 9.94"/>',
  eraser: '<path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/>',
  smile: '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>',
};
const icon = (name) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;
function hydrateIcons(root = document) {
  root.querySelectorAll("i[data-icon]").forEach((el) => { el.outerHTML = icon(el.dataset.icon); });
}

const EFFECT_ICON = {
  blur: "droplet", pixelate: "grid", glow: "sun", color_pop: "palette", spotlight: "focus",
  background_blur: "aperture", background_replace: "mountain", remove: "eraser", emoji: "smile", captions: "captions",
};

// Which Meta model a pipeline step belongs to.
const MODELS = [
  [/muse spark/i, "Muse Spark", "spark"],
  [/sam 3\.1/i, "SAM 3.1", "sam"],
  [/muse image/i, "Muse Image", "image"],
  [/voice transcribe/i, "Voice", "voice"],
  [/upload/i, "Upload", "render"],
  [/render|reconstruct/i, "Render", "render"],
];
function modelOf(text) {
  for (const [re, label, key] of MODELS) if (re.test(text)) return { label, key };
  return { label: "Step", key: "render" };
}
// "SAM 3.1 tracked 4 × face" → "Tracked 4 × face" (the model is shown as a tag).
function stepText(text) {
  const t = (text || "").replace(/^(Muse Spark|SAM 3\.1|Muse Image|Muse Voice Transcribe)(\s+plan)?\s*(:|is\b)?\s*/i, "").replace(/^·\s*/, "") || text;
  return t.charAt(0).toUpperCase() + t.slice(1);
}
const tagHTML = (m) => `<span class="tag"><span class="dot ${m.key}"></span>${m.label}</span>`;

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

function toast(message, kind = "error") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `${icon({ error: "alert", success: "check", info: "mic" }[kind] || "alert")}<span></span>`;
  el.querySelector("span").textContent = message;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function uploadWithProgress(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/projects");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

// ------------------------------------------------------------------ chat
function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = role === "user"
    ? `<div class="msg-text"></div>`
    : `<span class="msg-avatar">${icon("scissors")}</span><div class="msg-body"><div class="msg-text"></div></div>`;
  el.querySelector(".msg-text").textContent = text;
  $("chat").appendChild(el);
  scrollChat();
  return {
    el,
    setText(t) { el.querySelector(".msg-text").textContent = t; scrollChat(); },
    setError(t) { el.classList.add("error"); this.setText(t); this.setTrace([]); },
    setTrace(steps, live = false) {
      const body = el.querySelector(".msg-body");
      if (!body) return;
      let ul = body.querySelector(".trace");
      if (!steps.length) { ul?.remove(); return; }
      if (!ul) { ul = document.createElement("ul"); ul.className = "trace"; body.appendChild(ul); }
      ul.innerHTML = "";
      steps.forEach((text, i) => {
        const li = document.createElement("li");
        if (live && i === steps.length - 1) li.className = "live";
        li.innerHTML = `${tagHTML(modelOf(text))}<span class="trace-text"></span>`;
        li.querySelector(".trace-text").textContent = stepText(text);
        ul.appendChild(li);
      });
      scrollChat();
    },
  };
}
const scrollChat = () => { const c = $("chat"); c.scrollTop = c.scrollHeight; };

// ------------------------------------------------------------------ busy state + jobs
function setBusy(on) {
  state.busy = on;
  $("status").hidden = !on;
  $("frame-progress").hidden = !on;
  $("frame").classList.toggle("is-busy", on);
  ["mic", "undo", "cover-btn"].forEach((id) => ($(id).disabled = on));
  $("send").disabled = on || !$("prompt").value.trim();
  document.querySelectorAll(".suggestions button, .layer .icon-btn").forEach((b) => (b.disabled = on));
}

function setStatus(stage, progress) {
  const m = modelOf(stage || "");
  $("status-model").innerHTML = `<span class="dot ${m.key}"></span>${m.label}`;
  $("status-text").textContent = stepText(stage || "Working…");
  const bar = $("frame-progress");
  bar.classList.toggle("indeterminate", progress == null);
  $("frame-progress-bar").style.width = progress != null ? `${Math.round(progress * 100)}%` : "";
}

async function runJob(jobId, msg) {
  setBusy(true);
  try {
    while (true) {
      const job = await api(`/api/jobs/${jobId}`);
      setStatus(job.stage, job.progress);
      if (msg) msg.setTrace(job.steps.map((s) => s.stage), job.status === "running");
      if (job.status === "done") return job.result;
      if (job.status === "error") throw new Error(job.error);
      await new Promise((r) => setTimeout(r, 1500));
    }
  } finally {
    setBusy(false);
  }
}

// ------------------------------------------------------------------ project view
function showEditor() {
  $("upload").hidden = true;
  $("editor").hidden = false;
  fitFrame();
}

function showProject(p) {
  const prev = state.project;
  state.project = p;
  showEditor();

  const info = p.info || {};
  $("meta-name").textContent = p.filename || "clip.mp4";
  $("meta-specs").textContent = info.width ? `${fmtTime(info.duration)} · ${info.width}×${info.height} · ${Math.round(info.fps)} fps` : "";
  $("frame").style.setProperty("--ar", info.width ? info.width / info.height : 9 / 16);
  $("backdrop").style.backgroundImage = p.poster_url ? `url("${p.poster_url}")` : "";

  // New render → jump to the edited view.
  if (p.render_url && (!prev || prev.render_url !== p.render_url)) state.view = "edited";
  if (!p.render_url) state.view = "original";
  setView(state.view);

  const dl = $("download");
  dl.classList.toggle("is-disabled", !p.render_url);
  dl.setAttribute("aria-disabled", String(!p.render_url));
  if (p.render_url) { dl.href = p.render_url; dl.download = `saycut-${(p.filename || "clip").replace(/\.[^.]+$/, "")}.mp4`; }
  else dl.removeAttribute("href");

  // Analysis
  const a = p.analysis;
  const title = $("insight-title"), summary = $("insight-summary");
  title.classList.toggle("skeleton", !a);
  summary.classList.toggle("skeleton", !a);
  title.textContent = a ? a.title : " ";
  summary.textContent = a ? a.summary : " ";

  // Suggestions
  const sug = $("suggestions");
  sug.innerHTML = "";
  (a?.suggestions || []).concat(a ? ["Add captions"] : []).slice(0, 5).forEach((s) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = s;
    b.disabled = state.busy;
    b.onclick = () => sendCommand(s);
    sug.appendChild(b);
  });

  // Layers
  const list = $("edits");
  list.innerHTML = "";
  $("edit-count").textContent = p.ops.length;
  if (!p.ops.length) list.innerHTML = `<li class="layer-empty">No edits yet. Describe one in the assistant panel.</li>`;
  p.ops.forEach((op) => {
    const li = document.createElement("li");
    li.className = "layer";
    li.innerHTML = `<span class="layer-icon">${icon(EFFECT_ICON[op.effect] || "sparkles")}</span><span class="layer-label"></span>
      <button class="icon-btn" title="Remove edit" aria-label="Remove edit">${icon("x")}</button>`;
    li.querySelector(".layer-label").textContent = op.label;
    li.title = op.label;
    li.querySelector("button").onclick = () => removeOp(op.id, op.label);
    list.appendChild(li);
  });
  $("undo").disabled = state.busy || !p.ops.length;
  history.replaceState(null, "", `#${p.id}`);
}

function setView(view) {
  const p = state.project;
  if (!p) return;
  if (view === "edited" && !p.render_url) view = "original";
  state.view = view;
  $("view-original").setAttribute("aria-selected", String(view === "original"));
  $("view-edited").setAttribute("aria-selected", String(view === "edited"));
  $("view-edited").disabled = !p.render_url;

  const url = view === "edited" ? p.render_url : p.source_url;
  const video = $("video");
  if (video.getAttribute("src") === url) return;
  const t = video.currentTime || 0;
  const wasPlaying = !video.paused || !video.getAttribute("src");
  video.src = url;
  video.addEventListener("loadedmetadata", () => {
    video.currentTime = Math.min(t, Math.max(0, (video.duration || t) - 0.05));
    if (wasPlaying) video.play().catch(() => {});
  }, { once: true });
}
$("view-original").onclick = () => setView("original");
$("view-edited").onclick = () => setView("edited");

// Fit the frame to the clip's aspect ratio inside the stage.
function fitFrame() {
  const stage = $("stage"), frame = $("frame");
  const ar = parseFloat(getComputedStyle(frame).getPropertyValue("--ar")) || 9 / 16;
  const narrow = window.innerWidth <= 980;
  const pad = narrow ? 32 : 48;
  // On phones the stage grows with the video; on desktop the video fits the stage.
  const w = stage.clientWidth - pad, h = narrow ? window.innerHeight * 0.62 : stage.clientHeight - pad - 56;
  if (w <= 0 || h <= 0) return;
  let fw = w, fh = w / ar;
  if (fh > h) { fh = h; fw = h * ar; }
  frame.style.width = `${Math.floor(fw)}px`;
  frame.style.height = `${Math.floor(fh)}px`;
}
new ResizeObserver(fitFrame).observe($("stage"));
window.addEventListener("resize", fitFrame);
$("video").addEventListener("loadedmetadata", fitFrame);

// ------------------------------------------------------------------ actions
async function upload(file) {
  if (!file || state.busy) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|webm)$/i.test(file.name)) {
    toast("Please choose a video file (MP4 or MOV).");
    return;
  }
  $("dropzone").querySelector(".dz-idle").hidden = true;
  $("dropzone").querySelector(".dz-progress").hidden = false;
  const setPct = (f, label) => {
    $("dz-bar").style.width = `${Math.round(f * 100)}%`;
    $("dz-pct").textContent = `${Math.round(f * 100)}%`;
    if (label) $("dz-label").textContent = label;
  };
  try {
    const { project, job } = await uploadWithProgress(file, (f) => setPct(f, f < 1 ? `Uploading ${file.name}` : "Preparing clip…"));
    resetDropzone();
    $("chat").innerHTML = "";
    showProject(project);
    const msg = addMessage("assistant", "Analyzing your clip…");
    const result = await runJob(job, msg);
    showProject(result.project);
    msg.setTrace([]);
    msg.setText(`I watched it: ${result.project.analysis.summary}\n\nDescribe an edit, or pick a suggestion below.`);
    $("prompt").focus();
  } catch (err) {
    resetDropzone();
    toast(`Upload failed: ${err.message}`);
    if (!state.project) { $("editor").hidden = true; $("upload").hidden = false; }
  }
}
function resetDropzone() {
  $("dropzone").querySelector(".dz-idle").hidden = false;
  $("dropzone").querySelector(".dz-progress").hidden = true;
  $("dz-bar").style.width = "0";
  $("file").value = "";
}

async function sendCommand(text) {
  text = (text || "").trim();
  if (!text || state.busy || !state.project) return;
  $("prompt").value = "";
  autosize();
  addMessage("user", text);
  const msg = addMessage("assistant", "Working on it…");
  try {
    const { job } = await api(`/api/projects/${state.project.id}/command`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    const result = await runJob(job, msg);
    msg.setText(result.reply);
    msg.setTrace(result.trace || []);
    showProject(result.project);
  } catch (err) {
    msg.setError(err.message);
    toast(err.message);
  }
}

async function simpleJob(path, opts, pending) {
  if (state.busy || !state.project) return null;
  const msg = addMessage("assistant", pending);
  try {
    const { job } = await api(path, opts);
    const result = await runJob(job, msg);
    msg.setText(result.reply);
    msg.setTrace([]);
    showProject(result.project);
    return result;
  } catch (err) {
    msg.setError(err.message);
    toast(err.message);
    return null;
  }
}

const removeOp = (opId, label) =>
  simpleJob(`/api/projects/${state.project.id}/ops/${opId}`, { method: "DELETE" }, `Removing “${label}”…`);
const undo = () => state.project?.ops.length && simpleJob(`/api/projects/${state.project.id}/undo`, { method: "POST" }, "Undoing the last edit…");
$("undo").onclick = undo;

$("cover-btn").onclick = async () => {
  const time = $("video").currentTime;
  const result = await simpleJob(`/api/projects/${state.project.id}/cover`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ time }),
  }, `Designing a cover from ${fmtTime(time)}…`);
  if (result?.project?.cover_url) openCover(result.project.cover_url);
};
function openCover(url) {
  $("cover-img").src = url;
  $("cover-download").href = url;
  $("cover-download").download = "saycut-cover.png";
  $("cover-dialog").showModal();
}
$("cover-close").onclick = () => $("cover-dialog").close();
$("cover-dialog").addEventListener("click", (e) => { if (e.target === $("cover-dialog")) $("cover-dialog").close(); });

function newProject() {
  if (state.busy) return;
  history.replaceState(null, "", "/");
  state.project = null;
  $("video").pause();
  $("video").removeAttribute("src");
  $("chat").innerHTML = "";
  $("editor").hidden = true;
  $("upload").hidden = false;
}
$("new-btn").onclick = newProject;
$("home-link").onclick = (e) => { if (state.project) { e.preventDefault(); newProject(); } };

// ------------------------------------------------------------------ composer
const prompt = $("prompt");
function autosize() {
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 140)}px`;
  $("send").disabled = state.busy || !prompt.value.trim();
}
prompt.addEventListener("input", autosize);
prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendCommand(prompt.value); }
});
$("composer").onsubmit = (e) => { e.preventDefault(); sendCommand(prompt.value); };

// ------------------------------------------------------------------ voice (hold to talk)
// Hold Space (or the mic button), speak, release. The mic stream is kept warm for a
// short while after use so the first words of the next command aren't clipped.
const MIN_HOLD_MS = 450;

async function getMicStream() {
  clearTimeout(state.streamTimer);
  if (state.stream && state.stream.getAudioTracks().some((t) => t.readyState === "live")) return state.stream;
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  return state.stream;
}
function releaseMicLater() {
  clearTimeout(state.streamTimer);
  state.streamTimer = setTimeout(() => {
    if (state.recorder || !state.stream) return;
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }, 30000);
}

function showRecording(on, phase = "starting") {
  $("rec").hidden = !on;
  $("rec").classList.toggle("is-starting", phase === "starting");
  $("rec-label").textContent = phase === "starting" ? "Starting mic…" : "Listening";
  $("rec-hint").textContent = phase === "starting" ? "Wait a moment, then speak" : "Release to send";
  $("mic").classList.toggle("recording", on);
  clearInterval(state.recTimer);
  if (on) {
    $("rec-time").textContent = "0:00";
    const started = Date.now();
    state.recTimer = setInterval(() => ($("rec-time").textContent = fmtTime((Date.now() - started) / 1000)), 250);
  }
}

async function startRecording() {
  if (state.busy || state.recorder || state.recStarting || !state.project) return;
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    toast("Voice input isn't supported in this browser. Type your edit instead.");
    return;
  }
  state.holding = true;
  state.recStarting = true;
  showRecording(true); // instant feedback while the mic spins up
  let stream;
  try {
    stream = await getMicStream();
  } catch (err) {
    state.recStarting = false;
    state.holding = false;
    showRecording(false);
    toast(err.name === "NotAllowedError" ? "Microphone access was blocked. Allow it in your browser's site settings." : `Microphone unavailable: ${err.message}`);
    return;
  }
  state.recStarting = false;
  if (!state.holding) {
    // Released before the mic was ready (e.g. while the permission prompt was open).
    showRecording(false);
    releaseMicLater();
    toast("Mic ready. Hold Space (or the mic button) while you speak, then release.", "info");
    return;
  }
  const rec = new MediaRecorder(stream);
  state.chunks = [];
  rec.ondataavailable = (e) => e.data.size && state.chunks.push(e.data);
  rec.onstop = () => finishRecording(rec);
  rec.start();
  state.recorder = rec;
  state.recStartedAt = Date.now();
  $("rec").classList.remove("is-starting");
  $("rec-label").textContent = "Listening";
  $("rec-hint").textContent = "Release to send";
  $("rec-time").textContent = "0:00";
  clearInterval(state.recTimer);
  state.recTimer = setInterval(() => ($("rec-time").textContent = fmtTime((Date.now() - state.recStartedAt) / 1000)), 250);
}

function stopRecording() {
  state.holding = false;
  if (state.recorder && state.recorder.state === "recording") state.recorder.stop();
}

async function finishRecording(rec) {
  const heldFor = Date.now() - state.recStartedAt;
  state.recorder = null;
  showRecording(false);
  releaseMicLater();
  const blob = new Blob(state.chunks, { type: rec.mimeType || "audio/webm" });
  if (heldFor < MIN_HOLD_MS || blob.size < 1500) {
    toast("Keep holding while you speak, then release to send.", "info");
    return;
  }
  const previous = prompt.placeholder;
  prompt.placeholder = "Transcribing with Muse Voice Transcribe…";
  try {
    const form = new FormData();
    form.append("audio", blob, `voice.${(rec.mimeType || "").includes("mp4") ? "m4a" : "webm"}`);
    const { text } = await api("/api/transcribe", { method: "POST", body: form });
    if (text) sendCommand(text);
    else toast("Didn't catch that. Try again a little closer to the mic.");
  } catch (err) {
    toast(`Voice failed: ${err.message}`);
  } finally {
    prompt.placeholder = previous;
  }
}

const mic = $("mic");
mic.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  mic.setPointerCapture?.(e.pointerId); // keep receiving pointerup even if the finger drifts off
  startRecording();
});
mic.addEventListener("pointerup", stopRecording);
mic.addEventListener("pointercancel", stopRecording);
mic.addEventListener("contextmenu", (e) => e.preventDefault()); // long-press on mobile

// ------------------------------------------------------------------ keyboard
// Space is push-to-talk everywhere in the editor, except while typing an actual
// sentence in a text field (an empty composer still counts as "not typing").
function spaceIsTalk() {
  const el = document.activeElement;
  if (el === prompt) return prompt.value.trim() === "";
  if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return false;
  return true;
}
const voiceActive = () => state.holding || state.recStarting || !!state.recorder;

document.addEventListener("keydown", (e) => {
  if (!state.project || $("editor").hidden || $("cover-dialog").open) return;
  if (e.code === "Space") {
    if (voiceActive()) { e.preventDefault(); return; } // swallow auto-repeat
    if (!spaceIsTalk() || e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault(); // no scrolling, no stray space, no play/pause, no button press
    if (!e.repeat) startRecording();
    return;
  }
  const typing = ["TEXTAREA", "INPUT"].includes(document.activeElement?.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z" && !typing) {
    e.preventDefault(); undo();
  } else if (e.key === "/" && !typing) {
    e.preventDefault(); prompt.focus();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code !== "Space" || !voiceActive()) return;
  e.preventDefault(); // a focused button would otherwise "click" on Space release
  stopRecording();
});
// Never get stuck recording if the key-up happens outside the page.
window.addEventListener("blur", () => voiceActive() && stopRecording());
document.addEventListener("visibilitychange", () => document.hidden && voiceActive() && stopRecording());

// ------------------------------------------------------------------ upload wiring
$("file").onchange = (e) => upload(e.target.files[0]);
const dz = $("dropzone");
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
// Allow dropping a new clip anywhere on the landing page.
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); if (!$("upload").hidden && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });

// ------------------------------------------------------------------ boot
hydrateIcons();
(async () => {
  try {
    const health = await api("/api/health");
    $("key-warning").hidden = health.api_key;
  } catch {}
  const pid = location.hash.slice(1);
  if (!pid) return;
  try {
    const p = await api(`/api/projects/${pid}`);
    showProject(p);
    addMessage("assistant", p.analysis ? `Welcome back. ${p.analysis.summary}` : "Welcome back.");
  } catch {
    history.replaceState(null, "", "/");
  }
})();
