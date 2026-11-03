// SayCut front end: upload → talk → watch the Reel change.
const $ = (id) => document.getElementById(id);
const state = { project: null, busy: false, recorder: null, chunks: [] };

const MODEL_TAGS = [
  [/muse spark/i, "SPARK", "m-spark"],
  [/sam 3\.1/i, "SAM", "m-sam"],
  [/muse image/i, "IMAGE", "m-image"],
  [/voice transcribe/i, "VOICE", "m-voice"],
  [/upload/i, "API", "m-render"],
  [/render|reconstruct/i, "RENDER", "m-render"],
];
function tagFor(text) {
  for (const [re, label, cls] of MODEL_TAGS) if (re.test(text)) return [label, cls];
  return ["STEP", "m-render"];
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// ------------------------------------------------------------------ chat + trace
function addMsg(kind, text) {
  const el = document.createElement("div");
  el.className = `msg ${kind}`;
  el.textContent = text;
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
  return el;
}

function renderTrace(container, steps, live) {
  let ul = container.querySelector(".trace");
  if (!ul) { ul = document.createElement("ul"); ul.className = "trace"; container.appendChild(ul); }
  ul.innerHTML = "";
  steps.forEach((step, i) => {
    const [label, cls] = tagFor(step.stage || step);
    const li = document.createElement("li");
    if (live && i === steps.length - 1) li.className = "live";
    li.innerHTML = `<span class="dot ${cls}">${label}</span><span class="t"></span>`;
    li.querySelector(".t").textContent = step.stage || step;
    ul.appendChild(li);
  });
  $("chat").scrollTop = $("chat").scrollHeight;
}

// ------------------------------------------------------------------ jobs
async function runJob(jobId, bubble) {
  setBusy(true);
  try {
    while (true) {
      const job = await api(`/api/jobs/${jobId}`);
      $("busy-text").textContent = job.stage || "Working…";
      $("busy-bar").style.width = job.progress != null ? `${Math.round(job.progress * 100)}%` : "100%";
      $("busy-bar").style.opacity = job.progress != null ? 1 : 0.35;
      if (bubble) renderTrace(bubble, job.steps, job.status === "running");
      if (job.status === "done") return job.result;
      if (job.status === "error") throw new Error(job.error);
      await new Promise((r) => setTimeout(r, 500));
    }
  } finally {
    setBusy(false);
  }
}

function setBusy(on) {
  state.busy = on;
  $("busy-overlay").hidden = !on;
  ["send", "mic", "undo", "cover-btn"].forEach((id) => ($(id).disabled = on));
  document.querySelectorAll(".suggestions button, .edits li button").forEach((b) => (b.disabled = on));
}

// ------------------------------------------------------------------ project view
function showProject(p, { keepTime = false } = {}) {
  state.project = p;
  $("upload").hidden = true;
  $("editor").hidden = false;

  const url = (p.render_url || p.source_url) + `?v=${Date.now()}`;
  const video = $("video");
  const t = keepTime ? video.currentTime : 0;
  if (!video.src.includes(p.render_url || p.source_url) || keepTime) {
    video.src = url;
    $("bgvideo").src = url;
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.min(t, video.duration || t);
      $("bgvideo").currentTime = video.currentTime;
      video.play().catch(() => {});
      $("bgvideo").play().catch(() => {});
    }, { once: true });
  }
  $("badge").hidden = !p.render_url;
  $("badge").textContent = "EDITED";
  $("badge").classList.remove("original");

  const dl = $("download");
  dl.hidden = !p.render_url;
  if (p.render_url) { dl.href = p.render_url; dl.download = `saycut-${p.id}.mp4`; }

  const a = p.analysis;
  $("summary").innerHTML = a
    ? `<strong></strong><span></span>`
    : `<span>Muse Spark is watching your clip…</span>`;
  if (a) {
    $("summary").querySelector("strong").textContent = a.title;
    $("summary").querySelector("span").textContent = a.summary;
  }

  const sug = $("suggestions");
  sug.innerHTML = "";
  (a?.suggestions || []).concat(["Add captions"]).slice(0, 5).forEach((s) => {
    const b = document.createElement("button");
    b.textContent = s;
    b.onclick = () => sendCommand(s);
    sug.appendChild(b);
  });

  const edits = $("edits");
  edits.innerHTML = "";
  if (!p.ops.length) {
    edits.innerHTML = `<li class="empty">No edits yet: say something!</li>`;
  }
  p.ops.forEach((op) => {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><button title="Remove this edit">✕</button>`;
    li.querySelector("span").textContent = op.label;
    li.querySelector("button").onclick = () => removeOp(op.id);
    edits.appendChild(li);
  });
  $("undo").disabled = state.busy || !p.ops.length;

  if (p.cover_url) { $("cover").src = p.cover_url; }
  history.replaceState(null, "", `#${p.id}`);
}

// ------------------------------------------------------------------ actions
async function upload(file) {
  if (!file) return;
  $("upload").hidden = true;
  $("editor").hidden = false;
  $("chat").innerHTML = "";
  setBusy(true);
  $("busy-text").textContent = "Preparing your clip…";
  try {
    const form = new FormData();
    form.append("file", file);
    const { project, job } = await api("/api/projects", { method: "POST", body: form });
    showProject(project);
    const bubble = addMsg("ai", "Got it! Let me take a look…");
    const result = await runJob(job, bubble);
    showProject(result.project);
    bubble.firstChild.textContent = `I watched it: ${result.project.analysis.summary}\nTell me what to change, or tap a suggestion.`;
  } catch (err) {
    setBusy(false);
    addMsg("ai error", `Upload failed: ${err.message}`);
  }
}

async function sendCommand(text) {
  text = (text || "").trim();
  if (!text || state.busy || !state.project) return;
  $("prompt").value = "";
  $("cover").hidden = true;
  addMsg("user", text);
  const bubble = addMsg("ai", "On it…");
  try {
    const { job } = await api(`/api/projects/${state.project.id}/command`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    const result = await runJob(job, bubble);
    bubble.firstChild.textContent = result.reply;
    renderTrace(bubble, (result.trace || []).map((s) => ({ stage: s })), false);
    showProject(result.project, { keepTime: true });
  } catch (err) {
    bubble.classList.add("error");
    bubble.firstChild.textContent = `Something went wrong: ${err.message}`;
  }
}

async function simpleJob(path, opts, label) {
  if (state.busy || !state.project) return;
  const bubble = addMsg("ai", label);
  try {
    const { job } = await api(path, opts);
    const result = await runJob(job, bubble);
    bubble.firstChild.textContent = result.reply;
    bubble.querySelector(".trace")?.remove();
    showProject(result.project, { keepTime: true });
    return result;
  } catch (err) {
    bubble.classList.add("error");
    bubble.firstChild.textContent = `Something went wrong: ${err.message}`;
  }
}

const removeOp = (opId) => simpleJob(`/api/projects/${state.project.id}/ops/${opId}`, { method: "DELETE" }, "Removing that edit…");

$("undo").onclick = () => simpleJob(`/api/projects/${state.project.id}/undo`, { method: "POST" }, "Undoing…");

$("cover-btn").onclick = async () => {
  const time = $("video").currentTime;
  const result = await simpleJob(`/api/projects/${state.project.id}/cover`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ time }),
  }, "Designing a cover from this frame…");
  if (result?.project?.cover_url) {
    $("cover").src = result.project.cover_url + `?v=${Date.now()}`;
    $("cover").hidden = false;
    addMsg("ai", "Tap the cover to go back to the video. Right-click → save to download it.");
  }
};
$("cover").onclick = () => ($("cover").hidden = true);

$("new-btn").onclick = () => {
  history.replaceState(null, "", "/");
  state.project = null;
  $("editor").hidden = true;
  $("upload").hidden = false;
  $("video").removeAttribute("src");
  $("bgvideo").removeAttribute("src");
};

$("composer").onsubmit = (e) => { e.preventDefault(); sendCommand($("prompt").value); };

// Hold to compare with the original.
const compare = $("compare");
let compareTime = 0;
function showOriginal(on) {
  const p = state.project;
  if (!p || !p.render_url) return;
  const v = $("video");
  compareTime = v.currentTime;
  const src = (on ? p.source_url : p.render_url) + `?v=${on ? "o" : Date.now()}`;
  v.src = src;
  v.addEventListener("loadedmetadata", () => { v.currentTime = compareTime; v.play().catch(() => {}); }, { once: true });
  $("badge").textContent = on ? "ORIGINAL" : "EDITED";
  $("badge").classList.toggle("original", on);
}
compare.addEventListener("pointerdown", () => showOriginal(true));
compare.addEventListener("pointerup", () => showOriginal(false));
compare.addEventListener("pointerleave", (e) => { if (e.buttons) showOriginal(false); });

// Keep the blurred backdrop in sync with the main video.
$("video").addEventListener("play", () => $("bgvideo").play().catch(() => {}));
$("video").addEventListener("pause", () => $("bgvideo").pause());
$("video").addEventListener("seeked", () => ($("bgvideo").currentTime = $("video").currentTime));

// ------------------------------------------------------------------ voice (hold to talk)
async function startRecording() {
  if (state.busy || state.recorder) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    state.chunks = [];
    rec.ondataavailable = (e) => e.data.size && state.chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(state.chunks, { type: rec.mimeType || "audio/webm" });
      state.recorder = null;
      $("mic").classList.remove("rec");
      if (blob.size < 2000) return;
      $("prompt").placeholder = "Transcribing with Muse Voice Transcribe…";
      try {
        const form = new FormData();
        const ext = (rec.mimeType || "audio/webm").includes("mp4") ? "m4a" : "webm";
        form.append("audio", blob, `voice.${ext}`);
        const { text } = await api("/api/transcribe", { method: "POST", body: form });
        $("prompt").placeholder = "Say or type an edit…";
        if (text) sendCommand(text);
        else addMsg("ai", "I didn't catch that. Try again?");
      } catch (err) {
        $("prompt").placeholder = "Say or type an edit…";
        addMsg("ai error", `Voice failed: ${err.message}`);
      }
    };
    rec.start();
    state.recorder = rec;
    $("mic").classList.add("rec");
  } catch (err) {
    addMsg("ai error", `Microphone unavailable: ${err.message}`);
  }
}
function stopRecording() { if (state.recorder && state.recorder.state === "recording") state.recorder.stop(); }

$("mic").addEventListener("pointerdown", (e) => { e.preventDefault(); startRecording(); });
$("mic").addEventListener("pointerup", stopRecording);
$("mic").addEventListener("pointerleave", stopRecording);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement !== $("prompt") && state.project) { e.preventDefault(); startRecording(); }
});
document.addEventListener("keyup", (e) => { if (e.code === "Space" && document.activeElement !== $("prompt")) stopRecording(); });

// ------------------------------------------------------------------ upload wiring
$("file").onchange = (e) => upload(e.target.files[0]);
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

// ------------------------------------------------------------------ boot
(async () => {
  try {
    const health = await api("/api/health");
    $("key-warning").hidden = health.api_key;
  } catch {}
  const pid = location.hash.slice(1);
  if (pid) {
    try {
      const p = await api(`/api/projects/${pid}`);
      showProject(p);
      if (p.analysis) addMsg("ai", `Welcome back! ${p.analysis.summary}`);
    } catch { history.replaceState(null, "", "/"); }
  }
})();
