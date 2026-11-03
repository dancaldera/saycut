// Cloudflare Worker in front of the SayCut container.
//
// Cost model: the container (Python + ffmpeg + OpenCV) is billed for every second it
// is awake, so the Worker keeps it asleep unless real editing work arrives:
//   * the UI (web/) is served as Workers Static Assets: free, never wakes anything
//   * /api/health is answered here at the edge
//   * /media/* (immutable renders, covers, posters) is cached at the edge, so replays,
//     range requests and before/after toggles don't reach the container
//   * only uploads, edits, job polling and first-time media fetches hit the container
import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  SAYCUT: DurableObjectNamespace<SayCutContainer>;
  MODEL_API_KEY: string;
}

export class SayCutContainer extends Container<Env> {
  defaultPort = 8000;
  // Job polling keeps it awake during edits; sleep soon after people stop editing.
  sleepAfter = "8m";
  enableInternet = true; // calls api.meta.ai
  pingEndpoint = "/api/health";
  envVars = {
    MODEL_API_KEY: this.env.MODEL_API_KEY,
    SAYCUT_PUBLIC: "1",
  };

  override onError(error: unknown) {
    console.error("SayCut container error", error);
    throw error;
  }
}

const MODELS = {
  planner: "muse-spark-1.3",
  vision: "muse-spark-1.3",
  segmentation: "sam-3.1",
  image: "muse-image-1.0",
  voice: "muse-voice-transcribe-1.0",
};

const container = (env: Env) => getContainer(env.SAYCUT, "main");

async function cachedMedia(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const url = new URL(request.url);
  url.search = "";
  const key = new Request(url.toString(), { method: "GET" });
  const range = request.headers.get("range");
  // Cache API answers Range requests with 206 from a stored full response.
  const lookup = range ? new Request(url.toString(), { headers: { range } }) : key;

  const hit = await caches.default.match(lookup);
  if (hit) return hit;

  // Miss: fetch the whole file once (no Range), store it, then serve from cache.
  const full = await container(env).fetch(key);
  if (full.status !== 200 || !full.headers.get("cache-control")?.includes("immutable")) return full;
  await caches.default.put(key, full);
  const stored = await caches.default.match(lookup);
  if (stored) return stored;
  // Cache unavailable (e.g. workers.dev): fall back to proxying.
  return container(env).fetch(new Request(url.toString(), { headers: request.headers }));
}

function wakingUp(isApi: boolean): Response {
  if (isApi) return Response.json({ detail: "SayCut is starting up, try again in a few seconds." }, { status: 503 });
  const html = `<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="6">
<body style="background:#0b0b10;color:#f4f4f8;font:16px system-ui;display:grid;place-items:center;height:100vh;margin:0">
<div style="text-align:center"><div style="font-size:48px">✂️</div><h2>SayCut is waking up…</h2><p style="color:#9a9ab0">The editor is starting. This page refreshes automatically.</p></div>`;
  return new Response(html, { status: 503, headers: { "content-type": "text/html; charset=utf-8" } });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.hostname.endsWith(".workers.dev")) {
      url.hostname = "saycut.softmelody.dev";
      return Response.redirect(url.toString(), 301);
    }

    if (url.pathname === "/api/health") {
      return Response.json({ ok: true, api_key: Boolean(env.MODEL_API_KEY), public: true, models: MODELS });
    }

    try {
      if (request.method === "GET" && url.pathname.startsWith("/media/")) {
        return await cachedMedia(request, env, ctx);
      }
      if (url.pathname.startsWith("/api/")) {
        return await container(env).fetch(request);
      }
    } catch (err) {
      console.error("container fetch failed", err);
      return wakingUp(url.pathname.startsWith("/api/"));
    }

    // Anything else is a static asset miss: never wake the container for it.
    return new Response("Not found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;
