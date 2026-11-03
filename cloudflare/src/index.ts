// Cloudflare Worker in front of the SayCut container.
//
// SayCut needs ffmpeg, OpenCV and a Python server, so it runs as a Cloudflare
// Container. Every request goes to one named instance ("main") so in-memory job
// progress and project files stay consistent between polling requests.
import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  SAYCUT: DurableObjectNamespace<SayCutContainer>;
  MODEL_API_KEY: string;
}

export class SayCutContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "20m"; // stay warm between edits; sleep when nobody is using it
  enableInternet = true; // calls api.meta.ai
  pingEndpoint = "/api/health";
  envVars = {
    MODEL_API_KEY: this.env.MODEL_API_KEY,
    SAYCUT_PUBLIC: "1",
  };

  override onStart() {
    console.log("SayCut container started");
  }

  override onError(error: unknown) {
    console.error("SayCut container error", error);
    throw error;
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    // Redirect the workers.dev address to the custom domain.
    if (url.hostname.endsWith(".workers.dev")) {
      url.hostname = "saycut.softmelody.dev";
      return Response.redirect(url.toString(), 301);
    }
    try {
      return await getContainer(env.SAYCUT, "main").fetch(request);
    } catch (err) {
      console.error("container fetch failed", err);
      const html = `<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="6">
<body style="background:#0b0b10;color:#f4f4f8;font:16px system-ui;display:grid;place-items:center;height:100vh;margin:0">
<div style="text-align:center"><div style="font-size:48px">✂️</div><h2>SayCut is waking up…</h2><p style="color:#9a9ab0">The editor container is starting. This page refreshes automatically.</p></div>`;
      const isApi = url.pathname.startsWith("/api/");
      return isApi
        ? Response.json({ detail: "SayCut is starting up, try again in a few seconds." }, { status: 503 })
        : new Response(html, { status: 503, headers: { "content-type": "text/html; charset=utf-8" } });
    }
  },
} satisfies ExportedHandler<Env>;
