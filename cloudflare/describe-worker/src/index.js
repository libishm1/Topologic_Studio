/**
 * Topologic Studio's describer: a Cloudflare Worker that turns the facts the
 * app computed from an IFC model (core/describeFacts.js) into a short plain
 * description, with a Workers AI model.
 *
 *   POST /describe  {facts: string}  ->  {text, model}
 *   GET  /health                     ->  {ok, model}
 *
 * Only the origins in ALLOWED_ORIGINS may call it. The instructions live
 * here, not in the request, so a page can send facts but not change what the
 * model is told. Facts are capped at MAX_FACTS characters; nothing is stored.
 * On the Workers Free plan the AI allowance (10,000 neurons a day) is a hard
 * cap: past it requests fail, nothing is billed.
 */

export const DEFAULT_MODEL = "@cf/openai/gpt-oss-20b";
export const MAX_FACTS = 8000;

/** Kept equal to DESCRIBE_INSTRUCTIONS in topologicpy-web-frontend/src/core/describeFacts.js (a test checks). */
export const INSTRUCTIONS = [
  "You describe a building model for an architect or fire-safety reviewer, from facts that Topologic Studio computed from the model's IFC file and its own simulations.",
  "Use only the facts given. Never invent rooms, numbers, materials or code compliance; when the facts do not say something, say that they do not.",
  "Write plain English in four short parts: the building; how people get out; what the fire simulation showed, if one was run; what to check next.",
  "Say that the fire results come from uncalibrated models. Keep it under 250 words. No headings in capitals, no bullet lists longer than five items.",
].join(" ");

/** The generated text, whichever shape the model returns it in. */
export function textOf(result) {
  if (typeof result === "string") return result;
  if (typeof result?.response === "string") return result.response;
  const choice = result?.choices?.[0];
  if (typeof choice?.message?.content === "string") return choice.message.content;
  if (typeof choice?.text === "string") return choice.text;
  if (typeof result?.output_text === "string") return result.output_text;
  for (const item of result?.output || []) {
    for (const part of item?.content || []) if (part?.type === "output_text" && typeof part.text === "string") return part.text;
  }
  return "";
}

function json(body, status, headers) {
  return new Response(JSON.stringify(body), { status, headers: { ...headers, "Content-Type": "application/json" } });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const allowed = String(env.ALLOWED_ORIGINS || "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const ok = allowed.includes(origin);
    const cors = ok
      ? {
          "Access-Control-Allow-Origin": origin,
          "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
          Vary: "Origin",
        }
      : { Vary: "Origin" };
    const model = env.MODEL || DEFAULT_MODEL;
    const { pathname } = new URL(request.url);

    if (request.method === "OPTIONS") return new Response(null, { status: ok ? 204 : 403, headers: cors });
    if (request.method === "GET" && pathname === "/health") return json({ ok: true, model }, 200, cors);
    if (request.method !== "POST" || pathname !== "/describe") return json({ error: "Not found." }, 404, cors);
    if (!ok) return json({ error: "This site may not use the describer." }, 403, cors);
    if (Number(request.headers.get("Content-Length") || 0) > 4 * MAX_FACTS) return json({ error: "The facts are too long." }, 413, cors);

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Send JSON: {\"facts\": \"...\"}." }, 400, cors);
    }
    const facts = typeof body?.facts === "string" ? body.facts.slice(0, MAX_FACTS).trim() : "";
    if (!facts) return json({ error: "No facts were sent." }, 400, cors);

    try {
      const result = await env.AI.run(model, {
        messages: [
          { role: "system", content: INSTRUCTIONS },
          { role: "user", content: `Facts:\n${facts}` },
        ],
        max_tokens: 700,
      });
      const text = textOf(result).trim();
      if (!text) return json({ error: "The model returned no text." }, 502, cors);
      return json({ text, model }, 200, cors);
    } catch (error) {
      // Past the free daily allowance Workers AI refuses; say so plainly.
      return json({ error: `The model could not answer: ${error?.message || error}` }, 503, cors);
    }
  },
};
