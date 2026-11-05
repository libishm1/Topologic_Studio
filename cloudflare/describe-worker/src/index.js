/**
 * Topologic Studio's describer: a Cloudflare Worker that turns the facts the
 * app computed from an IFC model (core/describeFacts.js) into a short plain
 * description, with a Workers AI model.
 *
 *   POST /describe  {facts: string}  ->  {text, model}
 *   POST /rooms     {summary}        ->  {merge, names, dropped, model}
 *   GET  /health                     ->  {ok, model}
 *
 * /rooms reviews the rooms the app drafted from a floor-plan drawing
 * (core/drawingToLod1.js, draftSummary) and suggests rooms to merge and
 * names and classes for unlabelled ones. Suggestions are checked against
 * the summary before they are returned (checkSuggestions); the page shows
 * them for the user to accept or reject, and applies none on its own.
 *
 * Only the origins in ALLOWED_ORIGINS may call it. The instructions live
 * here, not in the request, so a page can send facts but not change what the
 * model is told. Facts are capped at MAX_FACTS characters, summaries at
 * MAX_SUMMARY; nothing is stored. On the Workers Free plan the AI allowance
 * (10,000 neurons a day) is a hard cap: past it requests fail, nothing is
 * billed.
 */

export const DEFAULT_MODEL = "@cf/openai/gpt-oss-20b";
export const MAX_FACTS = 8000;
export const MAX_SUMMARY = 24000;

/** The room classes of the LOD1 model (core/lod1.js); a suggested class must be one of them. */
export const ROOM_CLASSES = ["Unit", "Corridor", "Stair", "Ramp", "Toilet", "Service", "Office", "Commercial", "Parking", "Arcade", "Shaft", "Void", "Tank"];

export const ROOM_INSTRUCTIONS = [
  "You review rooms that a program traced from one storey of a CAD floor plan. Each room has an id, its area in square metres, the texts found inside it (labels), its centroid, the rooms it shares a boundary with (neighbours) and the rooms it opens to through a door (doors).",
  "Suggest two things only. First, merges: groups of neighbouring rooms that are one space the tracing split in error, such as a corridor cut into pieces with no door between them, or a sliver beside a larger room. Second, names: for rooms whose labels are missing, unclear or abbreviated, a short name and a class.",
  `The class must be one of: ${ROOM_CLASSES.join(", ")}.`,
  "Use only the ids given. Never invent rooms. Suggest nothing you are unsure of; an empty list is a good answer. Give each suggestion a reason of at most fifteen words.",
  'Answer with JSON only, in this form: {"merge": [{"rooms": ["id", "id"], "why": "..."}], "names": [{"room": "id", "name": "...", "class": "...", "why": "..."}]}',
].join(" ");

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

/** The first JSON object in a model's text, which may come fenced or with words around it; null if none parses. */
export function jsonOf(text) {
  const s = String(text || "");
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(s.slice(start, end + 1));
  } catch {
    return null;
  }
}

/** A short single-line string from whatever the model gave, or "" when it gave none. */
const clean = (value, max) =>
  typeof value === "string"
    ? value
        .replace(/[\u0000-\u001f\u007f]+/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, max)
    : "";

/**
 * The model's suggestions, kept only where they hold for the summary sent:
 * merges of two or more known rooms that form one connected group through
 * their neighbours, each room in one merge at most; names for known rooms,
 * one each, with a class from ROOM_CLASSES (an unknown class is dropped, the
 * name kept). `dropped` counts the suggestions left out.
 */
export function checkSuggestions(raw, summary) {
  const rooms = new Map((summary?.rooms || []).map((r) => [String(r.id), r]));
  const touching = (a, b) => {
    const ra = rooms.get(a), rb = rooms.get(b);
    const has = (r, id) => (r?.neighbours || []).map(String).includes(id) || (r?.doors || []).map(String).includes(id);
    return has(ra, b) || has(rb, a);
  };
  const merge = [];
  const names = [];
  let dropped = 0;
  const merged = new Set();
  for (const m of Array.isArray(raw?.merge) ? raw.merge : []) {
    const ids = [...new Set((Array.isArray(m?.rooms) ? m.rooms : []).map(String))];
    // Connected through neighbours: grow from the first room.
    const reached = new Set(ids.length ? [ids[0]] : []);
    for (let grew = true; grew; ) {
      grew = false;
      for (const id of ids) {
        if (!reached.has(id) && [...reached].some((r) => touching(r, id))) {
          reached.add(id);
          grew = true;
        }
      }
    }
    const ok = ids.length >= 2 && ids.every((id) => rooms.has(id) && !merged.has(id)) && reached.size === ids.length;
    if (!ok) {
      dropped += 1;
      continue;
    }
    ids.forEach((id) => merged.add(id));
    merge.push({ rooms: ids, why: clean(m.why, 160) });
  }
  const named = new Set();
  for (const n of Array.isArray(raw?.names) ? raw.names : []) {
    const room = String(n?.room ?? "");
    const name = clean(n?.name, 60);
    if (!rooms.has(room) || named.has(room) || !name) {
      dropped += 1;
      continue;
    }
    named.add(room);
    const cls = ROOM_CLASSES.find((c) => c.toLowerCase() === clean(n.class, 20).toLowerCase()) ?? null;
    names.push({ room, name, class: cls, why: clean(n.why, 160) });
  }
  return { merge, names, dropped };
}

function json(body, status, headers) {
  return new Response(JSON.stringify(body), { status, headers: { ...headers, "Content-Type": "application/json" } });
}

/** POST /rooms: suggestions for a drafted storey, checked against it. */
async function suggestRooms(request, env, model, cors) {
  if (Number(request.headers.get("Content-Length") || 0) > 2 * MAX_SUMMARY) return json({ error: "The summary is too long." }, 413, cors);
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Send JSON: {\"summary\": {\"rooms\": [...]}}." }, 400, cors);
  }
  const summary = body?.summary;
  if (!Array.isArray(summary?.rooms) || summary.rooms.length === 0) return json({ error: "No rooms were sent." }, 400, cors);
  const text = JSON.stringify(summary);
  if (text.length > MAX_SUMMARY) return json({ error: "The summary is too long: send one storey at a time." }, 413, cors);
  try {
    const result = await env.AI.run(model, {
      messages: [
        { role: "system", content: ROOM_INSTRUCTIONS },
        { role: "user", content: `Rooms:\n${text}` },
      ],
      max_tokens: 2500,
    });
    const raw = jsonOf(textOf(result));
    if (!raw) return json({ error: "The model did not answer in JSON." }, 502, cors);
    return json({ ...checkSuggestions(raw, summary), model }, 200, cors);
  } catch (error) {
    return json({ error: `The model could not answer: ${error?.message || error}` }, 503, cors);
  }
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
    if (request.method !== "POST" || (pathname !== "/describe" && pathname !== "/rooms")) return json({ error: "Not found." }, 404, cors);
    if (!ok) return json({ error: "This site may not use the describer." }, 403, cors);
    if (pathname === "/rooms") return suggestRooms(request, env, model, cors);
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
