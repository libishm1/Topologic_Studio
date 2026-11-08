# Topologic Studio describer (Cloudflare Worker)

"Describe this model" in Studio's Model step writes a short plain description of a building from the facts Studio has computed from its IFC file and its simulations. The facts are counts, storeys, rooms, doors and exits, the walkable graph, the route, and the fire's results (`topologicpy-web-frontend/src/core/describeFacts.js`). On Chrome 148 or later on a desktop computer, the browser's own model writes it on the device. This Worker is the alternative for every other browser: it runs a Workers AI model on Cloudflare's free allowance.

## What it does

- `POST /describe` with `{"facts": "..."}` returns `{"text", "model"}`. `GET /health` returns `{"ok", "model"}`.
- **Who may call it:** only the origins in `ALLOWED_ORIGINS`. The model's instructions live in the Worker, so a caller can send facts but cannot change what the model is told.
- **Input limit:** facts are capped at 8,000 characters.
- **Storage:** nothing is stored by the Worker. Cloudflare says it "does not use Customer Content processed by Workers AI to train AI models". It also says it "does not retain inputs or outputs after they are no longer needed to process the request", unless logging is turned on, for example in AI Gateway. Source: https://developers.cloudflare.com/workers-ai/platform/data-usage/, read on 2026-11-05. Keep logging off.
- **Cost:** on the Workers Free plan the AI allowance is 10,000 neurons a day, reset at 00:00 UTC. Past it, requests fail and nothing is billed. With `@cf/openai/gpt-oss-20b` (18,182 neurons per million input tokens, 27,273 per million output, on Cloudflare's pricing page on 2026-11-05), a description of about 1,000 tokens in and 400 out costs about 30 neurons, roughly 300 a day.

## Room suggestions for drawings

When Studio drafts rooms from a floor-plan drawing, it can ask the same Worker for suggestions.

- **Endpoint:** `POST /rooms` with `{"summary": {...}}`. The summary covers one storey: each room's id, area, the texts found inside it, its centroid, its neighbours and the rooms its doors open to.
- **What comes back:** `{"merge", "names", "dropped", "model"}`. `merge` lists groups of rooms the tracing split in error. `names` gives a name and class for rooms with no clear label.
- **Checks before returning:** the Worker keeps a suggestion only if:
  - it names rooms that were sent;
  - a merge joins rooms that touch;
  - no room appears in two merges or gets two names.

  A class must be one of the LOD1 model's (Unit, Corridor, Stair, Ramp, Toilet, Service, Office, Commercial, Parking, Arcade, Shaft, Void, Tank). `dropped` counts the suggestions left out.
- **What the page does with them:** it shows each suggestion with its reason, and applies only the ones the user accepts.
- **Input limit:** a summary is capped at 24,000 characters, enough for a storey of a few hundred rooms.
- **Cost:** with the default model, a storey of about 100 rooms costs roughly 100 to 200 neurons. That allows about 50 storeys a day on the free allowance.

## Deploy it (with your own Cloudflare account)

1. Create a free Cloudflare account at https://dash.cloudflare.com/sign-up, if you have none.
2. From this folder:

   ```
   npx wrangler login      # opens the browser to authorise wrangler
   npx wrangler deploy     # prints the Worker's address
   ```

   The address looks like `https://topologic-describe.<your-subdomain>.workers.dev`. Check it with `curl https://topologic-describe.<your-subdomain>.workers.dev/health`.
3. Give the site the address. In the GitHub repository, go to Settings, then Secrets and variables, then Actions, then Variables, and add `DESCRIBE_URL`. Its value is the address above, without a trailing slash or path.
4. Rebuild the site: Actions, then Deploy Frontend, then Run workflow. "Describe this model" then offers Cloudflare.

For local development, put `VITE_DESCRIBE_URL=<the address>` in `topologicpy-web-frontend/.env.local`. `.env.local` is not committed.

## The human check (Cloudflare Turnstile)

Turnstile is Cloudflare's "I'm human" check. It keeps bots from spending the Worker's free AI allowance, and it is optional.

- **The page:** it loads Turnstile only when someone chooses Cloudflare for a cloud feature, and shows one slim strip under the send button. It checks at once, usually with no click, and asks for the action `describe`.
- **The Worker:** it then refuses any request unless Cloudflare accepts its token, for the route's action (`describe` on `/describe`, `rooms` on `/rooms`), from a hostname in `TURNSTILE_HOSTNAMES` in `wrangler.toml` (the live site only). With the secret set and no hostnames, it refuses everything. This follows Cloudflare's Turnstile Spin checks.

To turn it on:

1. In the Cloudflare dashboard, go to Turnstile and add a widget. Set the mode to Managed, and the hostname to `libishm1.github.io` (add `localhost` only if you need it for development). Cloudflare gives a site key, which is public, and a secret key.
2. Give the Worker the secret key. From this folder, run `npx wrangler secret put TURNSTILE_SECRET` and paste the key when asked. Secrets are not kept in `wrangler.toml`.
3. Give the site the site key. In the GitHub repository, add the variable `TURNSTILE_SITE_KEY` next to `DESCRIBE_URL`, then run Deploy Frontend again.

For local testing, Cloudflare's test keys (site key `1x00000000000000000000AA`, secret `1x0000000000000000000000000000000AA`) make the page's check pass, but their answers report the hostname `example.com` and no action, so this Worker refuses them (https://developers.cloudflare.com/turnstile/troubleshooting/testing/, read on 2026-11-08). To try the whole check locally, use your own widget (Cloudflare adds `localhost` to its domains) with `wrangler dev`, and put `TURNSTILE_SECRET` and `TURNSTILE_HOSTNAMES=localhost` in a `.dev.vars` file in this folder (git ignores it). Never add `localhost` to the deployed Worker's `TURNSTILE_HOSTNAMES`.

Turnstile collects browser signals to tell people from bots. Cloudflare describes what it collects in its Turnstile privacy addendum.

## Change the model or the sites

Edit `wrangler.toml`:

- `MODEL`: any Workers AI text-generation model that takes chat messages, such as `@cf/meta/llama-3.1-8b-instruct-fp8-fast`, which is cheaper, or `@cf/meta/llama-3.3-70b-instruct-fp8-fast`, which is stronger and dearer.
- `ALLOWED_ORIGINS`: the pages that may call the Worker.

Then run `npx wrangler deploy` again.

## Test it

```
node --test test/index.test.mjs
```

The tests run the Worker with a stand-in for Workers AI. They check that only allowed sites are served, that the instructions match the page's, that facts are capped, and that a model error, such as a used-up daily allowance, is reported plainly.
