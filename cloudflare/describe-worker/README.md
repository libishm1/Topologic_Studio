# Topologic Studio describer (Cloudflare Worker)

"Describe this model" in Studio's Model step writes a short plain description of a building from the facts Studio has computed from its IFC file and its simulations. The facts are counts, storeys, rooms, doors and exits, the walkable graph, the route, and the fire's results (`topologicpy-web-frontend/src/core/describeFacts.js`). On Chrome 148 or later on a desktop computer, the browser's own model writes it on the device. This Worker is the alternative for every other browser: it runs a Workers AI model on Cloudflare's free allowance.

## What it does

- `POST /describe` with `{"facts": "..."}` returns `{"text", "model"}`. `GET /health` returns `{"ok", "model"}`.
- **Who may call it:** only the origins in `ALLOWED_ORIGINS`. The model's instructions live in the Worker, so a caller can send facts but cannot change what the model is told.
- **Input limit:** facts are capped at 8,000 characters.
- **Storage:** nothing is stored by the Worker. Cloudflare says it "does not use Customer Content processed by Workers AI to train AI models". It also says it "does not retain inputs or outputs after they are no longer needed to process the request", unless logging is turned on, for example in AI Gateway. Source: https://developers.cloudflare.com/workers-ai/platform/data-usage/, read on 2026-11-05. Keep logging off.
- **Cost:** on the Workers Free plan the AI allowance is 10,000 neurons a day, reset at 00:00 UTC. Past it, requests fail and nothing is billed. With `@cf/openai/gpt-oss-20b` (18,182 neurons per million input tokens, 27,273 per million output, on Cloudflare's pricing page on 2026-11-05), a description of about 1,000 tokens in and 400 out costs about 30 neurons, roughly 300 a day.

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
