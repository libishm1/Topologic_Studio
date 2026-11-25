# Free and low-cost LLM inference, GPU time and backend hosting for Topologic Studio

Date: 2026-11-05
Status: desk research. No account was created, no form was submitted, and no personal detail was sent to any site. Every claim below was read on the provider's own page on 2026-11-05 unless it is marked "secondary" (a third-party page, used only where the provider's page could not be read). Prices, quotas and model names change often; re-check before relying on a number.
Related: [ifc-identification-benchmark-2026-11-05.md](ifc-identification-benchmark-2026-11-05.md) (why the IFCNet DGCNN weights need retraining on openly licensed data), [../deployment](../deployment) (the current Pages + Render set-up).

The three needs:

- (a) LLM inference: an optional "describe this model" feature, research scripts, literature tooling.
- (b) GPU time: retrain IFCNet's DGCNN (PyTorch) on openly licensed geometry, a few hours a run on a T4 or A100 class GPU, plus experiments; tens of GPU hours in all.
- (c) Hosting for the FastAPI + TopologicPy backend, now on Render's free plan (cold start 30 to 50 s; the IFC cross-check took 74 s once and timed out at 240 s once).

---

## 0. Summary

| Need | 1st | 2nd | 3rd |
|---|---|---|---|
| (a) Inference | Cloudflare Workers AI free allocation (10,000 neurons a day, no training on content, no application) for the in-app feature | FreeInference, once approved, for research scripts and literature tooling only | Anthropic AI for Science API credits (apply; reviewed monthly), with the Gemini API free tier as the no-application stopgap for scripts on public text |
| (b) GPU | Kaggle Notebooks (free; about 30 GPU hours a week, secondary) | Modal ($30 of free credit a month, about 50 T4 hours or 14 A100-40GB hours; academic credits up to $10k on request) | Google Colab free tier for short experiments (GPU not guaranteed); Lightning AI's free credits as an alternative |
| (c) Hosting | Hugging Face PRO ($9 a month) running the backend as a Docker Space on the free CPU Basic tier: 2 vCPU and 16 GB against Render free's 0.1 CPU and 512 MB | Google Cloud Run free tier (180,000 vCPU-seconds a month; scale to zero; billing account needed) | Modal's monthly credit (CPU at $0.0000131 a core-second; scale to zero), or Render Starter ($7) as the smallest change |

Two findings shape the inference choice. First, Studio's README promises "Your model stays on your device"; any LLM description sends model-derived text to a third party, so it has to be opt-in and disclosed, and the provider must not log or publish it. Second, FreeInference logs every prompt and response, may publish sanitised prompts and responses as a public dataset, forbids embedding its key in client-side code, and has a manual-review queue for non-Harvard signups (3,296 waiting against 1,813 approved on its public stats). It suits the user's own research scripts on public or own text, not end users' building models.

For GPU time, the free notebook tiers cover the stated need (tens of hours) without any application. Credit programmes (Google Cloud research credits, which list Saudi Arabia; AWS; Modal academics; Nebius) are worth an application only if the work grows. KAUST's Shaheen III and Ibex are reachable from Alfaisal only through an organisation-level agreement or a KAUST collaborator, which is out of proportion to tens of GPU hours.

---

## 1. FreeInference

Sources read: home page <https://freeinference.org/>, terms <https://freeinference.org/terms> (last updated June 20, 2026), docs <https://doc.freeinference.org> (models page <https://doc.freeinference.org/models.html>, quick start <https://doc.freeinference.org/quickstart.html>), status <https://status.freeinference.org/>, team <https://freeinference.org/team>, the public stats JSON <https://freeinference.org/public-stats> (the human page <https://freeinference.org/stats> renders it in the browser), signup <https://freeinference.org/signup>, and the gateway's source code <https://github.com/HarvardMadSys/hybridInference> (MIT), where the signup form's second step and the consent text are defined. All read 2026-11-05.

**Who runs it.** "Built at Harvard SEAS · MadSys Lab" (home page). The team page names Juncheng Yang (Assistant Professor, Harvard; lead) and two research interns. The gateway, "HybridInference", is open source (MIT) and routes requests to local vLLM/SGLang servers and to remote providers.

**What it offers.** Inference only. "An OpenAI-compatible API powered by frontier open models, free for the research community" (home page, punctuation adjusted). Base URL `https://freeinference.org/v1` (quick start); the home page also says "OpenAI & Anthropic compatible", and the docs have a Claude Code page using an Anthropic-compatible endpoint. No GPU, notebook or fine-tuning access. Batch inference is "available to internal accounts only" (docs). The docs also list a "Cloud Agents" feature (sandboxed agent tasks), not read here.

**Models** (docs models page; the authenticated `GET /v1/models` is the source of truth):

| Model ID | Access | Context | Notes |
|---|---|---|---|
| glm-5.3-flash | Free | 1M | "Always reasons" |
| deepseek-v4-flash | Free | 1M | agentic coding, reasoning |
| minimax-m3 | Free | 1M | text, image, video |
| qwen3.6-35b | Free | 262K | fastest, non-thinking; max output 8K |
| bge-m3 | Free | 8K | embeddings, `/v1/embeddings` |
| glm-5.2, kimi-k2.7-code | Pro | 1M, 262K | "require a Pro-enabled key"; how to get one is not stated |

The status page at about 13:20 UTC listed 7 of 8 models up (deepseek-v4.1-flash down with HTTP 503) and names that differ slightly from the docs (glm-5.3, minimax-m3.1-flash). Throughput 66 to 264 tokens/s, time to first token 0.18 to 2.3 s.

**Rate limits.** Not published. Home page: "Generous quota for research and prototyping." Terms, section 4: "Quotas, rate limits, model access, and usage limits may change based on usage, demand, infrastructure capacity, abuse prevention ..." and "High-volume, automated, abusive, or operationally risky usage may be limited, delayed, deprioritized, or blocked without advance notice." The gateway lets admins set daily quotas and per-user concurrency (README).

**Eligibility.** Terms, section 2: "You must be at least 18 years old to create an account or use FreeInference." The terms say nothing about Harvard, students, institutions or countries. The signup form's second step shows a notice built from a template in `apps/frontend/src/app/signup/page.tsx`: "Open to {org} students" ... "sign up with your @{domain} email for instant access." ... "Everyone else: please describe your use case below" ... "we review and approve manually." On the live site the organisation is Harvard (as the user saw). Both optional questions are labelled "Both questions are optional, but answers help admins approve accounts faster." Field limits: "How did you find us?" 500 characters; "Use case" 2,000 characters.

**Is a non-Harvard academic allowed?** Yes on paper: nothing in the terms limits use to Harvard, and the stats show accounts from 89 countries. In practice approval is manual and slow to judge: the public stats for 2026-07-31 to 2026-11-05 report "Registrations: 3296 waiting, 1813 approved", 402 accounts with traffic, 13.0 million requests and 1.32 trillion tokens (93.6 % cached input), mostly from coding agents.

**Data use and logging** (quoted; all from <https://freeinference.org/terms> section 5 and 6, or from the consent step at <https://freeinference.org/signup>):

- "All prompts and responses may be logged for research purposes, stored, hashed, redacted, or otherwise processed depending on operator configuration and service needs." (terms 5)
- "anonymized data derived from them, such as sanitized prompts and responses, usage statistics, and routing metrics, may be published or open-sourced to support reproducible research." (terms 5; dash replaced by commas)
- "Sanitization is not a guarantee that all sensitive information will be removed from logs, derived data, or third-party provider systems." (terms 5)
- "Do not submit sensitive personal information, confidential information, regulated data, secrets, credentials, or data you are not authorized to process through the service." (terms 5)
- "Some requests may be forwarded to third-party model providers. Those providers may process submitted prompts, responses, metadata, and related usage information under their own terms and policies." (terms 6)
- "Do not share API keys publicly, embed them in client-side code, or use another person's account without permission." (terms 2)
- Signup requires four confirmations, among them "I consent to participate in this research and to the collection and analysis of my FreeInference usage data." and "I understand and consent to the sharing and possible public release of de-identified research data derived from my FreeInference usage." The consent text adds: "Your sanitized information may be used in future research studies or shared with other researchers for future studies without asking for your informed consent again." and "Once de-identified data have been publicly released, it may no longer be possible to withdraw or delete those copies."
- "The service is provided without guarantee and on an as-is and as-available basis." (terms 7). No retention period is stated.

**What this means for Studio.** Fine for the user's own research scripts and literature tooling on public text (abstracts, open-access papers, own notes). Not suitable for the in-app feature: the key cannot sit in the GitHub Pages bundle, routing end users' model data through the user's account would make them participants in a study they never consented to, and it would break the README's on-device promise. Do not send paywalled full texts either, since sanitised prompts may be released publicly.

---

## 2. Options compared

All rows checked 2026-11-05 at the URL given. "Saudi?" means whether a Saudi-based academic can use it, as far as the provider's page says.

### 2.1 LLM inference APIs (need a)

| Option | Free amount | Models | Saudi academic? | Data terms | API style | Effort | Source |
|---|---|---|---|---|---|---|---|
| FreeInference (Harvard SEAS MadSys) | Free; limits unpublished | GLM-5.3 Flash, DeepSeek V4 Flash, MiniMax M3, Qwen3.6 35B, bge-m3 | Yes after manual review (queue: 3,296 waiting) | All prompts and responses may be logged; sanitised prompts and responses may be published; third-party providers may process them | OpenAI- and Anthropic-compatible | Signup form, then wait | section 1 above |
| Cloudflare Workers AI | 10,000 neurons a day on the Free and Paid plans; on Free, requests fail with an error past the limit; resets 00:00 UTC | e.g. gpt-oss-120b, gpt-oss-20b, Llama 3.3 70B, Qwen 3.8 27B | Not restricted on the page | "Cloudflare does not use Customer Content processed by Workers AI to train AI models"; inputs and outputs not retained after the response unless you turn on logging or caching | OpenAI-compatible at `https://api.cloudflare.com/client/v4/accounts/{id}/ai/v1` | Free account | <https://developers.cloudflare.com/workers-ai/platform/pricing/>, <https://developers.cloudflare.com/workers-ai/platform/data-usage/>, <https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/> |
| Google Gemini API free tier | Free tier on Gemini 3.8 Flash, 3.6 Flash, 3.5 Flash-Lite, 2.5 Pro, 2.5 Flash, Gemma 4 and others; per-model limits shown only inside AI Studio | as listed | Saudi Arabia is an available region; 18+; "not for consumer use" | Unpaid services: content used "to provide, improve, and develop" Google products; "human reviewers may read, annotate, and process your API input and output"; unpaid services may not serve users in the EEA, UK or Switzerland | Gemini API (OpenAI-compatible layer exists; not checked) | Google account, API key | <https://aistudio.google.com/docs/pricing>, <https://aistudio.google.com/docs/terms>, <https://aistudio.google.com/docs/available-regions>, <https://aistudio.google.com/docs/rate-limits> |
| Hugging Face Inference Providers | Free users: none ("Free Users: None"); PRO ($9/month): $2.00 of compute credit a month | 200+ models via partner providers | Not restricted on the page | HF: "We do not store the request body or response"; debug logs up to 30 days; each provider's own policy applies | OpenAI-compatible router `https://router.huggingface.co/v1` | Account (+ PRO) | <https://huggingface.co/docs/inference-providers/pricing>, <https://huggingface.co/docs/inference-providers/security> |
| Hugging Face ZeroGPU (host your own small model in a Space) | Free accounts: 5 min of GPU a day; PRO: 40 min; free accounts in good standing (verified email, account older than 30 days) may host 2 ZeroGPU Spaces | Half or full RTX Pro 6000 Blackwell (48 or 96 GB) | Not restricted | Your own Space | Gradio only; default 60 s per call | Build a Space | <https://huggingface.co/docs/hub/spaces-zerogpu> |
| NVIDIA API catalog (build.nvidia.com) | Trial credits; amount not stated in the terms (third parties say 1,000 credits and 40 RPM, unverified) | NIM-hosted open models | Not stated | Trial terms 1.2: "limited trial purposes only and without use of the API Service or Generated Content in production"; 2.3: no storage at the end of each session, but 3.3(iv): content used "to improve NVIDIA products and services, including AI models"; use logged for security | OpenAI-compatible | NVIDIA Developer account | Trial Terms PDF <https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf>; NIM FAQ <https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317> |
| Groq | A free plan exists, but its numbers sit behind a tab the page did not render; the visible table, labelled Developer-plan base limits, gives gpt-oss-120b 30 RPM, 1K requests/day, 8K TPM, 200K tokens/day (third parties attribute these to the free plan) | gpt-oss-120b, gpt-oss-20b, Llama 3.3 70B, Llama 3.1 8B, Qwen 3.8 27B, Whisper | Not stated | "By default, Groq does not retain customer data for inference requests"; reliability and abuse logs up to 30 days; zero data retention available to all; training not addressed | OpenAI-compatible (`https://api.groq.com/openai/v1`) | Account | <https://console.groq.com/docs/rate-limits>, <https://console.groq.com/docs/your-data>, <https://console.groq.com/docs/models> |
| Cerebras | Free Trial tier: gpt-oss-120b and qwen-3.8-27b at 5 RPM, 1M tokens/day; plus a one-time $5 credit with a payment method, expiring after 30 days | as listed | Not stated | Not stated on these pages; "not intended for production use" | OpenAI-compatible | Account (+ card for the $5) | <https://inference-docs.cerebras.ai/support/rate-limits>, <https://www.cerebras.ai/pricing> |
| SambaNova Cloud | Free tier (no payment method linked): 20 RPM, 20 requests/day, 200K tokens/day per model | DeepSeek V3.1, V3.2, Llama 3.3 70B, gpt-oss-120b, Gemma 4 31B | Not stated | Not stated on the page | OpenAI-compatible | Account | <https://docs.sambanova.ai/docs/en/models/rate-limits> |
| OpenRouter free models (`:free`) | 20 RPM; 50 requests/day if under 10 credits ever bought, 1,000/day with 10 or more | Rotating set of free variants | Not stated | OpenRouter "does not store your prompts or responses" unless you opt in; keeps metadata; providers keep their own policies and some may train; separate account setting for free models | OpenAI-compatible | Account | <https://openrouter.ai/docs/api-reference/limits>, <https://openrouter.ai/docs/guides/privacy/data-collection>, <https://openrouter.ai/docs/guides/privacy/provider-logging> |
| Together AI | No free credit on the pricing page; research credits "invite-only", "small grants to students", "a few hundred dollars" | broad catalogue | Students only, by invitation | Not checked | OpenAI-compatible | Invitation | <https://www.together.ai/pricing>, <https://www.together.ai/research-credits-program-request> |
| Fireworks AI | No free credit or academic programme on the pricing page (only a fine-tuning promotion to November 30) | broad catalogue | n/a | "Zero data retention by default." | OpenAI-compatible | Prepaid credits | <https://fireworks.ai/pricing> |
| Anthropic AI for Science | Support article (dated March 16, 2026): "up to $20,000 in API credits" for "6-month period"; announcement of August 27, 2026: "up to $50,000 in credits per project", "Any researcher is eligible to apply" | Claude models (standard, public models only) | Saudi Arabia is a supported country; no country restriction in the programme pages | API data not used for training by default; API inputs and outputs deleted within 30 days, with stated exceptions | Messages API; key held server-side | Form; reviewed "on the first Monday of each month" | <https://support.claude.com/en/articles/11199177-anthropic-s-ai-for-science-program>, <https://www.anthropic.com/news/expanding-support-for-scientists>, <https://www.anthropic.com/supported-countries>, <https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data>, <https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training> |
| Claude Team plan for scientists | Standard seats $0 for 12 months, premium $15/month; no API allowance | Claude apps, not the API | PIs "or equivalent" at academic and nonprofit institutions "around the world" | Commercial terms | n/a | Verification form, 5 to 7 business days | <https://claude.com/programs/team-plan-for-scientists> |
| Anthropic External Researcher Access | API credits for AI safety and alignment research | n/a | Topic does not fit this project | n/a | n/a | Form | <https://support.claude.com/en/articles/9125743-what-is-the-external-researcher-access-program> |

### 2.2 GPU compute and credit programmes (need b)

| Option | Free amount | GPU | Saudi academic? | Data and terms | Style | Effort | Source |
|---|---|---|---|---|---|---|---|
| Kaggle Notebooks | About 30 GPU hours a week (secondary; Kaggle's own docs did not render), 12 h sessions | T4 x2 or P100 (secondary) | Open accounts; phone verification for GPU (secondary) | Notebooks and datasets can be private | Jupyter | Account, phone check | <https://www.kaggle.com/docs/notebooks> (did not render); secondary: <https://gpuperhour.com/blog/free-cloud-gpus-and-credits> |
| Google Colab free tier | Unpublished: "Colab does not publish these limits"; "access to expensive resources like GPUs is heavily restricted"; "notebooks can run for at most 12 hours"; idle timeouts | Varies ("vary over time") | Open | "Colab resources are not guaranteed and not unlimited" | Jupyter | Google account | <https://research.google.com/colaboratory/intl/en-GB/faq.html> |
| Modal | Starter: "$30 / month free credits"; T4 $0.000164/s (about $0.59/h), L4 $0.000222/s, A100 40 GB $0.000583/s (about $2.10/h), A100 80 GB $0.000694/s, H100 $0.001097/s; so about 50 T4 hours or 14 A100-40GB hours a month | T4 to H100 | Not stated | Your own code and data | Python SDK, scripts and web endpoints | Account (card requirement not checked) | <https://modal.com/pricing> |
| Modal for academics | "Unlock up to $10k of credits"; eligibility and process not stated ("Sign up to hear about our grant programs") | as above | Not stated | as above | as above | Interest form | <https://modal.com/academics> |
| Lightning AI | "up to 50 free GPU hours" on interruptible machines: 5 credits at registration, 25 more after adding a card; credits expire after 12 months; Free plan "Built for students and hobbyists"; 4 h session limit on A100/H100/H200 | T4 to B200 listed | Not stated; phone verification "To prevent abuse" | Not checked | Cloud studio (VS Code, Jupyter) | Account, phone, card | <https://lightning.ai/pricing> |
| Hugging Face ZeroGPU | 5 min/day free, 40 min/day PRO; Gradio only; one call defaults to 60 s | Half RTX Pro 6000 | Open | n/a | Gradio Space | Low | <https://huggingface.co/docs/hub/spaces-zerogpu> (not suited to multi-hour training) |
| Hugging Face Academia Hub | Institutional subscription, "Starting at $10/seat/month", "minimum of 250 annual seats"; $2/month/seat compute credit | n/a | Through the university's IT or procurement | n/a | n/a | Institutional | <https://huggingface.co/docs/hub/academia-hub> |
| Google TPU Research Cloud | Free Cloud TPUs for accepted researchers, granted "on a temporary basis"; other GCP services (VMs, storage) are paid | TPUs only, no GPUs | "Anyone can express interest"; no country list | Must share results publicly and give Google feedback | GCP + PyTorch/XLA, JAX | Interest form | <https://sites.research.google/trc/about/>, <https://sites.research.google/trc/faq/> |
| Google Cloud research credits | "Faculty and Postdoctoral researchers are eligible for one award of up to $5,000 USD"; PhD students "$1000 in GCP Credit per year" | Any GCP GPU | Saudi Arabia is on the eligible country list; faculty, postdocs and PhD students only | GCP terms | Full cloud | 250-word proposal, cost estimate, billing account first | <https://edu.google.com/programs/credits/research/>, <https://support.google.com/google-cloud-higher-ed/answer/10724468?hl=en> |
| AWS Cloud Credit for Research | Faculty and staff: amount by proposal, not capped; students capped at $5,000; rolling | Any EC2 (on-demand, spot) | Faculty, research staff and graduate students at accredited research institutions; no country list | AWS terms | Full cloud | Proposal; "Typical review cycles are 90 to 120 days" | <https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/> |
| Microsoft Azure research credits | Amount not stated; contact form ("Fill out the form and tell us a little about yourself") | Azure | Not stated | Azure terms | Full cloud | Form, then contact | <https://www.microsoft.com/en-us/azure-academic-research/> |
| Azure for Students | "$100 credit" for 12 months, renewable | Azure | "Available only to full-time university students" | Azure terms | Full cloud | School email | <https://azure.microsoft.com/en-us/free/students> |
| NVIDIA Academic Grant Program | Up to 30,000 H100 hours (training track) when open | H100, RTX PRO 6000 | Worldwide, but "full-time faculty member" only | n/a | n/a | Currently "not accepting new applications" | <https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/> |
| Nebius research grants | Amount "determined individually based on each proposal"; GPU credits and tokens | Nebius cloud | Postgraduates, PhD students, postdocs, faculty, researchers at accredited institutions; no country list | Co-marketing expected; identity checks | Cloud | Proposal; monthly review, "Applications are open year-round" | <https://nebius.com/nebius-research-grants> |

### 2.3 Regional routes for a Saudi academic

| Option | What is public | Fit | Source |
|---|---|---|---|
| KAUST Supercomputing Core Lab: Shaheen III CPU partition | External organisations must submit an Organisational Access Application signed by an authorised representative, then a PI project proposal and individual applications; "usage of the system and supporting services will be priced and charged in accordance with KAUST policies"; free only for "KAUST PIs and their team members at KAUST"; users screened under US export rules (Saudi Arabia is listed as a Computer Tier 3 country, which section 4.2.2 admits subject to conditions); progress reports every three months | CPU only; heavy process for tens of GPU hours | KSL procedures (issued October 24, 2024) <https://hpc.kaust.edu.sa/sites/default/files/2024-11/ksl_policy.pdf>; access slides <https://hpc.kaust.edu.sa/sites/default/files/2025-02/2025-02-04-2-Shaheen_III_Access.pdf>; <https://hpc.kaust.edu.sa/node/186> |
| KAUST Ibex GPUs | "For IBEX users external to KAUST, access is permitted to GPU resources only as approved by the Research Computing Allocation Committee ... and after screening"; projects with an external PI submit a proposal to the committee | Possible through a KAUST collaborator; no public quota | as above (section 4.2.4, 5.2.3) |
| Shaheen III GPU partition (GH200) | No external access policy found; reports say KAUST faculty are the first users; HUMAIN and KAUST are "exploring" offering the CPU partition through HUMAIN Cloud, with no agreement yet | Not available now | secondary: <https://www.middleeastainews.com/p/humain-to-share-kausts-shaheen-iii> |
| SDAIA, KACST, HUMAIN | No public compute or API-credit programme for individual researchers found. SDAIA's ALLaM 7B is on Hugging Face as open weights | None found | searches on 2026-11-05 |
| Alfaisal University HPC | Nothing public found on alfaisal.edu | Ask the university's IT or research office | search on 2026-11-05 |

### 2.4 Hosting the FastAPI + TopologicPy backend (need c)

Constraints from [project notes](../deployment): the backend keeps graphs in memory and streams with SSE, so a single long-lived container suits it best; TopologicPy's `topologic-core` 8.0.4 (February 17, 2026) ships Linux wheels for x86_64 only, no aarch64 (<https://pypi.org/project/topologic-core/#files>), which rules out ARM hosts unless it is built from source.

| Option | Free or low-cost amount | CPU / RAM | Sleep and cold start | Notes | Source |
|---|---|---|---|---|---|
| Render free (current) | 750 free instance hours a month | 512 MB; 0.1 CPU (Render article) | Spins down after 15 min idle; spin-up "takes about one minute" | Explains the 74 to 240 s IFC parses | <https://render.com/docs/free>, <https://render.com/pricing>, <https://render.com/articles/from-side-project-to-production-scaling-your-first-app> |
| Render Starter | $7/month | 0.5 CPU (compute plans doc; pricing page says "Less than 1 CPU"), 512 MB | No spin-down | Smallest change; RAM unchanged | <https://render.com/docs/compute-plans> |
| Hugging Face Docker Space on CPU Basic | CPU Basic is free per hour, but "Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts" ($9/month) | 2 vCPU, 16 GB, 50 GB non-persistent disk | Sleeps after 48 h without use; a visit restarts it | Long-lived single container; x86_64; ports 80, 443, 8080 outbound; PRO also brings $2/month credit and 40 min/day ZeroGPU | <https://huggingface.co/docs/hub/spaces-overview>, <https://huggingface.co/docs/hub/spaces-gpus>, <https://huggingface.co/pricing> |
| Google Cloud Run | Free tier: "2 million requests per month", "360,000 GB-seconds of memory, 180,000 vCPU-seconds of compute time" (50 vCPU-hours); "A Google Cloud billing account is required" | Configurable | Scale to zero; cold start includes importing TopologicPy | In-memory graphs are lost when the instance stops; cap at one instance | <https://docs.cloud.google.com/free/docs/free-cloud-features> |
| Modal web endpoint | Shares the $30/month credit; CPU $0.0000131 a physical core-second (about $34 a core-month if always on, before memory) | Configurable | "Booting the first container can take a few seconds" | FastAPI through `@modal.asgi_app()`; same state caveat | <https://modal.com/pricing>, <https://modal.com/docs/guide/webhooks> |
| Oracle Cloud Always Free (Ampere) | Not checked in detail | ARM | n/a | Ruled out: no aarch64 wheels for topologic-core | PyPI as above |
| Google Compute Engine e2-micro | One e2-micro, US regions only | Too small | n/a | Not useful here | <https://docs.cloud.google.com/free/docs/free-cloud-features> |

---

## 3. Recommendations

### (a) Inference

1. **Cloudflare Workers AI** for the in-app "describe this model" feature. No application, a daily free allocation, an OpenAI-compatible endpoint, and terms that rule out training and retention by default. A small Worker can hold the token and proxy calls from the GitHub Pages site, which no provider allows to hold a key itself. Rough capacity (my estimate from the page's neuron prices): with gpt-oss-120b at about 31,818 neurons per million input tokens and 68,182 per million output tokens, a request with 3,000 tokens in and 500 out costs about 130 neurons, so 10,000 neurons is about 75 descriptions a day. Make the feature opt-in, say on screen what is sent, and send the quantities the app has computed (storeys, spaces, exits, route lengths), not the IFC file.
2. **FreeInference** for research scripts and literature tooling, if the account is approved. It is free, offers the largest open models with 1M-token context, and is OpenAI-compatible. Limits: unpublished quotas, no service guarantee, a long approval queue, and prompts that are logged and may be released. Send only public or own text.
3. **Anthropic AI for Science credits**: apply. The programme says any researcher may apply, Saudi Arabia is supported, API data is not used for training by default and is deleted within 30 days, and it would serve both the scripts and a stronger describe feature through the backend. The award is uncertain and is reviewed monthly. Until then, the **Gemini API free tier** works for scripts on public text with no application, with two cautions: free-tier content is used to improve Google products and may be read by human reviewers, and unpaid services may not serve users in the EEA, UK or Switzerland, so it should not back a public feature.

Not recommended for this use: NVIDIA's API catalog (trial only, no production use, content may be used to improve NVIDIA models), SambaNova's free tier (20 requests a day), Hugging Face Inference Providers for a free account (no monthly credit).

### (b) GPU

1. **Kaggle Notebooks.** Free, no application, and a weekly quota (about 30 hours, secondary) that covers a full DGCNN retraining and several ablations in one to two weeks. Twelve-hour sessions mean checkpointing each epoch. The weekly figure could not be read on Kaggle's own page; check it in the account settings.
2. **Modal.** $30 a month of credit buys about 50 T4 hours or 14 A100-40GB hours, enough for the stated need over one to two months, and runs as scripts rather than notebooks, which keeps training reproducible. The academic programme (up to $10k) is an interest form with unstated criteria.
3. **Google Colab free tier** for short checks of the data pipeline and quick runs, not for planned training: GPU type and limits are unpublished and access is "heavily restricted". Lightning AI's free credits (about 50 interruptible GPU hours, card required) are the alternative.

If the work grows, apply for **Google Cloud research credits** first (Saudi Arabia is listed; up to $5,000 for faculty or postdocs, $1,000 a year for PhD students; check which category the user's position falls in), then AWS (90 to 120 day review). KAUST needs an organisation-level agreement or a KAUST collaborator and is not worth it for tens of GPU hours. The binding constraint is the openly licensed training data, not the GPU.

### (c) Hosting

1. **Hugging Face PRO and a Docker Space on CPU Basic, $9 a month.** About 20 times Render free's CPU share and 32 times its memory, sleep after 48 hours instead of 15 minutes, and one long-lived container, which suits the in-memory graphs and SSE. Measure the IFC cross-check there before moving; the speed-up is expected, not measured.
2. **Google Cloud Run free tier.** Free within 50 vCPU-hours a month and scale-to-zero, but needs a billing account and card, loses in-memory state whenever the instance stops, and pays TopologicPy's import on every cold start.
3. **Modal**, if Modal is already used for (b): the same credit pays for both, with the same state caveat as Cloud Run. If the aim is the smallest change, **Render Starter** ($7) removes the spin-down but keeps 512 MB.

---

## 4. Not found or unclear

- FreeInference: per-user rate limits, retention period, and how to get a Pro key are not published. The notice naming Harvard as the fast-track organisation comes from a configured value; the template was read in the source code and the user saw the live text.
- Kaggle's weekly GPU quota and phone-verification rule come from secondary sources; Kaggle's documentation pages render only in a browser.
- Groq's free-plan numbers are behind an unrendered tab; the visible table is labelled Developer plan.
- Gemini free-tier rate limits are shown only inside AI Studio.
- NVIDIA's trial credit amount is not in the trial terms.
- Cerebras and SambaNova data terms were not on the pages read.
- Modal's academic programme criteria and whether its free credit needs a card were not stated.
- No public compute programme was found for SDAIA, KACST or HUMAIN, and no public HPC page for Alfaisal.

---

## 5. Drafts to paste

### 5.1 FreeInference signup: "Use case" (111 words)

> I am a computational design researcher at Alfaisal University, Riyadh, and I develop Topologic Studio, an open-source browser application for fire egress analysis on IFC building models. I would use FreeInference for research scripts and literature tooling: screening and summarising papers for a literature review, extracting structured notes, and testing prompts for an optional feature that describes a building model in plain words from quantities the app has already computed. Tests would use my own or openly licensed sample models, not users' files. Volume would be low, at most a few hundred requests a week. I understand that prompts are logged for research and will not send personal or confidential data.

### 5.2 FreeInference signup: "How did you find us?" (one line)

> A web search for free LLM APIs for academic research.

(Replace with the real source if it was a colleague, a post or a course link.)

### 5.3 Application paragraph for GPU or credit programmes (177 words)

> Topologic Studio is an open-source browser application for fire egress analysis on IFC building models, developed at Alfaisal University, Riyadh. It turns a model's floors, stairs and doors into a walkable graph, finds routes to the exits and shows how a spreading fire changes them. An optional fire solver, after the Flamescapes paper, runs on the user's GPU through WebGPU, and properties of the fire and egress models are checked in Lean 4. The project needs GPU time to retrain a point-cloud classifier for IFC elements (the DGCNN network used in IFCNet) on openly licensed geometry, since the published weights carry no licence; the new weights and training code would be released openly. It also needs low-volume LLM inference for an optional feature that describes a loaded model in plain words, grounded in quantities the app has already computed, and for research scripts. Expected use is modest: about 30 to 60 GPU hours on a T4 or A100 class GPU, including ablations, and a few thousand inference requests over the project. Code and results will be public.

Before sending, check two figures against your own plan: the request volume in 5.1 and the GPU hours in 5.3. Both are estimates, not measurements.
