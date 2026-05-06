# Playwright Console Harness

Use this temporary harness to capture console output, page errors, failed requests, API statuses, and a screenshot.

```javascript
import { chromium } from "playwright";

const url = process.env.TEST_URL || "http://localhost:5173";
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });

const events = [];
page.on("console", msg => {
  events.push({ type: "console", level: msg.type(), text: msg.text() });
});
page.on("pageerror", err => {
  events.push({ type: "pageerror", text: err.message });
});
page.on("requestfailed", req => {
  events.push({ type: "requestfailed", url: req.url(), failure: req.failure()?.errorText });
});
page.on("response", res => {
  const u = res.url();
  if (u.includes("localhost:8000") || u.includes("/ifc-") || u.includes("/fire-sim")) {
    events.push({ type: "response", status: res.status(), url: u });
  }
});

await page.goto(url, { waitUntil: "networkidle" });
await page.screenshot({ path: "wiki/verification/headless-home.png", fullPage: true });
console.log(JSON.stringify(events, null, 2));
await browser.close();
```

Do not commit temporary scripts unless the user asks. Record summarized output in [verification status](../issues/verification-status.md) or the relevant issue page.

