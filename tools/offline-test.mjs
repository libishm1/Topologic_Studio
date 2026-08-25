import { chromium } from "playwright-core";
const b = await chromium.launch({executablePath:"C:/Program Files/Google/Chrome/Application/chrome.exe", headless:true, args:["--enable-unsafe-swiftshader"]});
const p = await b.newPage({viewport:{width:1400,height:900}});
let fail=0; const check=(ok,l,d="")=>{ if(!ok) fail++; console.log(`  ${ok?"PASS":"FAIL"}  ${l}${d?"  — "+d:""}`); };

await p.goto("http://localhost:5173", {waitUntil:"domcontentloaded"});
await p.waitForFunction(()=>!document.body.innerText.includes("Starting the 3D engine"),null,{timeout:90000});
await p.waitForTimeout(1500);

const badge = async () => (await p.locator(".statusbar").innerText()).replace(/\s+/g," ").trim();
check((await badge()).includes("backend up"), "starts as 'backend up'", await badge());

// Simulate the server going away, without touching the real one.
console.log("\n  blocking all requests to :8000 ...");
await p.route("**://*:8000/**", (route) => route.abort("connectionrefused"));

// Any action that hits the API should now flip the indicator.
await p.setInputFiles('input[type=file][accept=".ifc"]', "../Ifc2x3_Duplex_Architecture.ifc");
await p.waitForFunction(()=>document.body.innerText.toLowerCase().includes("floors"),null,{timeout:120000});
await p.waitForTimeout(800);
await p.getByRole("button", {name:/Build egress graph|Rebuild graph/}).first().click();
await p.waitForTimeout(2500);

const wentOffline = await p.waitForFunction(
  () => document.querySelector(".statusbar")?.innerText.includes("backend down"),
  null, {timeout:15000}).then(()=>true).catch(()=>false);
check(wentOffline, "status flips to 'backend down' after a failed call", await badge());

// Several toasts can be on screen; search them all rather than assuming order.
const toasts = (await p.locator(".toast").allInnerTexts()).join(" | ");
check(/cannot reach the backend/i.test(toasts), "a toast names the unreachable backend",
  toasts.replace(/\s+/g," ").slice(0,120));

// Let it come back.
console.log("\n  unblocking :8000 ...");
await p.unroute("**://*:8000/**");
const recovered = await p.waitForFunction(
  () => document.querySelector(".statusbar")?.innerText.includes("backend up"),
  null, {timeout:20000}).then(()=>true).catch(()=>false);
check(recovered, "status recovers on its own once the server is back", await badge());

console.log(`\n  ${fail} failed\n`);
await b.close();
process.exit(fail?1:0);
