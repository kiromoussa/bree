// Headless Firefox screenshots through WebDriver BiDi (no npm packages; node >= 22).
// usage: node scripts/product_shot.js <firefox-binary> <profile-dir> <width> <height> <out.png> <url> [js-to-run-before] [wait-ms]
const {spawn} = require("child_process");
const fs = require("fs");
const [bin, prof, W, H, out, url, pre, waitMs] = process.argv.slice(2);
const port = 9300 + Math.floor(Math.random() * 500);
fs.mkdirSync(prof, {recursive: true});
const ff = spawn(bin, ["--headless", "--no-remote", "--profile", prof, "--remote-debugging-port", String(port)], {stdio: "ignore"});
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  let ws;
  for (let i = 0; i < 60; i++) { try { ws = await new Promise((res, rej) => { const w = new WebSocket(`ws://127.0.0.1:${port}/session`); w.onopen = () => res(w); w.onerror = rej; }); break; } catch (e) { await sleep(500); } }
  if (!ws) throw new Error("no BiDi");
  let id = 0; const pend = {};
  ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && pend[d.id]) pend[d.id](d); };
  const cmd = (method, params = {}) => new Promise(res => { const i = ++id; pend[i] = res; ws.send(JSON.stringify({id: i, method, params})); });
  await cmd("session.new", {capabilities: {}});
  const tree = await cmd("browsingContext.getTree", {});
  const ctx = tree.result.contexts[0].context;
  await cmd("browsingContext.setViewport", {context: ctx, viewport: {width: +W, height: +H}, devicePixelRatio: 1});
  await cmd("browsingContext.navigate", {context: ctx, url, wait: "complete"});
  await sleep(+(waitMs || 1500));
  if (pre) { await cmd("script.evaluate", {expression: pre, target: {context: ctx}, awaitPromise: true}); await sleep(+(waitMs || 1500)); }
  const full = await cmd("script.evaluate", {expression: "document.documentElement.scrollHeight", target: {context: ctx}, awaitPromise: false});
  const h = Math.min(+H > 0 ? Math.max(+H, full.result.result.value) : full.result.result.value, 6000);
  await cmd("browsingContext.setViewport", {context: ctx, viewport: {width: +W, height: h}, devicePixelRatio: 1});
  await sleep(300);
  const shot = await cmd("browsingContext.captureScreenshot", {context: ctx, origin: "viewport"});
  fs.writeFileSync(out, Buffer.from(shot.result.data, "base64"));
  ff.kill(); process.exit(0);
})().catch(e => { console.error(e); ff.kill(); process.exit(1); });
