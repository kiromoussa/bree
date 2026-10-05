#!/usr/bin/env node
// Drive the PRIVATE copy of the browser store simulator headless and write SIMULATED camera frames with
// ground truth. Two modes:
//
//   dataset   node scripts/train/render_synth.mjs --out data/synth/frames --seeds 1000:1160
//             per seed: one randomised scene (scripts/train/sim/synth.js), a few capture moments, a few cameras
//             each -> <out>/s<seed>/<k>_<camera>.jpg + .json (boxes, SKU, pixels across a can, visibility)
//   clip      node scripts/train/render_synth.mjs --clip --seed 5001 --out data/synth/clip_5001
//             one scenario rendered as video frames in the folder shape `make sim-eval` reads
//             (<camera>/rgb_NNNN.png, events.jsonl, layout.json, config.yaml) plus truth_frames.jsonl (boxes of items in a hand)
//
// The copy lives in out/sim-copy (made from --sim-src on first use; the original simulator is never run or edited).
// Needs Chrome, Playwright (BREE_PLAYWRIGHT=/path/to/node_modules/playwright) and network for three.js (CDN).
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url)), REPO = path.resolve(HERE, '../..');
const argv = process.argv.slice(2), a = { _: [] };
for (let i = 0; i < argv.length; i++) { const k = argv[i]; if (!k.startsWith('--')) { a._.push(k); continue; } const n = argv[i + 1]; if (n === undefined || n.startsWith('--')) a[k.slice(2)] = true; else { a[k.slice(2)] = n; i++; } }
const home = process.env.HOME;
const simSrc = a['sim-src'] ?? path.join(home, 'bree/software/sim-prototype');
const simDir = path.resolve(a.sim ?? path.join(REPO, 'out/sim-copy'));
const layoutFile = a.layout ?? path.join(home, 'bree/software/shared/layouts/recommended-47.json');
if (!a.out || a.help) { console.error('usage: render_synth.mjs --out DIR (--seeds A:B | --clip --seed N) [--layout f.json] [--captures 4] [--hand 3] [--rand 3] [--shoppers 10]\n       clip: [--fps 10] [--cams id,id] [--max-cams 6] [--max-seconds 90] [--no-jitter]'); process.exit(2); }

// Which simulator version the copy is: recorded when the copy is made, written into every clip.json and into
// <out>/sim_copy.json for a dataset (dataset.py puts it in meta.json). The copy is only made on first use, so
// it can be older than the simulator next door: delete out/sim-copy to render from the current one.
const prov = path.join(simDir, 'COPIED_FROM.json');
if (!fs.existsSync(path.join(simDir, 'index.html'))) {
  fs.cpSync(simSrc, simDir, { recursive: true });
  const git = (...args) => { try { return execFileSync('git', ['-C', simSrc, ...args], { encoding: 'utf8' }).trim(); } catch { return null; } };
  fs.writeFileSync(prov, JSON.stringify({ source: simSrc, commit: git('log', '-1', '--format=%h %cI', '--', '.') ?? 'not a git checkout',
    uncommitted_changes: !!git('status', '--porcelain', '--', '.'), copied_at: new Date().toISOString() }, null, 1));
  console.error(`copied ${simSrc} -> ${simDir}`);
}
const simInfo = fs.existsSync(prov) ? JSON.parse(fs.readFileSync(prov, 'utf8')) : { source: simSrc, commit: 'not recorded (copied before 2026-10-05)' };
fs.copyFileSync(path.join(HERE, 'sim/synth.js'), path.join(simDir, 'js/synth.js'));
const { openSim } = await import(path.join(simDir, 'tools/lib.mjs'));
const layout = JSON.parse(fs.readFileSync(layoutFile, 'utf8'));
const s = await openSim({ headed: !!a.headed }), page = s.page;
await page.addScriptTag({ type: 'module', url: '/js/synth.js' });
await page.waitForFunction(() => window.__breeSynth, null, { timeout: 30000 });
const call = (fn, arg) => page.evaluate(([f, x]) => (0, eval)(`(${f})`)(window.__breeSynth, x), [fn.toString(), arg]);
const save = (file, dataUrl) => fs.writeFileSync(file, Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64'));
const out = path.resolve(a.out); fs.mkdirSync(out, { recursive: true });
const t0 = Date.now();

try {
  if (a.clip) {
    const seed = +(a.seed ?? 5001), fps = +(a.fps ?? 10), maxS = +(a['max-seconds'] ?? 90), maxCams = +(a['max-cams'] ?? 6);
    const opts = { shoppers: +(a.shoppers ?? 4), window: +(a.window ?? 12), theft: +(a.theft ?? 0.5), jitter: !a['no-jitter'] };
    // dry run: what happens, how long it takes, which cameras see it
    const dry = await call((y, { seed, layout, opts, maxS }) => { y.setup(seed, layout, opts); let st; do st = y.advance(y.state().t + 1); while ((st.inStore || st.pending) && st.t < maxS); return { end: st.t, events: y.events(), cams: y.layout().cameras }; }, { seed, layout, opts, maxS });
    let cams = a.cams ? a.cams.split(',') : null;
    if (!cams) { // the widest checkout camera and the widest overhead (register, door, floor), then the best item camera of each pick, up to maxCams
      const best = dry.events.map(e => e.cameras[0]?.id).filter(Boolean), widest = k => dry.cams.filter(c => c.kind === k).sort((x, y) => y.hfov - x.hfov).slice(0, 1).map(c => c.id);
      cams = [...new Set([...widest('checkout'), ...widest('overhead'), ...best])].slice(0, maxCams);
    }
    const params = await call((y, { seed, layout, opts }) => y.setup(seed, layout, opts), { seed, layout, opts });
    const n = Math.ceil(dry.end * fps);
    for (const c of cams) fs.mkdirSync(path.join(out, c), { recursive: true });
    const truth = [];
    for (let i = 0; i < n; i++) {
      await call((y, t) => y.advance(t), i / fps);
      for (const c of cams) { // colour frame, plus ground-truth boxes of items in a hand or on the counter (for the product-recognition check)
        const r = await call((y, c) => y.capture(c, { type: 'image/png' }), c);
        save(path.join(out, c, `rgb_${String(i + 1).padStart(4, '0')}.png`), r.image);
        const items = r.annotations.filter(x => x.kind === 'hand' || x.kind === 'counter').map(({ sku, kind, bbox, vis_px, px_eff, shopper }) => ({ sku, kind, bbox, vis_px, px_eff, shopper }));
        if (items.length) truth.push(JSON.stringify({ frame: i, t: +(i / fps).toFixed(3), camera: c, items }) + '\n');
      }
      if (i % 50 === 0) console.error(`frame ${i + 1}/${n}`);
    }
    await call((y, t) => y.advance(t), dry.end);
    const ev = await call((y, fps) => y.events(fps), fps);
    fs.writeFileSync(path.join(out, 'events.jsonl'), ev.map(e => JSON.stringify(e) + '\n').join(''));
    fs.writeFileSync(path.join(out, 'truth_frames.jsonl'), truth.join(''));
    fs.writeFileSync(path.join(out, 'layout.json'), JSON.stringify(await call(y => y.layout()), null, 1));
    fs.writeFileSync(path.join(out, 'config.yaml'), `# written by render_synth.mjs (browser simulator, SIMULATED)\nisaacsim.replicator.agent:\n  simulation_duration: ${(n / fps).toFixed(3)}\n`);
    fs.writeFileSync(path.join(out, 'clip.json'), JSON.stringify({ source: 'browser simulator copy (SIMULATED)', simulator: simInfo, seed, fps, frames: n, cameras: cams, params, events: ev.length }, null, 1));
    console.error(`clip: ${n} frames x ${cams.length} cameras (${cams.join(', ')}), ${ev.length} picks, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  } else {
    const [s0, s1] = String(a.seeds ?? '1000:1010').split(':').map(Number);
    const nCap = +(a.captures ?? 4), nHand = +(a.hand ?? 3), nRand = +(a.rand ?? 3), shoppers = +(a.shoppers ?? 10);
    fs.writeFileSync(path.join(out, 'skus.json'), JSON.stringify(await call(y => y.skus), null, 1));
    fs.writeFileSync(path.join(out, 'sim_copy.json'), JSON.stringify(simInfo, null, 1));
    let frames = 0, anns = 0;
    for (let seed = s0; seed < s1; seed++) {
      const dir = path.join(out, `s${seed}`);
      if (fs.existsSync(path.join(dir, 'done'))) continue;
      fs.mkdirSync(dir, { recursive: true });
      const params = await call((y, { seed, layout, shoppers }) => y.setup(seed, layout, { shoppers }), { seed, layout, shoppers });
      for (let k = 0; k < nCap; k++) {
        // capture moments spread over the visit: shoppers are walking in, reaching, carrying, paying
        const pick = await call((y, { t, nHand, nRand }) => { y.advance(t); return y.pickCameras(nHand, nRand); }, { t: 9 + k * 7 + (seed % 5), nHand, nRand });
        for (const [why, cam] of [...pick.hand.map(c => ['hand', c]), ...pick.random.map(c => ['random', c])]) {
          const r = await call((y, cam) => y.capture(cam), cam), stem = path.join(dir, `${k}_${cam}`);
          save(stem + '.jpg', r.image); delete r.image;
          fs.writeFileSync(stem + '.json', JSON.stringify({ seed, capture: k, chosen_for: why, params, ...r }));
          frames++; anns += r.annotations.length;
        }
      }
      fs.writeFileSync(path.join(dir, 'done'), '');
      if ((seed - s0) % 10 === 0) console.error(`seed ${seed}: ${frames} frames, ${anns} boxes, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    }
    console.error(`dataset: seeds ${s0}..${s1 - 1}, ${frames} new frames, ${anns} boxes, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  if (s.errors.length) { console.error('console errors:\n' + s.errors.join('\n')); process.exitCode = 1; }
} finally { await s.close(); }
