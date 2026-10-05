#!/usr/bin/env node
// SIMULATED training and evaluation frames for the hand and held-item detector, from the PRIVATE simulator copy.
//
//   node scripts/train/render_hands.mjs --out data/synth/hands/frames --seeds 2000:2060
//
// Per seed: the benchmark's scripted scenario (scripts/bench/sim/bench.js: reach, browse, put back, hold, conceal, counter,
// two shoppers at one shelf) in a randomised scene (scripts/train/sim/synth.js), stepped every --step seconds. At each
// step the cameras that see an arm that is up, a held item or an item on the counter are rendered at full resolution
// (scripts/train/sim/hands.js chooses) -> <out>/s<seed>/<k>_<camera>.jpg + .json: every item box with its SKU, every
// visible hand box, the people. Same file shape as render_synth.mjs, so bree.train.dataset reads both.
//
// SEEDS: only TRAIN-range seeds of scripts/bench/manifest.json (1000:5000). The script refuses any other seed.
// Needs Chrome, Playwright (BREE_PLAYWRIGHT=/path/to/node_modules/playwright) and network for three.js (CDN).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url)), REPO = path.resolve(HERE, '../..');
const argv = process.argv.slice(2), a = {};
for (let i = 0; i < argv.length; i++) { const k = argv[i], n = argv[i + 1]; if (n === undefined || n.startsWith('--')) a[k.slice(2)] = true; else { a[k.slice(2)] = n; i++; } }
if (!a.out || !a.seeds) { console.error('usage: render_hands.mjs --out DIR --seeds A:B [--layout f.json] [--step 0.3] [--p-act 0.7] [--p-carry 0.25] [--max-seconds 120]'); process.exit(2); }
const [s0, s1] = String(a.seeds).split(':').map(Number);
const manifest = JSON.parse(fs.readFileSync(path.join(REPO, 'scripts/bench/manifest.json'), 'utf8'));
const [t0s, t1s] = manifest.splits.train.seeds.split(':').map(Number);
if (!(s0 >= t0s && s1 <= t1s && s0 < s1)) { console.error(`seeds ${a.seeds} are not inside the benchmark TRAIN range ${t0s}:${t1s}`); process.exit(2); }
const simDir = path.join(REPO, 'out/sim-copy'), layouts = path.join(process.env.HOME, 'bree/software/shared/layouts');
const layoutFile = a.layout ?? path.join(layouts, 'recommended-3d-45.json');
for (const [src, name] of [[path.join(HERE, 'sim/synth.js'), 'synth.js'], [path.join(REPO, 'scripts/bench/sim/bench.js'), 'bench.js'], [path.join(HERE, 'sim/hands.js'), 'hands.js']]) {
  const dst = path.join(simDir, 'js', name), buf = fs.readFileSync(src);      // other renders may be running: only write when it differs
  if (!fs.existsSync(dst) || !fs.readFileSync(dst).equals(buf)) fs.writeFileSync(dst, buf);
}
const { openSim } = await import(path.join(simDir, 'tools/lib.mjs'));
const layout = JSON.parse(fs.readFileSync(layoutFile, 'utf8'));
const out = path.resolve(a.out); fs.mkdirSync(out, { recursive: true });
const step = +(a.step ?? 0.3), maxS = +(a['max-seconds'] ?? 120), opts = { pAct: +(a['p-act'] ?? 0.7), pCarry: +(a['p-carry'] ?? 0.25), pRandom: +(a['p-random'] ?? 0.06) };
const s = await openSim({ headed: !!a.headed }), page = s.page, t0 = Date.now();
try {
  for (const [m, g] of [['synth', '__breeSynth'], ['bench', '__breeBench'], ['hands', '__breeHands']]) { await page.addScriptTag({ type: 'module', url: `/js/${m}.js` }); await page.waitForFunction(g => window[g], g, { timeout: 30000 }); }
  const call = (fn, arg) => page.evaluate(([f, x]) => (0, eval)(`(${f})`)(window.__breeHands, x), [fn.toString(), arg]);
  fs.writeFileSync(path.join(out, 'skus.json'), JSON.stringify(await call(y => y.skus), null, 1));
  fs.writeFileSync(path.join(out, 'sim_copy.json'), JSON.stringify({ ...JSON.parse(fs.readFileSync(path.join(simDir, 'COPIED_FROM.json'), 'utf8')), layout: path.basename(layoutFile), scenario: 'scripts/bench/sim/bench.js', overlay: 'scripts/train/sim/hands.js' }, null, 1));
  let frames = 0;
  for (let seed = s0; seed < s1; seed++) {
    const dir = path.join(out, `s${seed}`);
    if (fs.existsSync(path.join(dir, 'done'))) continue;
    fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir, { recursive: true });
    const params = await call((y, { seed, layout }) => y.setup(seed, layout, { window: 14 + seed % 3 * 8 }), { seed, layout });
    for (let k = 0, st = { pending: 1 }; (st.inStore || st.pending) && k * step < maxS; k++) {
      const todo = await call((y, { t, opts }) => { y.advance(t); return y.plan(opts); }, { t: k * step, opts });
      for (const { cam, why } of todo) {
        const r = await call((y, cam) => y.capture(cam), cam), stem = path.join(dir, `${k}_${cam}`);
        fs.writeFileSync(stem + '.jpg', Buffer.from(r.image.slice(r.image.indexOf(',') + 1), 'base64')); delete r.image;
        fs.writeFileSync(stem + '.json', JSON.stringify({ seed, capture: k, chosen_for: why, params, ...r }));
        frames++;
      }
      st = await call(y => y.state());
    }
    fs.writeFileSync(path.join(dir, 'done'), '');
    console.error(`seed ${seed}: ${frames} frames so far, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  if (s.errors.length) { console.error('console errors:\n' + s.errors.join('\n')); process.exitCode = 1; }
} finally { await s.close(); }
