#!/usr/bin/env node
// Render one benchmark clip from the PRIVATE simulator copy (out/sim-copy). SIMULATED DATA.
//
//   node scripts/bench/render_clip.mjs --seed 7001 --out data/synth/bench/dev/clip_7001
//   options: --layout f.json  --fps 10  --dry (scenario and cameras only, no frames)  --from S --to S (render a slice)
//            --stills N (also keep every Nth frame as a JPEG in <out>/stills)  --max-item-cams 12  --spare-cams 2  --crf 16
//
// What the pipeline may read (top level of <out>):
//   <camera>.mp4       H.264, constant frame rate; frame i is sim time i / fps
//   calibration.json   true pose and intrinsics of every rendered camera
//   layout.json        store, fixtures, slots with their SKU (the planogram), all cameras of the layout
//   register.jsonl     receipts from the register (pipeline payment format), with a delay after the payment
//   clip.json          seed, fps, frames, cameras, how they were chosen, simulator version
// Ground truth, for scoring and (TRAIN seeds only) for training data, in <out>/truth:
//   events.jsonl  shoppers.json  tracks.jsonl (floor position per frame)  frames.jsonl (boxes per camera and frame)
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url)), REPO = path.resolve(HERE, '../..');
const argv = process.argv.slice(2), a = {};
for (let i = 0; i < argv.length; i++) { const k = argv[i], n = argv[i + 1]; if (n === undefined || n.startsWith('--')) a[k.slice(2)] = true; else { a[k.slice(2)] = n; i++; } }
if (!a.out || a.seed === undefined) { console.error('usage: render_clip.mjs --seed N --out DIR [--layout f.json] [--fps 10] [--dry] [--from S --to S] [--stills N]'); process.exit(2); }
const simDir = path.join(REPO, 'out/sim-copy'), layouts = path.join(process.env.HOME, 'bree/software/shared/layouts');
const layoutFile = a.layout ?? [path.join(layouts, 'recommended-3d-45.json'), path.join(layouts, 'recommended-47.json')].find(f => fs.existsSync(f));
if (!fs.existsSync(path.join(simDir, 'index.html'))) { console.error(`no simulator copy at ${simDir}: see scripts/bench/README.md (refresh the copy first)`); process.exit(2); }
for (const [src, name] of [[path.join(REPO, 'scripts/train/sim/synth.js'), 'synth.js'], [path.join(HERE, 'sim/bench.js'), 'bench.js']]) {
  const dst = path.join(simDir, 'js', name), buf = fs.readFileSync(src);      // other renders may be running: only write when it differs
  if (!fs.existsSync(dst) || !fs.readFileSync(dst).equals(buf)) fs.writeFileSync(dst, buf);
}
const sha = f => execFileSync('shasum', ['-a', '256', f], { encoding: 'utf8' }).slice(0, 12);
const simInfo = { ...JSON.parse(fs.readFileSync(path.join(simDir, 'COPIED_FROM.json'), 'utf8')), synth_js: sha(path.join(simDir, 'js/synth.js')), bench_js: sha(path.join(simDir, 'js/bench.js')) };
const { openSim } = await import(path.join(simDir, 'tools/lib.mjs'));
const layout = JSON.parse(fs.readFileSync(layoutFile, 'utf8'));
const seed = +a.seed, fps = +(a.fps ?? 10), out = path.resolve(a.out), t0 = Date.now();
const mulberry32 = s => () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };

const s = await openSim({ headed: !!a.headed }), page = s.page;
try {
  for (const m of ['synth', 'bench']) { await page.addScriptTag({ type: 'module', url: `/js/${m}.js` }); await page.waitForFunction(m => window[m === 'synth' ? '__breeSynth' : '__breeBench'], m, { timeout: 30000 }); }
  const call = (fn, arg) => page.evaluate(([f, x]) => (0, eval)(`(${f})`)(window.__breeBench, x), [fn.toString(), arg]);
  // 1. dry runs: the arrival window that makes the visit 60 to 90 s long, what happens, which cameras see it
  const dryRun = window => call((y, { seed, layout, window }) => { const params = y.setup(seed, layout, { window }); let st; do st = y.advance(y.state().t + 1); while ((st.inStore || st.pending) && st.t < 240); return { end: st.t, params, events: y.events(10), cams: y.layout().cameras, shoppers: y.shoppers() }; }, { seed, layout, window });
  let dry = null, window = null;
  for (const w of a.window ? [+a.window] : [30, 22, 38, 16, 46, 10, 4]) { const d = await dryRun(w); if (!dry || Math.abs(d.end - 75) < Math.abs(dry.end - 75)) { dry = d; window = w; } if (d.end >= 60 && d.end <= 90) { dry = d; window = w; break; } }
  const kind = Object.fromEntries(dry.cams.map(c => [c.id, c.kind])), R = mulberry32(seed * 4421 + 17);
  // cameras: every entrance and overhead camera and the register camera; the best item camera of every pick, then second
  // views (two-view 3D) while there is room; then a few item cameras that see no pick
  const fixed = dry.cams.filter(c => c.kind === 'entrance' || c.kind === 'overhead' || c.id === 'REGISTER-top').map(c => c.id);
  const isItem = id => !fixed.includes(id) && ['shelf', 'cooler', 'checkout'].includes(kind[id]);
  const views = dry.events.map(e => e.cameras.filter(v => isItem(v.id) && v.px >= 8).map(v => v.id));
  const maxItem = +(a['max-item-cams'] ?? 12), item = [];
  for (const rank of [0, 1]) for (const v of views) if (v[rank] && !item.includes(v[rank]) && (rank === 0 || item.length < maxItem)) item.push(v[rank]);   // the best view of a pick is never dropped
  const seesPick = new Set(dry.events.flatMap(e => e.cameras.map(v => v.id)));
  const idle = dry.cams.map(c => c.id).filter(id => isItem(id) && !seesPick.has(id)).sort(() => R() - 0.5).slice(0, +(a['spare-cams'] ?? 2));
  const cams = a.cams ? a.cams.split(',') : [...fixed, ...item, ...idle];
  const n = Math.ceil((dry.end + 1) * fps);
  const summary = { seed, window, sim_seconds: +dry.end.toFixed(2), frames: n, shoppers: dry.shoppers.length, thieves: dry.shoppers.filter(x => x.thief).length,
    picks: dry.events.length, outcomes: dry.events.reduce((o, e) => (o[e.outcome] = (o[e.outcome] ?? 0) + 1, o), {}), zones: dry.events.reduce((o, e) => (o[e.zone] = (o[e.zone] ?? 0) + 1, o), {}),
    picks_without_an_item_camera: views.filter(v => !v.length).length, picks_with_two_rendered_views: views.filter(v => v.filter(c => cams.includes(c)).length >= 2).length, cameras: cams };
  console.error(JSON.stringify(summary));
  if (a.dry) { console.log(JSON.stringify({ ...summary, events: dry.events, shoppers: dry.shoppers }, null, 1)); }
  else {
    // 2. render
    fs.mkdirSync(path.join(out, 'truth'), { recursive: true });
    if (a.stills) fs.mkdirSync(path.join(out, 'stills'), { recursive: true });
    const params = await call((y, { seed, layout, window }) => y.setup(seed, layout, { window }), { seed, layout, window });
    const enc = Object.fromEntries(cams.map(c => { // one H.264 encoder per camera, fed JPEG frames
      const p = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', String(fps), '-i', '-', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', String(a.crf ?? 16), '-pix_fmt', 'yuv420p', '-g', String(fps), path.join(out, `${c}.mp4`)], { stdio: ['pipe', 'inherit', 'inherit'] });
      return [c, p];
    }));
    const f0 = a.from ? Math.floor(+a.from * fps) : 0, f1 = a.to ? Math.min(n, Math.ceil(+a.to * fps)) : n;
    const frames = fs.createWriteStream(path.join(out, 'truth/frames.jsonl')), tracks = fs.createWriteStream(path.join(out, 'truth/tracks.jsonl')), last = {};
    let reused = 0;
    for (let i = f0; i < f1; i++) {
      const t = i / fps, k = i - f0;
      const w = await call((y, t) => { y.advance(t); return y.world(); }, t);
      tracks.write(JSON.stringify({ frame: k, t: +(k / fps).toFixed(3), shoppers: w }) + '\n');
      for (const c of cams) {
        let r = await call((y, c) => y.capture(c), c);          // { same: true }: nothing moved in view, repeat the last frame
        if (r.same) { r = last[c]; reused++; } else { r.buf = Buffer.from(r.image.slice(r.image.indexOf(',') + 1), 'base64'); delete r.image; last[c] = r; }
        const buf = r.buf;
        if (!enc[c].stdin.write(buf)) await new Promise(ok => enc[c].stdin.once('drain', ok));
        if (a.stills && k % +a.stills === 0) fs.writeFileSync(path.join(out, 'stills', `${c}_${String(k).padStart(4, '0')}.jpg`), buf);
        if (r.persons.length || r.items.length) frames.write(JSON.stringify({ frame: k, t: +(k / fps).toFixed(3), camera: c, persons: r.persons, items: r.items }) + '\n');
      }
      if (k % 50 === 0) console.error(`seed ${seed}: frame ${k + 1}/${f1 - f0}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    }
    await Promise.all([...Object.values(enc).map(p => new Promise(ok => { p.on('close', ok); p.stdin.end(); })), new Promise(ok => frames.end(ok)), new Promise(ok => tracks.end(ok))]);
    await call((y, t) => y.advance(t), dry.end + 1);
    const off = f0 / fps, sh = x => x === null || x === undefined ? x : +(x - off).toFixed(3);     // a slice starts at its own zero
    const ev = (await call((y, fps) => y.events(fps), fps)).map(e => ({ ...e, t: sh(e.t), frame: e.frame - f0, tResolved: sh(e.tResolved), tExit: sh(e.tExit), tConceal: sh(e.tConceal), tPay: sh(e.tPay), tPutBack: sh(e.tPutBack) }));
    const shoppers = (await call(y => y.shoppers())).map(x => ({ ...x, tEnter: sh(x.tEnter), tExit: sh(x.tExit) }));
    // register: one receipt per paying shopper, 1.5 to 4.5 s after the payment (the sale closes, the receipt prints)
    const paid = {}; for (const e of ev) if (e.outcome === 'paid') (paid[e.shopper] ??= []).push(e);
    const receipts = Object.values(paid).sort((x, y) => x[0].tPay - y[0].tPay).map((es, k) => {
      const items = {}; for (const e of es) items[e.skuId] = (items[e.skuId] ?? 0) + 1;
      return { t: +(Math.max(...es.map(e => e.tPay)) + 1.5 + 3 * R()).toFixed(2), terminal: 'pos_1', txn_id: `SIM${String(k + 1).padStart(4, '0')}`, items: Object.entries(items).map(([sku, qty]) => ({ sku, qty })) };
    }).sort((x, y) => x.t - y.t);
    const w = (f, rows) => fs.writeFileSync(path.join(out, f), rows.map(r => JSON.stringify(r) + '\n').join(''));
    w('truth/events.jsonl', ev); w('register.jsonl', receipts);
    fs.writeFileSync(path.join(out, 'truth/shoppers.json'), JSON.stringify(shoppers, null, 1));
    fs.writeFileSync(path.join(out, 'layout.json'), JSON.stringify(await call(y => y.layout()), null, 1));
    const calib = (await call(y => y.calibration())).filter(c => cams.includes(c.id));
    fs.writeFileSync(path.join(out, 'calibration.json'), JSON.stringify({ frame: 'store metres, y up, +z toward the front door (the layout frame)', model: 'pinhole, no distortion: pixel = K R (X - position); R rows are right, down, forward', cameras: calib }, null, 1));
    const seconds = +((Date.now() - t0) / 1000).toFixed(0);
    fs.writeFileSync(path.join(out, 'clip.json'), JSON.stringify({ source: 'browser simulator copy (SIMULATED)', simulator: simInfo, layout: path.basename(layoutFile), seed, fps, frames: f1 - f0, sim_seconds: +((f1 - f0) / fps).toFixed(2),
      slice: a.from || a.to ? [f0 / fps, f1 / fps] : null, cameras: cams, camera_choice: { always: fixed, item_cameras_that_see_a_pick: item, item_cameras_that_see_no_pick: idle }, params, video: `H.264 (libx264 crf ${a.crf ?? 16}) from JPEG quality 0.92; a camera with nothing moving in view repeats its last frame` }, null, 1));
    fs.writeFileSync(path.join(out, 'truth/render.json'), JSON.stringify({ ...summary, frames: f1 - f0, frames_repeated_because_nothing_moved: reused, render_seconds: seconds }, null, 1));
    console.error(`seed ${seed}: ${f1 - f0} frames x ${cams.length} cameras, ${ev.length} picks, ${seconds} s`);
  }
  if (s.errors.length) { console.error('console errors:\n' + s.errors.join('\n')); process.exitCode = 1; }
} finally { await s.close(); }
