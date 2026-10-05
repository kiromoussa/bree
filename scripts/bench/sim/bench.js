// Benchmark overlay for the PRIVATE simulator copy (bree-vision/out/sim-copy/js/bench.js). SIMULATED DATA.
// Loaded after scripts/train/sim/synth.js by scripts/bench/render_clip.mjs. Neither the simulator nor synth.js
// is edited. synth.js gives the randomised scene (lights, clothing range, slot fill, SKU swaps, camera mounting
// error); this file adds what the fixed benchmark needs on top:
//   - a scripted, seeded scenario per shopper: picks in a chosen zone (cooler, gondola, counter), a browse pause,
//     a put-back, a walk before concealing, two shoppers at the same shelf, pay or leave
//   - ground truth per frame and camera read from an id render: the visible box of every person and of every
//     item in a hand or on the counter (what is hidden is not in the box), plus the projected hand, head and feet
//   - the true pose of every camera (the layout pose plus the mounting error, roll included) as calibration
import * as THREE from 'three';
import { Shopper } from './agents.js';
import { mulberry32 } from './harness.js';

const Y = window.__breeSynth, sim = window.__sim, store = window.__store, renderer = window.__renderer;
const scene = sim.scene, canvas = renderer.domElement;
const PERSON = 0xF00000;                    // id of a person in the id render: PERSON + shopper number (0 = the clerk)
const idRGB = id => [(id & 255) / 255, ((id >> 8) & 255) / 255, ((id >> 16) & 255) / 255];
const idMat = new THREE.ShaderMaterial({
  vertexShader: 'attribute vec3 idColor; varying vec3 vId;\nvoid main() { vId = idColor; vec4 p = vec4(position, 1.0);\n#ifdef USE_INSTANCING\n p = instanceMatrix * p;\n#endif\n gl_Position = projectionMatrix * modelViewMatrix * p; }',
  fragmentShader: 'varying vec3 vId; void main() { gl_FragColor = vec4(vId, 1.0); }',
});
let idRT = null;
const B = { plans: [], arrive: [], k: 0, rng: Math.random, dr: Math.random, items: new Map(), minItem: Infinity, people: new Map(), log: [], shoppers: [], first: [] };

// items in a hand: synth.js gives each one an id colour when the simulator makes it; read it back
const make0 = store.makeItemMesh;
store.makeItemMesh = sku => {
  const m = make0(sku), c = m.geometry.attributes.idColor.array;
  const id = Math.round(c[0] * 255) | (Math.round(c[1] * 255) << 8) | (Math.round(c[2] * 255) << 16);
  B.items.set(id, { mesh: m, sku: sku.id }); B.minItem = Math.min(B.minItem, id); B.lastItem = B.items.get(id);
  return m;
};
function tagPerson(root, n) {
  const rgb = idRGB(PERSON + n);
  root.traverse(o => { if (o.isMesh) { const k = o.geometry.attributes.position.count, a = new Float32Array(k * 3); for (let i = 0; i < k; i++) a.set(rgb, i * 3); o.geometry.setAttribute('idColor', new THREE.BufferAttribute(a, 3)); } });
}

// ---------- scenario ----------
const HOT = new Set(['charger', 'pain', 'energy', 'jerky', 'candy', 'oil']);
const ZONES = [['gondola', 0.5], ['cooler', 0.3], ['checkout', 0.2]];
// One plan per shopper, from its own generator (not the DR stream, not the shoppers' stream).
function planScenario(seed, { shoppers, window }) {
  const R = mulberry32(seed * 9176 + 3), n = shoppers ?? 4 + Math.floor(R() * 5), zone = () => { let u = R(); for (const [z, p] of ZONES) if ((u -= p) < 0) return z; return 'gondola'; };
  let thief = Array.from({ length: n }, () => R() < 0.35);
  if (!thief.some(Boolean)) thief[Math.floor(R() * n)] = true;
  while (thief.filter(t => !t).length < 2) thief[thief.indexOf(true)] = false;       // at least two honest shoppers and one thief
  if (!thief.some(Boolean)) thief[n - 1] = true;
  const pair = 1 + Math.floor(R() * (n - 1));                                        // this shopper goes to the shelf of the one before
  const putter = (() => { const h = thief.map((t, i) => t ? -1 : i).filter(i => i >= 0); return h[Math.floor(R() * h.length)]; })();
  const plans = [], arrive = [];
  let t = 1 + R() * 2;
  for (let i = 0; i < n; i++) {
    const tasks = [], pick = (o = {}) => tasks.push({ type: 'pick', zone: zone(), browse: R() < 0.3 ? +(0.8 + R() * 1.7).toFixed(2) : 0, ...o });   // browse: looks at the shelf first
    if (thief[i]) {
      const steals = 1 + (R() < 0.3 ? 1 : 0), buys = R() < 0.5 ? 1 : 0, order = [...Array(steals).fill('steal'), ...Array(buys).fill('buy')];
      if (buys && R() < 0.5) order.reverse();
      for (const what of order) {
        pick(what === 'steal' ? { hot: true } : {});
        if (what === 'steal') { if (R() < 0.5) tasks.push({ type: 'go' }); tasks.push({ type: 'conceal' }); }
      }
      if (buys) tasks.push({ type: 'pay' });
    } else {
      const picks = R() < 0.4 ? 1 : R() < 0.67 ? 2 : 3;
      for (let k = 0; k < picks; k++) {
        pick();
        if (R() < 0.15 || (i === putter && k === 0)) { tasks.push({ type: 'wait', s: 1 + R() * 2, hold: true }, { type: 'putback' }); if (picks === 1 || R() < 0.5) pick(); }
      }
      tasks.push({ type: 'pay' });
    }
    const first = tasks.find(x => x.type === 'pick');
    if (i === pair - 1 && first.zone === 'checkout') first.zone = 'gondola';
    if (i === pair) { first.near = pair - 1; delete first.zone; t = arrive[i - 1] + 1.5 + R() * 1.5; }
    tasks.push({ type: 'exit' });
    plans.push({ thief: thief[i], tasks }); arrive.push(+t.toFixed(3));
    t += window / n * (0.5 + R());
  }
  return { plans, arrive, pair: [pair - 1, pair], rng: R };
}
function chooseSlot(sh, task) {
  const ok = s => s.present && !s.reserved && s.stand;
  let c = [];
  const ref = task.near !== undefined ? B.first[task.near] : null;
  if (ref) { // the same fixture as the other shopper, an arm's length or two along the shelf
    const d = s => s.stand.distanceTo(ref.stand), side = store.slots.filter(s => ok(s) && s.zoneId === ref.zoneId && d(s) > 0.45);   // same gondola side, or the cooler wall
    c = side.filter(s => d(s) < 1.2);
    if (!c.length && side.length) c = [side.reduce((x, y) => d(y) < d(x) ? y : x)];
  }
  if (!c.length) c = store.slots.filter(s => ok(s) && s.zone === (task.zone ?? 'gondola'));
  if (task.hot && c.some(s => HOT.has(s.sku.group))) c = c.filter(s => HOT.has(s.sku.group));
  return c.length ? c[Math.floor(B.rng() * c.length)] : null;
}
const hold = sh => sh.pos.clone().addScaledVector(sh.fwd(), 0.3).setY(1.08 * sh.fig.scale);
const next0 = Shopper.prototype.next, update0 = Shopper.prototype.update, sync0 = Shopper.prototype.sync;
Shopper.prototype.planTasks = function () { return B.plan.tasks.map(t => ({ ...t })); };
Shopper.prototype.next = function () {
  const task = this.tasks[0];
  if (!task || task.type === 'pay' || task.type === 'exit') return next0.call(this);
  this.tasks.shift();
  if (task.type === 'pick') {
    const slot = chooseSlot(this, task);
    if (!slot) return this.next();
    slot.reserved = true; if (!B.first[this.id - 1]) B.first[this.id - 1] = slot;
    this.goTo(slot.stand, () => { this.target = slot; this.face(slot.face); this.t = 0; if (task.browse) { this.state = 'wait'; this.waitS = task.browse; this.holdUp = false; this.afterWait = 'reach'; } else this.state = 'reach'; });
  } else if (task.type === 'wait') { this.state = 'wait'; this.t = 0; this.waitS = task.s; this.holdUp = !!task.hold; this.afterWait = null; }
  else if (task.type === 'go') { const c = store.slots.filter(s => s.stand && s.zone !== 'checkout'); this.goTo(c[Math.floor(B.rng() * c.length)].stand, () => { this.state = 'idle'; }); }
  else if (task.type === 'conceal') {
    const it = [...this.items].reverse().find(i => !i.concealed && !i.paid);
    if (!it) return this.next();
    this.concealItem = it; this.state = 'conceal'; this.t = 0;
  } else if (task.type === 'putback') {
    const it = [...this.items].reverse().find(i => !i.concealed && !i.paid);
    if (!it) return this.next();
    this.pb = it; this.target = it.slot; this.face(it.slot.face); this.state = 'putback'; this.t = 0;
  }
};
Shopper.prototype.update = function (dt) {
  if (this.state === 'wait') {
    this.t += dt; if (this.t >= this.waitS) { this.state = this.afterWait ?? 'idle'; this.afterWait = null; this.t = 0; }
    update0.call(this, dt);
    if (this.holdUp && this.state === 'wait') { this.hand = hold(this); this.sync(); }
    return;
  }
  if (this.state !== 'putback') return update0.call(this, dt);
  // the item goes back to its slot (0.9 s), then the empty hand comes back (0.8 s)
  this.t += dt;
  const slot = this.target, T1 = 0.9, T2 = 0.8;
  update0.call(this, dt);           // turns the body; no hand
  let hand;
  if (this.t < T1) {
    const k = this.t / T1, e = Math.sin(k * Math.PI / 2);
    if (slot.door !== undefined) store.setDoorOpen(slot.door, Math.min(1, k * 1.6));
    this.reachPose(slot.face, e); hand = hold(this).lerp(slot.face, e);
  } else {
    if (this.pb) {
      const it = this.pb; this.pb = null;
      sim.scene.remove(it.mesh); this.items.splice(this.items.indexOf(it), 1);
      store.setPresent(slot, true);
      it.event.outcome = 'put_back'; it.event.putBackT = +sim.time.toFixed(2);
    }
    const k = Math.min(1, (this.t - T1) / T2), e = 1 - Math.pow(1 - k, 2);
    if (slot.door !== undefined) store.setDoorOpen(slot.door, k < 0.4 ? 1 : Math.max(0, 1 - (k - 0.4) / 0.6));
    this.reachPose(slot.face, 1 - e); hand = slot.face.clone().lerp(this.restHand(), e);
    if (k >= 1) { this.state = 'idle'; if (slot.door !== undefined) store.setDoorOpen(slot.door, 0); }
  }
  this.hand = hand; this.sync();
};
Shopper.prototype.sync = function (dt) { // a second and third item sit beside the first, not inside it
  sync0.call(this, dt);
  let k = 0;
  for (const it of this.items) if (!it.concealed && !it.hidden && !it.paid && !it.onCounter) { if (k) it.mesh.position.addScaledVector(this.fwd(), 0.075 * k); k++; }
};

function setup(seed, layout, opts = {}) {
  const params = Y.setup(seed, layout, { shoppers: 0, jitter: opts.jitter ?? true, swap: opts.swap ?? true });
  B.items.clear(); B.minItem = Infinity; B.first = []; B.shoppers = []; B.k = 0; still.clear();
  const sc = planScenario(seed, { shoppers: opts.shoppers, window: opts.window ?? 30 });
  Object.assign(B, { plans: sc.plans, arrive: sc.arrive, rng: sc.rng, dr: mulberry32(seed * 6151 + 29), pair: sc.pair });
  const clerk = scene.children.find(o => o.isGroup && Math.hypot(o.position.x - 6.2, o.position.z - 3.3) < 1e-6 && o.children.length && !o.userData.vcam);
  if (!clerk) throw new Error('bench: clerk figure not found');
  tagPerson(clerk, 0);
  return { ...params, shoppers: sc.plans.length, window: opts.window ?? 30, thieves: sc.plans.filter(p => p.thief).length, same_shelf_pair: sc.pair.map(i => `P${String(i + 1).padStart(3, '0')}`) };
}
const SKIN = [0xc69c7b, 0x8d5a3b, 0xe0b899, 0x5c3a24, 0xb57f5a, 0xf1d0b5, 0x70452b, 0xd8a47f];
function spawn(plan) {
  B.plan = plan;
  const sh = new Shopper(sim, plan.thief), R = B.dr, old = sh.fig, c = (s0, s1, l0, l1) => new THREE.Color().setHSL(R(), s0 + (s1 - s0) * R(), l0 + (l1 - l0) * R()).getHex();
  // clothing: any hue (no dark-shirt tell for thieves), as in synth.js
  sh.group.remove(old.root);
  sh.fig = new sh.fig.constructor({ shirt: c(0, 0.9, 0.08, 0.9), pants: c(0, 0.6, 0.06, 0.6), skin: SKIN[Math.floor(R() * SKIN.length)], hair: c(0, 0.5, 0.03, 0.6), shoes: c(0, 0.5, 0.05, 0.8), longSleeve: R() < 0.5, scale: old.scale });
  sh.group.add(sh.fig.root); tagPerson(sh.fig.root, sh.id); sh.sync(0);
  sim.shoppers.push(sh);
  B.shoppers.push({ shopper: sh.name, thief: sh.thief, tEnter: +sim.time.toFixed(2), tExit: null, height: +sh.height.toFixed(3), tasks: plan.tasks.map(t => t.type) });
}
function advance(until, dt = 1 / 30) {
  while (sim.time < until - 1e-9) {
    while (B.k < B.arrive.length && B.arrive[B.k] <= sim.time) spawn(B.plans[B.k++]);
    sim.time += dt;
    for (const sh of sim.shoppers) sh.update(dt);
    for (const sh of sim.shoppers.filter(s => s.done)) {
      sh.items.forEach(i => { i.event.exitT = +sim.time.toFixed(2); if (!i.paid) i.event.outcome = 'theft'; });
      B.shoppers[sh.id - 1].tExit = +sim.time.toFixed(2);
      sh.dispose(); sim.shoppers.splice(sim.shoppers.indexOf(sh), 1);
    }
  }
  return { t: sim.time, inStore: sim.shoppers.length, pending: B.arrive.length - B.k };
}

// ---------- capture ----------
const camOf = name => { const vc = sim.vcams.find(v => v.name === name); if (!vc) throw new Error('no camera ' + name); return vc; };
const _v = new THREE.Vector3();
const px = (vc, P) => { _v.copy(P).project(vc.cam); return _v.z > 1 || _v.z < -1 ? null : [+((_v.x + 1) / 2 * vc.res[0]).toFixed(1), +((1 - _v.y) / 2 * vc.res[1]).toFixed(1)]; };
// Can anything that moves show up in this camera's image now? A shopper's body, held items, or the floor within 2 m of
// the shopper (a shadow). No occlusion test, so this says yes more often than needed. An open cooler door: yes for all.
const _q = new THREE.Vector3();
function busy(vc) {
  if (store.doorOpen.some(v => v > 0)) return true;
  const inView = (x, y, z) => { _q.set(x, y, z).project(vc.cam); return _q.z > -1 && _q.z < 1 && Math.abs(_q.x) < 1.1 && Math.abs(_q.y) < 1.1; };
  for (const sh of sim.shoppers) {
    const { x, z } = sh.pos, c = vc.cam.position;
    if (Math.hypot(x - c.x, z - c.z) < 1.5) return true;
    for (const y of [0, 0.5, 1, 1.5, sh.height]) if (inView(x, y, z)) return true;
    for (const [dx, dz] of [[2, 0], [-2, 0], [0, 2], [0, -2], [1.4, 1.4], [1.4, -1.4], [-1.4, 1.4], [-1.4, -1.4], [1, 0], [-1, 0], [0, 1], [0, -1]]) if (inView(x + dx, 0, z + dz)) return true;
    const h = sh.fig.handWorld(0); if (inView(h.x, h.y, h.z)) return true;
  }
  return false;
}
const still = new Map();   // camera -> what the scene looked like when its last empty frame was rendered
// A camera with nothing moving in view, in a scene that has not changed since its last such frame, would render the same
// pixels again: capture() then returns { same: true } and the caller repeats that frame. The truth rows are repeated too.
function capture(name, quality = 0.92) {
  const vc = camOf(name), [w, h] = vc.res;
  vc.cam.updateMatrixWorld(true);
  const idle = !busy(vc), sceneKey = sim.events.length + ':' + sim.events.filter(e => e.outcome === 'put_back').length;
  if (idle && still.get(name) === sceneKey) return { same: true };
  renderer.setPixelRatio(1); renderer.setSize(w, h, false); renderer.setRenderTarget(null); renderer.setScissorTest(false); renderer.setViewport(0, 0, w, h);
  renderer.setClearColor(0x0d1014, 1); renderer.clear(); renderer.shadowMap.needsUpdate = true; renderer.render(scene, vc.cam);
  const image = canvas.toDataURL('image/jpeg', quality);
  if (!idRT || idRT.width !== w || idRT.height !== h) { idRT?.dispose(); idRT = new THREE.WebGLRenderTarget(w, h, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }); }
  const hidden = []; // glass does not hide what is behind it
  scene.traverse(o => { if (o.isMesh && o.visible && [].concat(o.material).some(m => m.transparent)) { o.visible = false; hidden.push(o); } });
  scene.overrideMaterial = idMat; renderer.setRenderTarget(idRT); renderer.setClearColor(0x000000, 1); renderer.clear(); renderer.render(scene, vc.cam);
  scene.overrideMaterial = null; hidden.forEach(o => { o.visible = true; });
  const buf = new Uint8Array(w * h * 4); renderer.readRenderTargetPixels(idRT, 0, 0, w, h, buf); renderer.setRenderTarget(null);
  const box = new Map(), lo = B.minItem;
  for (let r = 0, i = 0; r < h; r++) for (let x = 0; x < w; x++, i += 4) {
    const id = buf[i] | (buf[i + 1] << 8) | (buf[i + 2] << 16);
    if (id < lo) continue;
    let b = box.get(id); if (!b) box.set(id, b = [x, r, x, r, 0]);       // r counts from the bottom row
    if (x < b[0]) b[0] = x; if (x > b[2]) b[2] = x; if (r < b[1]) b[1] = r; if (r > b[3]) b[3] = r; b[4]++;
  }
  const bb = b => [b[0], h - 1 - b[3], b[2] + 1, h - b[1]];
  const held = new Map(sim.shoppers.flatMap(sh => sh.items.map(it => [it.mesh, { sh, it }])));
  const items = [], persons = [];
  for (const [id, b] of box) {
    if (id >= PERSON) {
      const n = id - PERSON, sh = sim.shoppers.find(s => s.id === n);
      if (n && !sh) continue;
      const p = { shopper: n ? sh.name : 'clerk', bbox: bb(b), vis_px: b[4] };
      if (sh) Object.assign(p, { hand: px(vc, sh.fig.handWorld(0)), head: px(vc, _v.copy(sh.pos).setY(sh.height - sh.crouch - 0.1).clone()), feet: px(vc, sh.pos.clone()), state: sh.state });
      persons.push(p);
    } else {
      const r = B.items.get(id), st = r && held.get(r.mesh);
      if (r) items.push({ sku: r.sku, kind: st?.it.onCounter ? 'counter' : 'hand', bbox: bb(b), vis_px: b[4], shopper: st?.sh.name ?? null, slot: st?.it.slot.label ?? null });
    }
  }
  if (idle) still.set(name, sceneKey); else still.delete(name);
  return { image, persons, items };
}
// where every shopper is on the floor now (store metres, layout x and z), and where the reaching hand is
const world = () => sim.shoppers.map(sh => ({ shopper: sh.name, x: +sh.pos.x.toFixed(3), z: +sh.pos.z.toFixed(3), heading: +sh.heading.toFixed(3), state: sh.state,
  hand: sh.fig.handWorld(0).toArray().map(v => +v.toFixed(3)), carrying: sh.items.filter(i => !i.concealed && !i.hidden && !i.paid).length }));

// Ground truth of the scenario, one row per picked item. outcome: concealed (theft), paid, put_back, theft (left with it in a
// hand, not concealed), in_hand (the clip ended first). The first seven keys are the shape `make sim-eval` reads.
const fixtureOf = new Map(store.slots.map(s => [s.label, s]));
function events(fps) {
  return sim.events.map(e => {
    const outcome = e.concealed ? 'concealed' : e.outcome === 'paid' ? 'paid' : e.outcome === 'put_back' ? 'put_back' : e.outcome === 'theft' ? 'theft' : 'in_hand', s = fixtureOf.get(e.slot);
    return { t: e.t, frame: Math.round(e.t * fps), shopper: e.shopper, thief: e.thief, skuId: e.sku, slotId: e.slot, outcome,
      tResolved: e.concealed ? e.concealT : e.payT ?? e.putBackT ?? null, tExit: e.exitT ?? null, cameras: e.views.map(v => ({ id: v.cam, kind: v.kind, px: v.px })),
      fixtureId: s.fixtureId, zone: e.zone, zoneId: e.zoneId, tConceal: e.concealT ?? null, tPay: e.payT ?? null, tPutBack: e.putBackT ?? null,
      slotFace: s.face.toArray().map(v => +v.toFixed(4)), concealSeenBy: e.concealSeenBy ?? null };
  });
}
// True pose of each camera as a calibration would measure it. R rows are the camera's right, down and forward axes in the
// store frame (three.js metres, y up, +z toward the front door): pixel = K * R * (X - position).
function calibration() {
  const m = new THREE.Matrix4(), r = new THREE.Vector3(), u = new THREE.Vector3(), b = new THREE.Vector3(), f = v => v.toArray().map(x => +x.toFixed(8));
  return sim.vcams.map(vc => {
    vc.cam.updateMatrixWorld(true); m.copy(vc.cam.matrixWorld).extractBasis(r, u, b);
    const [w, h] = vc.res;
    return { id: vc.name, kind: Y.layout().cameras.find(c => c.id === vc.name).kind, resolution: [w, h], fx: +vc.fx.toFixed(4), fy: +vc.fy.toFixed(4), cx: w / 2, cy: h / 2,
      position: f(vc.cam.position), R: [f(r), f(u.clone().negate()), f(b.clone().negate())], roll: +vc.cam.rotation.z.toFixed(6) };
  });
}
function layout() { // the store, the planogram (which slot holds which SKU in this scene) and the cameras with their true pose
  const L = Y.layout(), roll = new Map(sim.vcams.map(vc => [vc.name, +vc.cam.rotation.z.toFixed(6)]));
  for (const c of L.cameras) c.roll = roll.get(c.id);
  const face = new Map(store.slots.map(s => [s.label, s]));
  for (const s of L.slots) { const x = face.get(s.id); Object.assign(s, { zone: x.zone, zoneId: x.zoneId, face: x.face.toArray().map(v => +v.toFixed(4)), normal: x.normal.toArray().map(v => +v.toFixed(4)) }); }
  L.poi = Object.fromEntries(Object.entries(store.poi).map(([k, v]) => [k, v.toArray().map(x => +x.toFixed(3))]));
  return L;
}

window.__breeBench = { setup, advance, capture, world, events, calibration, layout, shoppers: () => B.shoppers, plan: () => ({ arrive: B.arrive, plans: B.plans }),
  state: () => ({ t: sim.time, inStore: sim.shoppers.length, pending: B.arrive.length - B.k }) };
