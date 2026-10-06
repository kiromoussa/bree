// Benchmark overlay for the PRIVATE simulator copy (bree-vision/out/sim-copy/js/bench.js). SIMULATED DATA.
// Loaded after scripts/train/sim/synth.js by scripts/bench/render_clip.mjs. Neither the simulator nor synth.js
// is edited. synth.js gives the randomised scene (lights, clothing range, slot fill, SKU swaps, camera mounting
// error); this file adds what the fixed benchmark needs on top:
//   - a scripted, seeded scenario per shopper: picks in a chosen zone (cooler, gondola, counter), a browse pause,
//     a put-back, a walk before concealing, two shoppers at the same shelf, pay or leave
//   - ground truth per frame and camera read from an id render: the visible box of every person and of every
//     item in a hand or on the counter (what is hidden is not in the box), plus the projected hand, head and feet
//   - the true pose of every camera (the layout pose plus the mounting error, roll included) as calibration
// Generator 2 (setup(seed, layout, { gen: 2 }), used by the DEV2 and CHECKPOINT sets; generator 1 is unchanged, so the
// DEV, TEST and TRAIN-seed clips can be rendered again as they were). Each addition is a seeded option (FEATURES):
//   - group: two shoppers enter together, one carries the items to the counter and the other pays
//   - staff: a vendor or employee restocks (puts items into empty places, pulls items, moves an item to another slot)
//   - wrongSlot: a put-back goes into an empty place of another slot    - shift: an item is nudged on the shelf, not taken
//   - bump: two cameras are knocked a few degrees partway through       - block: an item camera is covered for a while
//   - dropScan: one paid item is missing from the register feed (done in render_clip.mjs)    - night: night lighting
//   - the simulator's camera pass (lens distortion, exposure, motion blur, noise, codec blocks, white balance, glare)
//   - cameras by aisle: every item camera that covers an aisle the clip uses, chosen from the layout, not from the picks
//   - a nominal planogram: layout.json names the product a slot is meant to hold; what is really there is truth
import * as THREE from 'three';
import { Shopper } from './agents.js';
import { effPixelsAt, effViews } from './cameras.js';
import { mulberry32 } from './harness.js';

const Y = window.__breeSynth, sim = window.__sim, store = window.__store, renderer = window.__renderer;
const scene = sim.scene, canvas = renderer.domElement, real = window.__breeCam;
const PERSON = 0xF00000;                    // id of a person in the id render: PERSON + shopper number (0 = the clerk)
const idRGB = id => [(id & 255) / 255, ((id >> 8) & 255) / 255, ((id >> 16) & 255) / 255];
const idMat = new THREE.ShaderMaterial({
  vertexShader: 'attribute vec3 idColor; varying vec3 vId;\nvoid main() { vId = idColor; vec4 p = vec4(position, 1.0);\n#ifdef USE_INSTANCING\n p = instanceMatrix * p;\n#endif\n gl_Position = projectionMatrix * modelViewMatrix * p; }',
  fragmentShader: 'varying vec3 vId; void main() { gl_FragColor = vec4(vId, 1.0); }',
});
let idRT = null;
const B = { plans: [], arrive: [], k: 0, rng: Math.random, dr: Math.random, items: new Map(), minItem: Infinity, people: new Map(), log: [], shoppers: [], first: [], acts: [], rev: 0, gen: 1, zones: null, faults: null, cams: [] };

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
  const ok = s => s.present && !s.reserved && s.stand && !s.occupied && (!B.zones || B.zones.has(s.zoneId));
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
  const open = () => [...this.items].reverse().find(i => !i.concealed && !i.paid), R = store.poi.register;
  const empty = s => !s.present && !s.reserved && !s.occupied && s.stand;
  if (task.type === 'pick' || task.type === 'touch') {
    const slot = chooseSlot(this, task), go = task.type === 'touch' ? 'touch' : 'reach';
    if (!slot) return this.next();
    slot.reserved = true; if (!B.first[this.id - 1]) B.first[this.id - 1] = slot;
    this.goTo(slot.stand, () => { this.target = slot; this.face(slot.face); this.t = 0; if (task.browse) { this.state = 'wait'; this.waitS = task.browse; this.holdUp = false; this.afterWait = go; } else this.state = go; });
  } else if (task.type === 'wait') { this.state = 'wait'; this.t = 0; this.waitS = task.s; this.holdUp = !!task.hold; this.afterWait = null; }
  else if (task.type === 'go') { const c = store.slots.filter(s => s.stand && s.zone !== 'checkout'); this.goTo(c[Math.floor(B.rng() * c.length)].stand, () => { this.state = 'idle'; }); }
  else if (task.type === 'conceal') {
    const it = open();
    if (!it) return this.next();
    this.concealItem = it; this.state = 'conceal'; this.t = 0;
  } else if (task.type === 'putback') {
    const it = open();
    if (!it) return this.next();
    let slot = it.slot;
    if (task.wrong) { // an empty place on the same shelf, up to about an arm's length from where the item came from
      const d = s => s.face.distanceTo(it.slot.face), c = store.slots.filter(s => empty(s) && s !== it.slot && s.zoneId === it.slot.zoneId && d(s) > 0.2 && d(s) < 1.2);
      if (c.length) slot = c[Math.floor(B.rng() * c.length)];
    }
    if (slot !== it.slot) { slot.reserved = true; this.goTo(slot.stand, () => startPut(this, it, slot)); } else startPut(this, it, slot);
  } else if (task.type === 'stock') { // staff: a new item from the tote, or the item just pulled, goes into an empty place
    let it = task.held ? open() : null;
    const c = store.slots.filter(s => empty(s) && s.zone !== 'checkout' && (!B.zones || B.zones.has(s.zoneId)) && (!it || s !== it.slot));
    if (!c.length || (task.held && !it)) return this.next();
    const slot = c[Math.floor(B.rng() * c.length)]; slot.reserved = true;
    if (!it) { it = { slot, sku: slot.sku, mesh: store.makeItemMesh(slot.sku), concealed: false, event: {} }; sim.scene.add(it.mesh); this.items.push(it); }
    this.goTo(slot.stand, () => startPut(this, it, slot));
  } else if (task.type === 'handoff') { // group: the one who carries stands beside the one who pays
    this.partner = task.to; this.goTo(R.clone().add(new THREE.Vector3(0, 0, 0.6)), () => { this.face(R.clone().add(new THREE.Vector3(1, 0, 0.6))); this.state = 'drop'; this.t = 0; this.dropT = null; });
  } else if (task.type === 'grouppay') {
    this.partner = task.from; this.goTo(R, () => { this.face(R.clone().add(new THREE.Vector3(1, 0, 0))); this.state = 'gwait'; this.t = 0; });
  }
};
function startPut(sh, it, slot) { sh.pb = it; sh.target = slot; sh.face(slot.face); sh.state = 'putback'; sh.t = 0; }
// What happened at a slot that is not a shopper's pick: a touch, a staff put. Same camera list as a pick event.
function act(kind, sh, slot, sku, extra = {}) {
  return { kind, t: +sim.time.toFixed(2), shopper: sh.name, skuId: sku.id, slotId: slot.label, fixtureId: slot.fixtureId, zone: slot.zone, zoneId: slot.zoneId,
    cameras: effViews(sim.vcams, slot.face, slot.normal, store.occluders, { bodies: sim.bodies() }).map(v => ({ id: v.cam, kind: v.kind, px: v.px })), ...extra };
}
// The item stands in the empty front place of another slot, as its own mesh (any shape fits any slot this way).
function place(it, slot) {
  const p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3(); slot.matrix.decompose(p, q, sc);
  it.mesh.position.set(p.x, p.y - sc.y / 2 + it.sku.size[1] / 2, p.z); it.mesh.quaternion.copy(q); it.mesh.visible = true;
  slot.occupied = it; const r = [...B.items.values()].find(r => r.mesh === it.mesh); if (r) r.misplaced = slot.label;
}
// The front item slides a few centimetres along the shelf and turns. Nothing leaves the shelf.
function nudge(slot) {
  const p = new THREE.Vector3(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), m = slot.matrix.clone(); m.decompose(p, q, sc);
  const sgn = B.rng() < 0.5 ? -1 : 1, d = 0.02 + 0.02 * B.rng();
  p.x -= slot.normal.z * sgn * d; p.z += slot.normal.x * sgn * d; p.addScaledVector(slot.normal, -0.008);
  q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), sgn * (0.35 + 0.4 * B.rng())));
  slot.matrix = m.compose(p, q, sc); store.setPresent(slot, true);
}
function touchUpdate(sh, dt) { // reach in, shift the item, come back with an empty hand
  sh.t += dt; update0.call(sh, dt);
  const slot = sh.target, T1 = 0.9, T2 = 0.8; let hand;
  if (sh.t < T1) {
    const k = sh.t / T1, e = Math.sin(k * Math.PI / 2);
    if (slot.door !== undefined) store.setDoorOpen(slot.door, Math.min(1, k * 1.6));
    sh.reachPose(slot.face, e); hand = sh.restHand().lerp(slot.face, e);
  } else {
    if (!sh.touched) { sh.touched = true; nudge(slot); slot.reserved = false; B.acts.push(act('touch', sh, slot, slot.sku)); B.rev++; }
    const k = Math.min(1, (sh.t - T1) / T2), e = 1 - Math.pow(1 - k, 2);
    if (slot.door !== undefined) store.setDoorOpen(slot.door, k < 0.4 ? 1 : Math.max(0, 1 - (k - 0.4) / 0.6));
    sh.reachPose(slot.face, 1 - e); hand = slot.face.clone().lerp(sh.restHand(), e);
    if (k >= 1) { sh.state = 'idle'; sh.touched = false; if (slot.door !== undefined) store.setDoorOpen(slot.door, 0); }
  }
  sh.hand = hand; sh.sync();
}
function groupUpdate(sh, dt) {
  sh.t += dt; update0.call(sh, dt);
  const mate = sim.shoppers.find(x => x.id === sh.partner), R = store.poi.register; let hand = null;
  if (sh.state === 'drop') { // waits for the one who pays, puts what it carries on the counter and lets go of it
    const open = sh.items.filter(i => !i.concealed && !i.paid);
    if (sh.dropT === null) { hand = open.length ? hold(sh) : null; if (!mate || mate.state === 'gwait' || !open.length || sh.t > 40) sh.dropT = sh.t; }
    else {
      const k = sh.t - sh.dropT;
      if (k < 1.0 && (open.length || sh.dropped)) hand = hold(sh).lerp(R.clone().add(new THREE.Vector3(0.5, 1.0, 0.3)), Math.min(1, k / 0.7));
      if (k >= 0.7 && !sh.dropped) {
        sh.dropped = true;
        if (mate) open.forEach((it, n) => { sh.items.splice(sh.items.indexOf(it), 1); mate.items.push(it); it.onCounter = true; it.event.handT = +sim.time.toFixed(2);
          it.mesh.position.set(R.x + 0.55, 0.95 + it.sku.size[1] / 2, 3.0 + n * 0.14); it.mesh.rotation.set(0, -Math.PI / 2, 0); });
      }
      if (k >= 1.2) sh.state = 'idle';
    }
  } else if (!mate || mate.dropped || sh.t > 60) { sh.state = sh.items.some(i => !i.concealed && !i.paid) ? 'pay' : 'idle'; sh.t = 0; }   // gwait
  sh.hand = hand; sh.sync();
}
Shopper.prototype.update = function (dt) {
  if (this.state === 'wait') {
    this.t += dt; if (this.t >= this.waitS) { this.state = this.afterWait ?? 'idle'; this.afterWait = null; this.t = 0; }
    update0.call(this, dt);
    if (this.holdUp && this.state === 'wait') { this.hand = hold(this); this.sync(); }
    return;
  }
  if (this.state === 'touch') return touchUpdate(this, dt);
  if (this.state === 'drop' || this.state === 'gwait') return groupUpdate(this, dt);
  if (this.state !== 'putback') return update0.call(this, dt);
  // the item goes back to its slot, or into the slot the put was sent to (0.9 s), then the empty hand comes back (0.8 s)
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
      this.items.splice(this.items.indexOf(it), 1); slot.reserved = false;
      if (slot === it.slot) { sim.scene.remove(it.mesh); store.setPresent(slot, true); } else place(it, slot);
      it.event.outcome = 'put_back'; it.event.putBackT = +sim.time.toFixed(2); if (slot !== it.slot) it.event.putBackSlot = slot.label;
      if (this.role === 'staff') B.acts.push(act('staff_put', this, slot, it.sku, { from: it.event.id ? it.slot.label : null }));
      B.rev++;
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
  for (const s of store.slots) if (s.occupied) { scene.remove(s.occupied.mesh); s.occupied = null; }
  B.items.clear(); B.minItem = Infinity; B.first = []; B.shoppers = []; B.k = 0; still.clear();
  Object.assign(B, { acts: [], rev: 0, gen: opts.gen ?? 1, zones: null, faults: null, cams: [] });
  if (B.gen === 2) return setup2(seed, layout, opts, params);
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
  if (plan.role === 'staff') { // a vendor in a work vest or an employee in the clerk's shirt
    sh.name = `STAFF${B.shoppers.filter(x => x.role === 'staff').length + 1}`; sh.role = 'staff';
    const f = sh.fig; f.root.traverse(o => o.geometry?.dispose());
    sh.fig = new f.constructor(plan.vest ? { shirt: 0xf2e21d, pants: 0x1d1f23, skin: SKIN[(sh.id * 3) % SKIN.length], hair: 0x1a1410, scale: old.scale, look: { top: 'vest', inner: 0x2b5f8a } }
      : { shirt: 0x2b5f8a, pants: 0x1d1f23, skin: SKIN[(sh.id * 3) % SKIN.length], hair: 0x1a1410, longSleeve: false, scale: old.scale });
  }
  if (plan.group) sh.gid = plan.group;
  sh.group.add(sh.fig.root); tagPerson(sh.fig.root, sh.id); sh.sync(0);
  sim.shoppers.push(sh);
  B.shoppers.push({ shopper: sh.name, thief: sh.thief, role: plan.role, group: plan.group, tEnter: +sim.time.toFixed(2), tExit: null, height: +sh.height.toFixed(3), tasks: plan.tasks.map(t => t.type) });
}
function stepBy(dt) {
  while (B.k < B.arrive.length && B.arrive[B.k] <= sim.time) spawn(B.plans[B.k++]);
  sim.time += dt;
  for (const sh of sim.shoppers) sh.update(dt);
  for (const sh of sim.shoppers.filter(s => s.done)) {
    sh.items.forEach(i => { i.event.exitT = +sim.time.toFixed(2); if (!i.paid) i.event.outcome = sh.role === 'staff' ? 'staff_removed' : 'theft'; });
    B.shoppers[sh.id - 1].tExit = +sim.time.toFixed(2);
    sh.dispose(); sim.shoppers.splice(sim.shoppers.indexOf(sh), 1);
  }
}
function advance(until, dt = 1 / 30) {
  while (sim.time < until - 1e-9) stepBy(dt);
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
  const m = B.gen === 2 ? 1.1 * real.camParams(vc).overscan : 1.1;
  const inView = (x, y, z) => { _q.set(x, y, z).project(vc.cam); return _q.z > -1 && _q.z < 1 && Math.abs(_q.x) < m && Math.abs(_q.y) < m; };
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
const world = () => sim.shoppers.map(sh => ({ shopper: sh.name, ...(sh.role && { role: sh.role }), x: +sh.pos.x.toFixed(3), z: +sh.pos.z.toFixed(3), heading: +sh.heading.toFixed(3), state: sh.state,
  hand: sh.fig.handWorld(0).toArray().map(v => +v.toFixed(3)), carrying: sh.items.filter(i => !i.concealed && !i.hidden && !i.paid).length }));

// Ground truth of the scenario, one row per picked item. outcome: concealed (theft), paid, put_back, theft (left with it in a
// hand, not concealed), in_hand (the clip ended first). The first seven keys are the shape `make sim-eval` reads.
const fixtureOf = new Map(store.slots.map(s => [s.label, s]));
function events(fps) {
  return sim.events.filter(e => !e.staff).map(e => {
    const outcome = e.concealed ? 'concealed' : e.outcome === 'paid' ? 'paid' : e.outcome === 'put_back' ? 'put_back' : e.outcome === 'theft' ? 'theft' : 'in_hand', s = fixtureOf.get(e.slot);
    return { t: e.t, frame: Math.round(e.t * fps), shopper: e.shopper, thief: e.thief, skuId: e.sku, slotId: e.slot, outcome,
      tResolved: e.concealed ? e.concealT : e.payT ?? e.putBackT ?? null, tExit: e.exitT ?? null, cameras: e.views.map(v => ({ id: v.cam, kind: v.kind, px: v.px })),
      fixtureId: s.fixtureId, zone: e.zone, zoneId: e.zoneId, tConceal: e.concealT ?? null, tPay: e.payT ?? null, tPutBack: e.putBackT ?? null,
      slotFace: s.face.toArray().map(v => +v.toFixed(4)), concealSeenBy: e.concealSeenBy ?? null,
      ...(B.gen === 2 && { group: e.group ?? null, paidBy: e.paidBy ?? null, tHand: e.handT ?? null, putBackSlot: e.putBackSlot ?? null }) };
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
      position: f(vc.cam.position), R: [f(r), f(u.clone().negate()), f(b.clone().negate())], roll: +vc.cam.rotation.z.toFixed(6),
      ...(B.gen === 2 && { k_div: +real.camParams(vc).a.toFixed(6), dist: real.camParams(vc).opencv_dist_coeffs }) };
  });
}
function layoutOut() { // the store, the planogram (which slot holds which SKU in this scene) and the cameras with their true pose
  const L = Y.layout(), roll = new Map(sim.vcams.map(vc => [vc.name, +vc.cam.rotation.z.toFixed(6)]));
  for (const c of L.cameras) c.roll = roll.get(c.id);
  const face = new Map(store.slots.map(s => [s.label, s]));
  for (const s of L.slots) { const x = face.get(s.id); Object.assign(s, { zone: x.zone, zoneId: x.zoneId, face: x.face.toArray().map(v => +v.toFixed(4)), normal: x.normal.toArray().map(v => +v.toFixed(4)) }); }
  L.poi = Object.fromEntries(Object.entries(store.poi).map(([k, v]) => [k, v.toArray().map(x => +x.toFixed(3))]));
  return L;
}

// ---------- generator 2: the harder, fairer clips (DEV2, CHECKPOINT) ----------
const pick0 = sim.onPick, pay1 = sim.onPay;
sim.onPick = (sh, slot) => { pick0(sh, slot); const e = sim.events[sim.events.length - 1]; if (sh.role === 'staff') e.staff = true; if (sh.gid) e.group = sh.gid; };
sim.onPay = (sh, items) => { items.forEach(i => { if (i.event.shopper && i.event.shopper !== sh.name) i.event.paidBy = sh.name; }); pay1(sh, items); };

// option, chance that a clip has it. Every clip draws all of them, in this order, from its own generator.
const FEATURES = [['group', 0.5], ['staff', 0.45], ['wrongSlot', 0.45], ['shift', 0.5], ['bump', 0.3], ['block', 0.3], ['dropScan', 0.35], ['night', 0.25]];
const MIN_SLOTS = 8;      // an item camera belongs to the aisle it sees most slots of (at the simulator's pixel threshold), if it sees this many
const shuffled = (a, R) => { a = [...a]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
// zoneId -> aisle. Two gondola sides that face the same walkway are one aisle; the cooler wall is one; counter and back bar are one.
function aisleOf() {
  const gx = {}; for (const s of store.slots) if (s.zone === 'gondola' && s.stand) (gx[s.zoneId] ??= []).push(s.stand.x);
  const mean = Object.entries(gx).map(([z, v]) => [z, v.reduce((a, b) => a + b, 0) / v.length]).sort((a, b) => a[1] - b[1]);
  const out = {}; let k = 0, last = -1e9;
  for (const [z, x] of mean) { if (x - last > 0.8) k++; last = x; out[z] = 'aisle' + k; }
  for (const s of store.slots) if (!(s.zoneId in out)) out[s.zoneId] = s.zoneId === 'cooler' ? 'cooler' : 'counter';
  return out;
}
// per camera: how many slots of each zone it sees well enough. Layout geometry only: no shopper, no pick.
function coverage() {
  const out = {};
  for (const vc of sim.vcams) { vc.cam.updateMatrixWorld(true); const c = {}; for (const s of store.slots) if (effPixelsAt(vc, s.face, s.normal, store.occluders) >= sim.pxMin) c[s.zoneId] = (c[s.zoneId] ?? 0) + 1; out[vc.name] = c; }
  return out;
}
// The product each slot is meant to hold: the most common one of its planogram block (consecutive slots of one base product on a
// fixture, which the scene restocks with one variant). The single misplaced items of the scene are not in it.
function planogram() {
  const nominal = {}, exact = {}, S = store.slots;
  for (let i = 0; i < S.length;) {
    let j = i; const n = new Map();
    while (j < S.length && S[j]._sku0 === S[i]._sku0 && S[j].fixtureId === S[i].fixtureId) { n.set(S[j].sku.id, (n.get(S[j].sku.id) ?? 0) + 1); j++; }
    const top = [...n].reduce((a, b) => b[1] > a[1] ? b : a)[0];
    for (; i < j; i++) { nominal[S[i].label] = top; exact[S[i].label] = S[i].sku.id; }
  }
  return { nominal, exact };
}

function planScenario2(seed, { window, feat, kinds }) {
  const R = mulberry32(seed * 9176 + 3), n = 4 + Math.floor(R() * 4), Z = ZONES.filter(([z]) => kinds.has(z)), tot = Z.reduce((a, [, p]) => a + p, 0);
  const zone = () => { let u = R() * tot; for (const [z, p] of Z) if ((u -= p) < 0) return z; return Z[0][0]; };
  const pick = (tasks, o = {}) => tasks.push({ type: 'pick', zone: zone(), browse: R() < 0.3 ? +(0.8 + R() * 1.7).toFixed(2) : 0, ...o });
  const steal = tasks => { pick(tasks, { hot: true }); if (R() < 0.5) tasks.push({ type: 'go' }); tasks.push({ type: 'conceal' }); };
  let thief = Array.from({ length: n }, () => R() < 0.35);
  if (!thief.some(Boolean)) thief[Math.floor(R() * n)] = true;
  while (thief.filter(t => !t).length < 2) thief[thief.indexOf(true)] = false;
  if (!thief.some(Boolean)) thief[n - 1] = true;
  const honest = thief.map((t, i) => t ? -1 : i).filter(i => i >= 0), one = a => a[Math.floor(R() * a.length)];
  const pair = 1 + Math.floor(R() * (n - 1)), putter = one(honest);
  const P = [];
  let t = 1 + R() * 2;
  for (let i = 0; i < n; i++) {
    const tasks = [];
    if (thief[i]) {
      const steals = 1 + (R() < 0.3 ? 1 : 0), buys = R() < 0.5 ? 1 : 0, order = [...Array(steals).fill('steal'), ...Array(buys).fill('buy')];
      if (buys && R() < 0.5) order.reverse();
      for (const what of order) if (what === 'steal') steal(tasks); else pick(tasks);
      if (buys) tasks.push({ type: 'pay' });
    } else {
      const picks = R() < 0.4 ? 1 : R() < 0.67 ? 2 : 3;
      for (let k = 0; k < picks; k++) {
        pick(tasks);
        if (R() < 0.15 || (i === putter && k === 0)) { tasks.push({ type: 'wait', s: 1 + R() * 2, hold: true }, { type: 'putback', wrong: feat.wrongSlot && (i === putter || R() < 0.5) }); if (picks === 1 || R() < 0.5) pick(tasks); }
      }
      tasks.push({ type: 'pay' });
    }
    const first = tasks.find(x => x.type === 'pick');
    if (i === pair - 1 && first.zone === 'checkout') first.zone = 'gondola';
    if (i === pair) { first.near = pair - 1; delete first.zone; t = P[i - 1].arrive + 1.5 + R() * 1.5; }
    tasks.push({ type: 'exit' });
    P.push({ key: i, thief: thief[i], role: 'shopper', group: null, tasks, arrive: +t.toFixed(3) });
    t += window / n * (0.5 + R());
  }
  const tags = { group: null, staff: null, touches: 0 };
  if (feat.shift) for (const i of shuffled(P.map(p => p.key), R).slice(0, 1 + (R() < 0.5 ? 1 : 0))) { // reaches in, shifts an item, takes nothing
    const tasks = P[i].tasks, at = R() < 0.5 ? 0 : tasks.findIndex(x => x.type === 'pay' || x.type === 'exit' || x.type === 'handoff');
    tasks.splice(at, 0, { type: 'touch', zone: zone() }); tags.touches++;
  }
  if (feat.group) { // two enter together; the first carries the items to the counter, the second pays for them
    const lead = P[one(honest)], mate = { key: P.length, thief: false, role: 'shopper', group: 'G1', tasks: [], arrive: +(lead.arrive + 0.6).toFixed(3) };
    lead.group = 'G1';
    if (R() < 0.7) pick(mate.tasks, { near: lead.key, zone: undefined });
    mate.tasks.push({ type: 'grouppay', from: lead.key }, { type: 'exit' });
    const k = lead.tasks.findIndex(x => x.type === 'pay'), tail = [{ type: 'handoff', to: mate.key }];
    if (R() < 0.3) { const hide = []; steal(hide); lead.tasks.splice(k, 0, ...hide); lead.thief = true; }      // and pockets one more thing on the way
    if (R() < 0.5) tail.push({ type: 'wait', s: 4.5 });                                                          // waits for the other, or walks out first
    lead.tasks.splice(lead.tasks.findIndex(x => x.type === 'pay'), 1, ...tail);
    P.push(mate); tags.group = { carries: lead.key, pays: mate.key };
  }
  if (feat.staff) { // restocks: puts new items into empty places, pulls an item and moves it or carries it off
    const tasks = [], acts = 2 + Math.floor(R() * 3);
    for (let k = 0; k < acts; k++) { if (R() < 0.6) tasks.push({ type: 'stock' }); else { pick(tasks, { browse: 0 }); if (R() < 0.7) tasks.push({ type: 'stock', held: true }); } }
    tasks.push({ type: 'exit' });
    P.push({ key: P.length, thief: false, role: 'staff', vest: R() < 0.5, group: null, tasks, arrive: +(3 + R() * 0.6 * window).toFixed(3) }); tags.staff = P.length - 1;
  }
  // shoppers come in by arrival time; tasks that point at another shopper follow the new numbering
  const order = [...P].sort((a, b) => a.arrive - b.arrive), at = new Map(order.map((p, i) => [p.key, i]));
  for (const p of order) for (const x of p.tasks) for (const k of ['near', 'to', 'from']) if (x[k] !== undefined) x[k] = at.get(x[k]) + (k === 'near' ? 0 : 1);   // near: index into B.first; to, from: shopper id
  return { plans: order, arrive: order.map(p => p.arrive), pair: [at.get(pair - 1), at.get(pair)], rng: R, tags };
}

function setup2(seed, layout, opts, params) {
  if (!real) throw new Error('bench: this simulator copy has no camera pass (window.__breeCam): refresh out/sim-copy from the simulator');
  const F = mulberry32(seed * 7151 + 101), want = opts.features ?? {}, feat = {};
  for (const [k, p] of FEATURES) { const u = F(); feat[k] = k in want ? !!want[k] : u < p; }
  // which aisles this store has cameras in, and so where the clip's shoppers pick: the counter always, the cooler mostly, two or three gondola aisles
  const AZ = aisleOf(), gondola = shuffled([...new Set(Object.values(AZ))].filter(a => a.startsWith('aisle')).sort(), F);
  const cooler = F() < 0.6, ng = F() < 0.7 ? 2 : 3, used = new Set(opts.aisles === 'all' ? Object.values(AZ) : ['counter', ...(cooler ? ['cooler'] : []), ...gondola.slice(0, ng)]);
  B.zones = new Set(Object.keys(AZ).filter(z => used.has(AZ[z])));
  const cov = coverage(), home = {};      // camera -> [the aisle it mostly looks at, slots seen there]
  for (const vc of sim.vcams) { const n = {}; for (const [z, k] of Object.entries(cov[vc.name])) n[AZ[z]] = (n[AZ[z]] ?? 0) + k; home[vc.name] = Object.entries(n).sort((a, b) => b[1] - a[1])[0] ?? ['none', 0]; }
  const kind = Object.fromEntries(Y.layout().cameras.map(c => [c.id, c.kind]));
  const people = sim.vcams.filter(vc => kind[vc.name] === 'entrance' || kind[vc.name] === 'overhead' || vc.name === 'REGISTER-top').map(v => v.name);
  const item = sim.vcams.filter(vc => !people.includes(vc.name) && ['shelf', 'cooler', 'checkout'].includes(kind[vc.name]) && home[vc.name][1] >= MIN_SLOTS && used.has(home[vc.name][0])).map(v => v.name);
  B.cams = [...people, ...item];
  // the camera pass at one strength for every effect; people and shelf mess stay as synth.js makes them
  const amt = opts.realism ?? 0.5;
  real.set(amt > 0 ? { ...real.PRESETS.on, people: false, shelves: false, seed, amount: Object.fromEntries(real.CAMERA_EFFECTS.map(k => [k, amt])) } : { ...real.PRESETS.off, seed });
  real.setLight(feat.night ? 'night' : 'day');
  const kinds = new Set(store.slots.filter(s => s.stand && B.zones.has(s.zoneId)).map(s => s.zone));
  const sc = planScenario2(seed, { window: opts.window ?? 24, feat, kinds });
  Object.assign(B, { plans: sc.plans, arrive: sc.arrive, rng: sc.rng, dr: mulberry32(seed * 6151 + 29), pair: sc.pair });
  // camera faults, drawn from the cameras of the clip (not from what they see)
  const deg = () => (F() < 0.5 ? -1 : 1) * (1.5 + 2 * F()), rad = d => d * Math.PI / 180;
  B.faults = { bump: [], block: [] };
  if (feat.bump) for (const c of shuffled(B.cams, F).slice(0, 2)) B.faults.bump.push({ camera: c, t: +(12 + 30 * F()).toFixed(1), yaw_deg: +deg().toFixed(2), pitch_deg: +deg().toFixed(2), roll_deg: +(deg() * 0.4).toFixed(2) });
  if (feat.block) { const t0 = +(10 + 30 * F()).toFixed(1); B.faults.block.push({ camera: shuffled(item, F)[0], t0, t1: +(t0 + 10 + 15 * F()).toFixed(1) }); }
  B.bumpRad = B.faults.bump.map(b => [rad(b.yaw_deg), rad(b.pitch_deg), rad(b.roll_deg)]);
  const clerk = scene.children.find(o => o.isGroup && Math.hypot(o.position.x - 6.2, o.position.z - 3.3) < 1e-6 && o.children.length && !o.userData.vcam);
  if (!clerk) throw new Error('bench: clerk figure not found');
  tagPerson(clerk, 0);
  // what the pipeline is given is fixed here, before anything moves: the pose a calibration measured, the planogram on file
  const plan = planogram(), L = layoutOut(); for (const s of L.slots) s.skuId = plan.nominal[s.id];
  B.given = { calibration: calibration(), layout: L, planogram: plan };
  return { ...params, gen: 2, features: feat, aisles: [...used].sort(), zones: [...B.zones].sort(), cameras: B.cams, item_cameras: item.length, realism: real.get(), shutter_s: real.shutter(),
    aisle_of_zone: AZ, camera_aisle: Object.fromEntries(Object.entries(home).filter(([c]) => !people.includes(c)).map(([c, h]) => [c, h[0]])),
    shoppers: sc.plans.length, window: opts.window ?? 24, thieves: sc.plans.filter(p => p.thief).length, tags: sc.tags, faults: B.faults,
    same_shelf_pair: sc.pair.map(i => `P${String(i + 1).padStart(3, '0')}`), planogram_slots_where_the_item_differs: Object.keys(plan.exact).filter(k => plan.exact[k] !== plan.nominal[k]).length };
}

// ---- stepping and capture. One frame is 0.1 s: two steps of 1/30 s, then two of 1/60 s. The shutter is open over the last
// steps (1/60 s by day, 1/30 s at night), and a camera averages what it saw then. A dry run takes the same steps.
const SUB = [1 / 30, 1 / 30, 1 / 60, 1 / 60];
const cover = new THREE.Mesh(new THREE.PlaneGeometry(3, 3), new THREE.MeshStandardMaterial({ color: 0x5a4a38, roughness: 1, side: THREE.DoubleSide }));
cover.visible = false; cover.frustumCulled = false; scene.add(cover);
const blockedAt = (name, t) => !!B.faults?.block.some(b => b.camera === name && t >= b.t0 && t < b.t1);
function withCover(vc, fn) { // a covered camera looks at a sheet of cardboard a hand's width from the lens
  if (!vc._blocked) return fn();
  cover.position.copy(vc.cam.position).add(vc.cam.getWorldDirection(new THREE.Vector3()).multiplyScalar(0.25)); cover.quaternion.copy(vc.cam.quaternion); cover.visible = true; cover.updateMatrixWorld(true);
  try { return fn(); } finally { cover.visible = false; }
}
function frame2(i, render = true) {
  const t = i / 10;
  B.faults.bump.forEach((b, k) => { if (!b.done && t >= b.t) { const vc = camOf(b.camera), [y, p, r] = B.bumpRad[k]; vc.cam.rotateY(y); vc.cam.rotateX(p); vc.cam.rotateZ(r); vc.cam.updateMatrixWorld(true); b.done = true; B.rev++; } });
  let need = [];
  if (render) need = B.cams.map(camOf).filter(vc => {
    vc.cam.updateMatrixWorld(true); vc._blocked = blockedAt(vc.name, t); vc._key = `${sim.events.length}:${B.rev}:${vc._blocked}`; vc._busy = !vc._blocked && busy(vc);
    return vc._busy || still.get(vc.name) !== vc._key;
  });
  const n = real.shutter() > 0 ? (real.light() === 'night' ? 3 : 2) : 0;
  if (i > 0) SUB.forEach((dt, k) => { stepBy(dt); if (n && k >= SUB.length - n) need.forEach((vc, j) => withCover(vc, () => expose2(vc, j === 0))); });
  sim.time = t;
  return { t, world: world(), need: need.map(v => v.name), inStore: sim.shoppers.length, pending: B.arrive.length - B.k };
}
const idRTs = new Map();
// Every read of pixels from the GPU waits for the GPU, and on a busy machine that wait is most of a frame (measured: about 50 ms a
// read, two reads per camera). So the exposures are summed here (as realism.js expose() does), each camera's picture and id render
// are read into a pixel buffer object without waiting, and the buffers are fetched together once every camera has been drawn.
const acc2 = new Map(), pbos = new Map();
function expose2(vc, shadow) {
  const [W, H] = real.srcSize(vc, ...vc.res), src = real.target(W, H, 'src');
  let a = acc2.get(vc.name);
  if (!a || a.rt.width !== W || a.rt.height !== H) { a?.rt.dispose(); a = { rt: new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: false }), n: 0 }; acc2.set(vc.name, a); }
  if (shadow) renderer.shadowMap.needsUpdate = true;
  real.renderScene(vc, src);
  a.rt.viewport.set(0, 0, W, H); a.rt.scissorTest = false; renderer.setRenderTarget(a.rt);
  const m = real.copyMat; m.uniforms.map.value = src.texture; m.uniforms.k.value = 1; m.blending = THREE.AdditiveBlending;
  if (!a.n) { renderer.setClearColor(0x000000, 1); renderer.clear(); }
  real.drawQuad(m); a.n++; renderer.setRenderTarget(null);
}
function queueRead(key, rt, w, h) {
  const gl = renderer.getContext(); let b = pbos.get(key);
  if (!b || b.size !== w * h * 4) { if (b) gl.deleteBuffer(b.buf); b = { buf: gl.createBuffer(), size: w * h * 4 }; gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b.buf); gl.bufferData(gl.PIXEL_PACK_BUFFER, b.size, gl.STREAM_READ); pbos.set(key, b); }
  renderer.setRenderTarget(rt); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b.buf); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, 0); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  return b;
}
function fetchRead(b, out) { const gl = renderer.getContext(); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b.buf); gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, out); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null); return out; }
// JPEG encoding and the hand-over to render_clip.mjs happen in a few workers: the pixels of a frame are passed to a worker, which
// encodes them and posts the JPEG to the address render_clip.mjs listens on. (A data URL through the debugger took most of a frame's time.)
const enc = { workers: null, pending: new Map(), n: 0, size: 4 };
function encode(px, w, h, url, quality) {
  if (!enc.workers) {
    const src = `onmessage = async e => { const { id, url, w, h, buf, quality } = e.data; try { const c = new OffscreenCanvas(w, h); c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(buf), w, h), 0, 0);
      const b = await c.convertToBlob({ type: 'image/jpeg', quality }); const r = await fetch(url, { method: 'POST', body: new Blob([b], { type: 'text/plain' }) }); if (!r.ok) throw new Error('post ' + r.status); postMessage({ id }); } catch (err) { postMessage({ id, error: String(err) }); } };`;
    const u = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    enc.workers = Array.from({ length: enc.size }, () => { const k = new Worker(u); k.onmessage = e => { const p = enc.pending.get(e.data.id); enc.pending.delete(e.data.id); if (e.data.error) p[1](new Error(e.data.error)); else p[0](); }; return k; });
  }
  return new Promise((ok, no) => { const id = ++enc.n; enc.pending.set(id, [ok, no]); enc.workers[id % enc.workers.length].postMessage({ id, url, w, h, buf: px.buffer, quality }, [px.buffer]); });
}
// The frame of each named camera (the cameras frame2 said need one), with the ground truth read from an id render at half size.
// The pictures go to `post` as JPEG (one POST per camera, ?f=frame&c=camera); the truth rows come back once every picture has arrived.
function shots(names, i, post, quality = 0.9) {
  const t = i / 10, held = new Map(sim.shoppers.flatMap(sh => sh.items.map(it => [it.mesh, { sh, it }]))), lo = B.minItem, sent = [];
  const drawn = names.map((name, j) => {
    const vc = camOf(name), [w, h] = vc.res, p = real.camParams(vc), s = p.overscan, a = p.a, Wi = Math.ceil(w * s / 2), Hi = Math.ceil(h * s / 2);
    const acc = acc2.get(name), ldr = real.target(w, h, 'ldr');
    let tex, scale = 1;
    if (acc && acc.n) { tex = acc.rt.texture; scale = 1 / acc.n; acc.n = 0; }
    else { if (j === 0) renderer.shadowMap.needsUpdate = true; const src = real.target(...real.srcSize(vc, w, h), 'src'); withCover(vc, () => real.renderScene(vc, src)); tex = src.texture; }
    ldr.viewport.set(0, 0, w, h); ldr.scissorTest = false; renderer.setRenderTarget(ldr);
    real.develop(vc, tex, { size: [w, h], flipY: true, srcScale: scale, frame: i, t });
    const picture = queueRead('p:' + name, ldr, w, h);
    let rt = idRTs.get(`${Wi}x${Hi}`); if (!rt) { rt = new THREE.WebGLRenderTarget(Wi, Hi, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }); idRTs.set(`${Wi}x${Hi}`, rt); }
    const hidden = [], fov = vc.cam.fov;
    scene.traverse(o => { if (o.isMesh && o.visible && [].concat(o.material).some(m => m.transparent)) { o.visible = false; hidden.push(o); } });
    vc.cam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(s * Math.tan(vc.vfov / 2))); vc.cam.updateProjectionMatrix();      // the lens shows a little more than the pinhole field
    scene.overrideMaterial = idMat; renderer.setRenderTarget(rt); renderer.setClearColor(0x000000, 1); renderer.clear(); withCover(vc, () => renderer.render(scene, vc.cam));
    scene.overrideMaterial = null; hidden.forEach(o => { o.visible = true; }); vc.cam.fov = fov; vc.cam.updateProjectionMatrix();
    const ids = queueRead('i:' + name, rt, Wi, Hi); renderer.setRenderTarget(null);
    return { name, vc, w, h, p, s, a, Wi, Hi, picture, ids };
  });
  const rows = drawn.map(({ name, vc, w, h, p, s, a, Wi, Hi, picture, ids }) => {
    sent.push(encode(fetchRead(picture, new Uint8ClampedArray(w * h * 4)), w, h, `${post}?f=${i}&c=${encodeURIComponent(name)}`, quality));
    const buf = fetchRead(ids, new Uint8Array(Wi * Hi * 4));
    const box = new Map(), fx = vc.fx, fy = vc.fy, kx = 2 * p.tanH * s / Wi, ky = 2 * p.tanV * s / Hi;
    for (let r = 0, q = 0; r < Hi; r++) for (let x = 0; x < Wi; x++, q += 4) {
      const id = buf[q] | (buf[q + 1] << 8) | (buf[q + 2] << 16);
      if (id < lo) continue;
      const xu = (x + 0.5 - Wi / 2) * kx, yu = (Hi / 2 - r - 0.5) * ky, k = 1 / (1 + a * (xu * xu + yu * yu)), u = xu * k * fx + w / 2, v = yu * k * fy + h / 2;     // r counts from the bottom row; v from the top
      if (u < 0 || u >= w || v < 0 || v >= h) continue;
      let b = box.get(id); if (!b) box.set(id, b = [u, v, u, v, 0]);
      if (u < b[0]) b[0] = u; if (u > b[2]) b[2] = u; if (v < b[1]) b[1] = v; if (v > b[3]) b[3] = v; b[4]++;
    }
    const bb = b => [Math.max(0, Math.floor(b[0] - 1)), Math.max(0, Math.floor(b[1] - 1)), Math.min(w, Math.ceil(b[2] + 1)), Math.min(h, Math.ceil(b[3] + 1))];
    const pt = P => { const q = px(vc, P); if (!q) return null; const d = real.distortPoint(vc, q); return d && d.map(v => +v.toFixed(1)); };
    const items = [], persons = [];
    for (const [id, b] of box) {
      if (id >= PERSON) {
        const n = id - PERSON, sh = sim.shoppers.find(x => x.id === n);
        if (n && !sh) continue;
        const row = { shopper: n ? sh.name : 'clerk', bbox: bb(b), vis_px: b[4] * 4 };
        if (sh) Object.assign(row, { hand: pt(sh.fig.handWorld(0)), head: pt(_v.copy(sh.pos).setY(sh.height - sh.crouch - 0.1).clone()), feet: pt(sh.pos.clone()), state: sh.state });
        persons.push(row);
      } else {
        const r = B.items.get(id), st = r && held.get(r.mesh);
        if (r) items.push({ sku: r.sku, kind: r.misplaced ? 'misplaced' : st?.it.onCounter ? 'counter' : 'hand', bbox: bb(b), vis_px: b[4] * 4, shopper: st?.sh.name ?? null, slot: r.misplaced ?? st?.it.slot.label ?? null });
      }
    }
    if (vc._busy) still.delete(name); else still.set(name, vc._key);
    return { name, persons, items };
  });
  return Promise.all(sent).then(() => rows);
}
// A whole visit without rendering: how long it is and what happens in it.
function run2(maxT = 240) { let i = 0, st; do st = frame2(i++, false); while ((st.inStore || st.pending) && st.t < maxT); return { end: st.t }; }
// staff takes are not shopper picks: they are listed with the touches and the staff puts
const acts = () => [...B.acts, ...sim.events.filter(e => e.staff).map(e => { const s = fixtureOf.get(e.slot); return { kind: 'staff_take', t: e.t, shopper: e.shopper, skuId: e.sku, slotId: e.slot, fixtureId: s.fixtureId, zone: e.zone, zoneId: e.zoneId,
  cameras: e.views.map(v => ({ id: v.cam, kind: v.kind, px: v.px })), outcome: e.outcome === 'put_back' ? 'moved' : e.outcome === 'staff_removed' ? 'carried_off' : 'in_hand', tPut: e.putBackT ?? null, putSlot: e.putBackSlot ?? null }; })].sort((x, y) => x.t - y.t);

window.__breeBench = { setup, advance, capture, world, events, calibration, layout: layoutOut, shoppers: () => B.shoppers, plan: () => ({ arrive: B.arrive, plans: B.plans }),
  state: () => ({ t: sim.time, inStore: sim.shoppers.length, pending: B.arrive.length - B.k }),
  frame2, shots, run2, acts, given: () => B.given, faults: () => B.faults, FEATURES, encoders: n => { enc.size = n; } };
