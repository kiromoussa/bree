// Dataset overlay for the PRIVATE simulator copy (bree-vision/out/sim-copy/js/synth.js). SIMULATED DATA.
// Loaded into the page by scripts/train/render_synth.mjs; the original simulator files are not changed.
// Adds: domain randomisation (lights, exposure, colour cast of the lights, shopper clothing, slot fill,
// item pose jitter, same-size SKU swaps, camera pose jitter), a seeded shopper scenario, and a capture
// that returns the colour frame plus ground-truth boxes read from an instance-id render (so a box is
// what is actually visible, with hands, bodies, neighbours and shelves hiding what they hide).
import * as THREE from 'three';
import { Shopper, Figure, resetShopperIds } from './agents.js';
import { CAN_W, pixelsAt } from './cameras.js';
import { SKUS } from './textures.js';
import { mulberry32 } from './harness.js';

const sim = window.__sim, store = window.__store, renderer = window.__renderer, api = window.__breeSim;
const scene = sim.scene, canvas = renderer.domElement;
const key = scene.children.find(o => o.isDirectionalLight), hemi = scene.children.find(o => o.isHemisphereLight);
const BASE = { key: key.intensity, keyPos: key.position.clone(), hemi: hemi.intensity, exposure: renderer.toneMappingExposure };

// ---------- every drawn item: front facings (the slots) then backstock, in store.js build order ----------
const VIS_DEPTH = { cooler: 3, gondola: 2, checkout: 2, tobacco: 2 }; // same numbers as store.js
const reg = [null];   // id -> record; id 0 is "not an item"
{
  const next = new Map(store.itemMeshes.map(m => [m, 0]));
  for (const s of store.slots) next.set(s.mesh, next.get(s.mesh) + 1);
  for (const s of store.slots) { s._sku0 = s.sku; reg.push({ kind: 'shelf', mesh: s.mesh, idx: s.idx, slot: s, depth: 0, base: s.matrix.clone(), cur: s.matrix.clone() }); }
  const p = new THREE.Vector3();
  for (const s of store.slots) for (let k = 1; k < Math.min(s.depthCount, VIS_DEPTH[s.zone]); k++) {
    const idx = next.get(s.mesh); next.set(s.mesh, idx + 1);
    const m = new THREE.Matrix4(); s.mesh.getMatrixAt(idx, m); p.setFromMatrixPosition(m).sub(s.center);
    const along = -p.dot(s.normal);
    if (!(along > 0.005) || Math.abs(p.length() - along) > 1e-3) throw new Error(`synth: backstock order does not match store.js at ${s.label}`);
    reg.push({ kind: 'backstock', mesh: s.mesh, idx, slot: s, depth: k, base: m, cur: m.clone() });
  }
  for (const [m, n] of next) if (n !== m.count) throw new Error('synth: item instance count does not match store.js');
}
const N_SHELF = reg.length;
const idRGB = id => [(id & 255) / 255, ((id >> 8) & 255) / 255, ((id >> 16) & 255) / 255];
for (const m of store.itemMeshes) m.geometry.setAttribute('idColor', new THREE.InstancedBufferAttribute(new Float32Array(m.count * 3), 3));
for (let id = 1; id < N_SHELF; id++) reg[id].mesh.geometry.attributes.idColor.setXYZ(reg[id].idx, ...idRGB(id));
// items in a hand or on the counter get an id when the sim creates them
const makeItem0 = store.makeItemMesh;
store.makeItemMesh = sku => {
  const m = makeItem0(sku), id = reg.length;
  m.geometry.setAttribute('idColor', new THREE.InstancedBufferAttribute(new Float32Array(idRGB(id)), 3));
  reg.push({ kind: 'hand', mesh: m, sku }); return m;
};
const pay0 = sim.onPay; sim.onPay = (sh, items) => { items.forEach(i => { i.event.payT = +sim.time.toFixed(2); }); pay0(sh, items); };

// SKUs that share a shape and size can stand in each other's slots (only the label art differs)
const FAMILY = {}; for (const s of SKUS) (FAMILY[s.geo + s.size.join('x')] ??= []).push(s);
const family = s => FAMILY[s.geo + s.size.join('x')];

const idMat = new THREE.ShaderMaterial({
  vertexShader: 'attribute vec3 idColor; varying vec3 vId;\nvoid main() { vId = idColor; vec4 p = vec4(position, 1.0);\n#ifdef USE_INSTANCING\n p = instanceMatrix * p;\n#endif\n gl_Position = projectionMatrix * modelViewMatrix * p; }',
  fragmentShader: 'varying vec3 vId; void main() { gl_FragColor = vec4(vId, 1.0); }',
});
let idRT = null;

const S = { arrive: [], k: 0, theft: 0.4, dr: Math.random, params: null };
const ZERO = new THREE.Matrix4().makeScale(0, 0, 0);
const SKIN = [0xc69c7b, 0x8d5a3b, 0xe0b899, 0x5c3a24, 0xb57f5a, 0xf1d0b5, 0x70452b, 0xd8a47f];

// The clerk behind the counter. main.js builds it once per page load and leaves its sleeve length to
// Math.random, so two renders of the same seed differed in the clerk's forearm (a few hundred pixels in
// every frame of a camera that sees the counter from above; found by the audit of 2026-10-05). It is
// rebuilt per seed from its own generator. The DR stream and the shopper stream are not touched, so every
// other pixel of a seed is what it was. The constants are main.js's (simulator commit a4f479f).
const CLERK = { look: { shirt: 0x2b5f8a, pants: 0x1d1f23, skin: 0xb57f5a, hair: 0x1a1410, scale: 0.98 }, pos: new THREE.Vector3(6.2, 0, 3.3), hand: new THREE.Vector3(5.62, 1.02, 3.15) };
let clerk = scene.children.find(o => o.isGroup && o.position.distanceTo(CLERK.pos) < 1e-6 && Math.abs(o.scale.x - CLERK.look.scale) < 1e-9);
if (!clerk) throw new Error('synth: clerk figure not found where main.js puts it (the simulator changed)');
function seedClerk(seed) {
  const fig = new Figure({ ...CLERK.look, longSleeve: mulberry32(seed * 104729 + CLERK_SALT)() < 0.5 }), i = scene.children.indexOf(clerk);
  clerk.traverse(o => o.geometry?.dispose());
  scene.children[i] = fig.root; fig.root.parent = scene; clerk.parent = null; clerk = fig.root; // same place in the scene's child list
  fig.update({ heading: -Math.PI / 2, pos: CLERK.pos, hands: [CLERK.hand, null] });
}
const CLERK_SALT = 5; // any number; 5 makes seed 5001 draw the short sleeve the recorded clip (data/synth/clip_5001_door) happened to get

// One randomised scene. layout: version 1 layout (cameras). Everything here draws from the DR stream;
// shoppers draw from their own stream (sim.rand), so shopper behaviour for a seed does not depend on DR options.
function setup(seed, layout, { shoppers = 10, window = 20, theft = 0.4, jitter = true, swap = true } = {}) {
  const R = mulberry32(seed * 7919 + 13), u = (a, b) => a + (b - a) * R();
  S.dr = R;
  // cameras: small mounting error on top of the layout
  const J = jitter ? { pos: 0.03, ang: 0.026, fov: 0.03, roll: 0.02 } : { pos: 0, ang: 0, fov: 0, roll: 0 };
  const cams = layout.cameras.map(c => ({ id: c.id, kind: c.kind, resolution: c.resolution, position: c.position.map(v => v + u(-J.pos, J.pos)),
    yaw: c.yaw + u(-J.ang, J.ang), pitch: c.pitch + u(-J.ang, J.ang), hfov: c.hfov * (1 + u(-J.fov, J.fov)) }));
  sim.shoppers.forEach(s => s.dispose()); sim.shoppers.length = 0; sim.events.length = 0; sim.tracks.length = 0; sim.time = 0;
  seedClerk(seed);
  api.loadLayout({ version: 1, cameras: cams });
  sim.vcams.forEach(vc => { vc.cam.rotation.z = u(-J.roll, J.roll); vc.cam.updateMatrixWorld(true); });
  // lights: brightness, direction, warm or cool cast, exposure
  const warm = u(-1, 1), tint = k => new THREE.Color(1 + 0.12 * Math.max(0, k), 1 - 0.04 * Math.abs(k), 1 + 0.16 * Math.max(0, -k));
  key.intensity = BASE.key * u(0.3, 1.5); key.color.copy(tint(warm));
  key.position.copy(BASE.keyPos).add(new THREE.Vector3(u(-8, 8), u(-3, 3), u(-8, 8)));
  hemi.intensity = u(0.12, 0.8); hemi.color.copy(tint(warm * 0.6 + u(-0.3, 0.3)));
  renderer.toneMappingExposure = BASE.exposure * u(0.5, 1.5);
  // shelves: SKU swaps inside a size family, fill level, item pose jitter
  const fill = u(0.45, 1), q = new THREE.Quaternion(), p = new THREE.Vector3(), sc = new THREE.Vector3(), qj = new THREE.Quaternion(), Y = new THREE.Vector3(0, 1, 0), tan = new THREE.Vector3();
  let prev = null, run = null;
  for (const s of store.slots) {
    s.reserved = false;
    const fam = family(s._sku0);
    if (!prev || prev._sku0 !== s._sku0 || prev.fixtureId !== s.fixtureId) run = swap ? fam[Math.floor(R() * fam.length)] : s._sku0; // a planogram block is restocked with one variant
    s.sku = swap && R() < 0.12 ? fam[Math.floor(R() * fam.length)] : run;                                                              // and a few single items are misplaced
    prev = s;
  }
  for (let id = 1; id < N_SHELF; id++) {
    const r = reg[id], s = r.slot, on = R() < fill;
    r.base.decompose(p, q, sc);
    tan.set(-s.normal.z, 0, s.normal.x);
    p.addScaledVector(tan, u(-0.004, 0.004)).addScaledVector(s.normal, -u(0, 0.015));
    q.multiply(qj.setFromAxisAngle(Y, u(-0.14, 0.14)));
    r.cur.compose(p, q, sc); r.on = on;
    r.mesh.setMatrixAt(r.idx, on ? r.cur : ZERO);
    r.mesh.geometry.attributes.uvRect.setXYZW(r.idx, ...store.atlas.rects[s.sku.id]);
    if (r.depth === 0) { s.matrix = r.cur; s.present = on; }
  }
  for (const m of store.itemMeshes) { m.instanceMatrix.needsUpdate = true; m.geometry.attributes.uvRect.needsUpdate = true; }
  for (let i = 0; i < store.doorOpen.length; i++) store.setDoorOpen(i, 0);
  // shoppers
  reg.length = N_SHELF; resetShopperIds();
  const rand = sim.rand = mulberry32(seed);
  S.arrive = Array.from({ length: shoppers }, (_, k) => (k + rand()) * window / shoppers); S.k = 0; S.theft = theft;
  S.params = { seed, shoppers, window, theft, fill: +fill.toFixed(3), keyIntensity: +key.intensity.toFixed(3), hemiIntensity: +hemi.intensity.toFixed(3), exposure: +renderer.toneMappingExposure.toFixed(3), warm: +warm.toFixed(3), jitter, swap };
  return S.params;
}

function spawn(thief) {
  const sh = new Shopper(sim, thief), R = S.dr, old = sh.fig, c = (s0, s1, l0, l1) => new THREE.Color().setHSL(R(), s0 + (s1 - s0) * R(), l0 + (l1 - l0) * R()).getHex();
  // clothing: any hue, not the sim's short palette (and no dark-shirt tell for thieves)
  sh.group.remove(old.root);
  sh.fig = new Figure({ shirt: c(0, 0.9, 0.08, 0.9), pants: c(0, 0.6, 0.06, 0.6), skin: SKIN[Math.floor(R() * SKIN.length)], hair: c(0, 0.5, 0.03, 0.6), shoes: c(0, 0.5, 0.05, 0.8), longSleeve: R() < 0.5, scale: old.scale });
  sh.group.add(sh.fig.root); sh.sync(0);
  sim.shoppers.push(sh);
}

// Same stepping as main.js step(), without the UI arrivals.
function advance(until, dt = 1 / 30) {
  while (sim.time < until - 1e-9) {
    while (S.k < S.arrive.length && S.arrive[S.k] <= sim.time) { spawn(sim.rand() < S.theft); S.k++; }
    sim.time += dt;
    for (const sh of sim.shoppers) sh.update(dt);
    for (const sh of sim.shoppers.filter(s => s.done)) {
      sh.items.forEach(i => { i.event.exitT = +sim.time.toFixed(2); if (!i.paid) i.event.outcome = 'theft'; });
      sh.dispose(); sim.shoppers.splice(sim.shoppers.indexOf(sh), 1);
    }
  }
  return { t: sim.time, inStore: sim.shoppers.length, pending: S.arrive.length - S.k };
}

const carried = () => sim.shoppers.flatMap(sh => sh.items.filter(i => !i.concealed && !i.hidden && !i.paid).map(i => ({ sh, it: i })));
// Cameras worth rendering now: up to nHand that see an item in a hand or on the counter, plus nRand other item cameras.
function pickCameras(nHand = 3, nRand = 3, kinds = ['shelf', 'cooler', 'checkout']) {
  const R = S.dr, shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const held = carried(), seeHand = new Set();
  for (const vc of sim.vcams) { vc.cam.updateMatrixWorld(true); if (kinds.includes(vc.kind) && held.some(({ it }) => pixelsAt(vc, it.mesh.position, store.occluders) >= 8)) seeHand.add(vc.name); }
  const a = shuffle([...seeHand]).slice(0, nHand);
  const b = shuffle(sim.vcams.filter(vc => kinds.includes(vc.kind) && !a.includes(vc.name)).map(vc => vc.name)).slice(0, nRand);
  return { hand: a, random: b };
}

function renderColor(vc) {
  const [w, h] = vc.res;
  vc.cam.updateMatrixWorld(true);
  renderer.setPixelRatio(1); renderer.setSize(w, h, false); renderer.setRenderTarget(null); renderer.setScissorTest(false); renderer.setViewport(0, 0, w, h);
  renderer.setClearColor(0x0d1014, 1); renderer.clear(); renderer.shadowMap.needsUpdate = true; renderer.render(scene, vc.cam);
}
const camOf = name => { const vc = sim.vcams.find(v => v.name === name); if (!vc) throw new Error('no camera ' + name); return vc; };
const CORNERS = [-0.5, 0.5].flatMap(x => [-0.5, 0.5].flatMap(y => [-0.5, 0.5].map(z => [x, y, z])));

// Colour frame plus ground truth for one camera. Boxes are tight around the pixels of the item that are visible.
function capture(name, { quality = 0.92, type = 'image/jpeg' } = {}) {
  const vc = camOf(name), [w, h] = vc.res;
  renderColor(vc);
  const image = canvas.toDataURL(type, quality);
  if (!idRT || idRT.width !== w || idRT.height !== h) { idRT?.dispose(); idRT = new THREE.WebGLRenderTarget(w, h, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }); }
  const hidden = []; // glass does not hide what is behind it
  scene.traverse(o => { if (o.isMesh && o.visible && [].concat(o.material).some(m => m.transparent)) { o.visible = false; hidden.push(o); } });
  scene.overrideMaterial = idMat; renderer.setRenderTarget(idRT); renderer.setClearColor(0x000000, 1); renderer.clear(); renderer.render(scene, vc.cam);
  scene.overrideMaterial = null; hidden.forEach(o => { o.visible = true; });
  const buf = new Uint8Array(w * h * 4); renderer.readRenderTargetPixels(idRT, 0, 0, w, h, buf); renderer.setRenderTarget(null);
  const n = reg.length, x0 = new Int32Array(n).fill(1e9), x1 = new Int32Array(n).fill(-1), y0 = new Int32Array(n).fill(1e9), y1 = new Int32Array(n).fill(-1), cnt = new Int32Array(n);
  for (let r = 0, i = 0; r < h; r++) { const y = h - 1 - r; for (let x = 0; x < w; x++, i += 4) { const id = buf[i] | (buf[i + 1] << 8) | (buf[i + 2] << 16); if (id && id < n) { cnt[id]++; if (x < x0[id]) x0[id] = x; if (x > x1[id]) x1[id] = x; if (y < y0[id]) y0[id] = y; if (y > y1[id]) y1[id] = y; } } }
  const state = new Map(sim.shoppers.flatMap(sh => sh.items.map(it => [it.mesh, { shopper: sh.name, state: sh.state, counter: !!it.onCounter }])));
  const C = vc.cam.getWorldPosition(new THREE.Vector3()), P = new THREE.Vector3(), v = new THREE.Vector3(), M = new THREE.Matrix4(), anns = [];
  for (let id = 1; id < n; id++) {
    if (!cnt[id]) continue;
    const r = reg[id], shelf = id < N_SHELF, sku = shelf ? r.slot.sku : r.sku;
    if (shelf) M.copy(r.cur); else { r.mesh.updateMatrixWorld(true); M.copy(r.mesh.matrixWorld); }
    // box the item would fill if nothing hid it (projected corners of its bounding box); null when part is behind the camera
    let fx0 = 1e9, fx1 = -1e9, fy0 = 1e9, fy1 = -1e9, ok = true;
    for (const c of CORNERS) { v.set(...c).applyMatrix4(M).project(vc.cam); if (v.z > 1 || v.z < -1) { ok = false; break; } const px = (v.x + 1) / 2 * w, py = (1 - v.y) / 2 * h; fx0 = Math.min(fx0, px); fx1 = Math.max(fx1, px); fy0 = Math.min(fy0, py); fy1 = Math.max(fy1, py); }
    // pixels across a 6.6 cm can at this item, the simulator's metric (effective_px_v1) without its lens edge term, which the render does not have
    P.setFromMatrixPosition(M); let cos = 1;
    if (shelf) { P.addScaledVector(r.slot.normal, sku.size[2] / 2); cos = Math.max(0, v.copy(C).sub(P).normalize().dot(r.slot.normal)); }
    const range = C.distanceTo(P), st = state.get(r.mesh);
    anns.push({ id, sku: sku.id, kind: shelf ? r.kind : st?.counter ? 'counter' : 'hand', slot: shelf ? r.slot.label : null, geo: sku.geo,
      bbox: [x0[id], y0[id], x1[id] + 1, y1[id] + 1], vis_px: cnt[id], full: ok ? [fx0, fy0, fx1, fy1].map(a => +a.toFixed(1)) : null,
      px_eff: +(CAN_W * vc.fx / range * cos).toFixed(2), px_item: +(sku.size[0] * vc.fx / range).toFixed(2), range: +range.toFixed(3), cos: +cos.toFixed(3),
      ...(st ? { shopper: st.shopper, shopper_state: st.state } : {}) });
  }
  return { image, width: w, height: h, t: +sim.time.toFixed(2), camera: { id: vc.name, kind: vc.kind, hfov: +vc.hfov.toFixed(3), fx: +vc.fx.toFixed(2) }, annotations: anns };
}


// Ground truth of the scenario in the Isaac kit's events.jsonl shape (one line per picked item).
function events(fps = 30) {
  return sim.events.map(e => ({ t: e.t, frame: Math.round(e.t * fps), shopper: e.shopper, thief: e.thief, skuId: e.sku, slotId: e.slot,
    outcome: e.concealed ? 'concealed' : e.outcome === 'paid' ? 'paid' : 'in_hand', tResolved: e.concealed ? e.concealT : e.payT ?? null, tExit: e.exitT ?? null,
    cameras: e.views.map(v => ({ id: v.cam, px: v.px })) }));
}

window.__breeSynth = { setup, advance, pickCameras, capture, events, layout: () => api.getLayout(), skus: SKUS.map(s => ({ id: s.id, name: s.name, group: s.group, geo: s.geo, size: s.size, family: family(s).map(f => f.id) })),
  state: () => ({ t: sim.time, inStore: sim.shoppers.length, pending: S.arrive.length - S.k, carried: carried().length, params: S.params }) };
