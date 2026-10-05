// Hand and held-item overlay for the PRIVATE simulator copy (bree-vision/out/sim-copy/js/hands.js). SIMULATED DATA.
// Loaded after scripts/train/sim/synth.js and scripts/bench/sim/bench.js by scripts/train/render_hands.mjs; neither
// of those files is edited (the benchmark records their hashes). The scene and the item boxes come from synth.js, the
// shopper behaviour (reach, put back, hold, conceal, counter, two shoppers at one shelf) from bench.js. This file adds:
//   - a ground-truth box for every visible HAND (the hand and the last 7 cm of the forearm), read from an id render,
//     with whether that arm is reaching and whether it holds an item
//   - a choice of capture moments and cameras that puts reaches, held items and put-backs in most frames
import * as THREE from 'three';
import { CAN_W, pixelsAt } from './cameras.js';
import { mulberry32 } from './harness.js';

const Y = window.__breeSynth, B = window.__breeBench, sim = window.__sim, store = window.__store, renderer = window.__renderer;
const scene = sim.scene;
const HAND = 0xE00000, PERSON = 0xF00000;   // id render: HAND + 2 * shopper number + arm (0 = the reaching arm); bench.js uses PERSON + n
const WRIST_Y = -0.2;                        // forearm mesh, local metres: everything below this is "hand" (the forearm is 0.27 long)
const idRGB = id => [(id & 255) / 255, ((id >> 8) & 255) / 255, ((id >> 16) & 255) / 255];
const idMat = new THREE.ShaderMaterial({
  vertexShader: 'attribute vec3 idColor; varying vec3 vId;\nvoid main() { vId = idColor; vec4 p = vec4(position, 1.0);\n#ifdef USE_INSTANCING\n p = instanceMatrix * p;\n#endif\n gl_Position = projectionMatrix * modelViewMatrix * p; }',
  fragmentShader: 'varying vec3 vId; void main() { gl_FragColor = vec4(vId, 1.0); }',
});
let idRT = null, R = Math.random;

// Whole triangles get the hand id (the id is interpolated across a triangle, so a mixed triangle would invent ids).
function tagFore(mesh, id) {
  if (mesh.userData.handId === id) return;
  const g = mesh.geometry, pos = g.attributes.position, n = pos.count, rgb = idRGB(id);
  const a = g.attributes.idColor?.count === n ? g.attributes.idColor.array : new Float32Array(n * 3);
  for (let i = 0; i < n; i += 3) if ((pos.getY(i) + pos.getY(i + 1) + pos.getY(i + 2)) / 3 < WRIST_Y) for (let k = 0; k < 3; k++) a.set(rgb, (i + k) * 3);
  g.setAttribute('idColor', new THREE.BufferAttribute(a, 3));
  mesh.userData.handId = id;
}
const clerkRoot = () => scene.children.find(o => o.isGroup && Math.hypot(o.position.x - 6.2, o.position.z - 3.3) < 1e-6 && o.children.length && !o.userData.vcam);
function tagAll() {
  for (const sh of sim.shoppers) sh.fig.arms.forEach((arm, i) => tagFore(arm.fore, HAND + 2 * sh.id + i));
  const spine = clerkRoot()?.children[0]?.children[0];      // Figure: root > hips > spine > [body, upper, fore, upper, fore]
  if (spine) [spine.children[2], spine.children[4]].forEach((m, i) => tagFore(m, HAND + i));
}

function setup(seed, layout, opts = {}) {
  R = mulberry32(seed * 3301 + 11);
  return B.setup(seed, layout, opts);
}
const inHand = sh => sh.items.filter(i => !i.concealed && !i.hidden && !i.paid && !i.onCounter);
const ITEM_CAMS = ['shelf', 'cooler', 'checkout'];
// Which cameras to render now. An arm that is up (reach, retract, put back, hold, conceal): the item camera with the most
// pixels on the hand and one other that sees it. An item carried while walking, or on the counter: one camera, less often.
function plan({ pAct = 0.7, pCarry = 0.25, pRandom = 0.06 } = {}) {
  const out = new Map(), cams = sim.vcams.filter(vc => ITEM_CAMS.includes(vc.kind));
  for (const vc of cams) vc.cam.updateMatrixWorld(true);
  const pick = a => a[Math.floor(R() * a.length)];
  for (const sh of sim.shoppers) {
    const up = !!sh.hand, held = inHand(sh).length > 0, counter = sh.items.some(i => i.onCounter && !i.paid);
    if (!up && !held && !counter) continue;
    if (R() > (up ? pAct : pCarry)) continue;
    const P = counter && !up ? sh.items.find(i => i.onCounter).mesh.position : sh.fig.handWorld(0);
    const see = cams.map(vc => [pixelsAt(vc, P, store.occluders), vc.name]).filter(x => x[0] >= 8).sort((a, b) => b[0] - a[0]);
    if (!see.length) continue;
    const why = up ? sh.state : counter ? 'counter' : 'carry';
    out.set(see[0][1], why);
    if (up && see.length > 1 && !out.has(pick(see.slice(1))[1])) out.set(pick(see.slice(1))[1], why);
  }
  if (R() < pRandom) { const c = pick(cams).name; if (!out.has(c)) out.set(c, 'random'); }
  return [...out].map(([cam, why]) => ({ cam, why }));
}

// synth.js capture (colour frame, every item box) plus the visible hands and people of this camera.
function capture(name, opts) {
  tagAll();
  const r = Y.capture(name, opts), vc = sim.vcams.find(v => v.name === name), [w, h] = vc.res;
  if (!idRT || idRT.width !== w || idRT.height !== h) { idRT?.dispose(); idRT = new THREE.WebGLRenderTarget(w, h, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }); }
  const hidden = [];
  scene.traverse(o => { if (o.isMesh && o.visible && [].concat(o.material).some(m => m.transparent)) { o.visible = false; hidden.push(o); } });
  scene.overrideMaterial = idMat; renderer.setRenderTarget(idRT); renderer.setClearColor(0x000000, 1); renderer.clear(); renderer.render(scene, vc.cam);
  scene.overrideMaterial = null; hidden.forEach(o => { o.visible = true; });
  const buf = new Uint8Array(w * h * 4); renderer.readRenderTargetPixels(idRT, 0, 0, w, h, buf); renderer.setRenderTarget(null);
  const box = new Map();
  for (let row = 0, i = 0; row < h; row++) for (let x = 0; x < w; x++, i += 4) {
    if (buf[i + 2] < 0xE0) continue;
    const id = buf[i] | (buf[i + 1] << 8) | (buf[i + 2] << 16);
    let b = box.get(id); if (!b) box.set(id, b = [x, row, x, row, 0]);
    if (x < b[0]) b[0] = x; if (x > b[2]) b[2] = x; if (row < b[1]) b[1] = row; if (row > b[3]) b[3] = row; b[4]++;
  }
  const C = vc.cam.getWorldPosition(new THREE.Vector3()), P = new THREE.Vector3(), hands = [], persons = [];
  for (const [id, b] of box) {
    const bbox = [b[0], h - 1 - b[3], b[2] + 1, h - b[1]];
    if (id >= PERSON) { const sh = sim.shoppers.find(s => s.id === id - PERSON); persons.push({ shopper: sh ? sh.name : 'clerk', bbox, vis_px: b[4], state: sh?.state ?? 'clerk' }); continue; }
    const n = (id - HAND) >> 1, arm = (id - HAND) & 1, sh = sim.shoppers.find(s => s.id === n);
    if (n && !sh) continue;
    const rec = { shopper: sh ? sh.name : 'clerk', arm, bbox, vis_px: b[4], state: sh?.state ?? 'clerk', reaching: !!sh && arm === 0 && !!sh.hand, holding: !!sh && arm === 0 && inHand(sh).length > 0 };
    if (sh) { sh.fig.handWorld(arm, P); rec.px_eff = +(CAN_W * vc.fx / C.distanceTo(P)).toFixed(2); }
    hands.push(rec);
  }
  return { ...r, hands, persons };
}

window.__breeHands = { setup, advance: B.advance, plan, capture, state: B.state, events: B.events, skus: Y.skus };
