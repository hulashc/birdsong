import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { initUpload } from './upload.js';
import { buildIndex, classify, distToSimilarity } from './knn.js';

// ── Renderers ──────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x000000, 0);           // transparent — CSS bg shows through

// Mount into #canvas-root so layout stays clean
const canvasRoot = document.getElementById('canvas-root') ?? document.body;
canvasRoot.appendChild(renderer.domElement);

const css2d = new CSS2DRenderer();
css2d.setSize(innerWidth, innerHeight);
css2d.domElement.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:5;';
document.body.appendChild(css2d.domElement);

// ── Scene / Camera ────────────────────────────────────────────────────────
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.001, 200);
camera.position.set(2.4, 1.8, 3.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping   = true;
controls.dampingFactor   = 0.06;
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.25;
controls.minDistance     = 0.5;
controls.maxDistance     = 12;

// ── Colour map (energy → heat colour) ────────────────────────────────────
function heatColor(e) {
  const t = Math.max(0, Math.min(1, e));
  const c = new THREE.Color();
  if (t < 0.25) {
    const s = t / 0.25;
    c.setRGB(0.22 + s * 0.48, 0.06 + s * 0.04, 0.02);
  } else if (t < 0.55) {
    const s = (t - 0.25) / 0.30;
    c.setRGB(0.70 + s * 0.20, 0.10 + s * 0.24, 0.02);
  } else if (t < 0.80) {
    const s = (t - 0.55) / 0.25;
    c.setRGB(0.90 + s * 0.05, 0.34 + s * 0.30, 0.02);
  } else {
    const s = (t - 0.80) / 0.20;
    c.setRGB(0.95 - s * 0.85, 0.64 - s * 0.60, 0.02 - s * 0.01);
  }
  return c;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ── Error display ─────────────────────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById('error-state');
  if (el) { el.querySelector('.error-msg').textContent = msg; el.style.display = 'flex'; }
  const ov = document.getElementById('overlay');
  if (ov) ov.style.display = 'none';
}

// ═══════════════════════════════════════════════════════════════════════════
//  AESTHETIC OVERHAUL — Axes, Grid, Labels
// ═══════════════════════════════════════════════════════════════════════════

// ── Colour palette for axes ───────────────────────────────────────────────
// Warmer, more muted values that sit comfortably in both light and dark modes
const AX_R = 0xc0230a;   // PC1 Timbre  — red
const AX_G = 0x1a7a40;   // PC2 Texture — green
const AX_B = 0x1a44a8;   // PC3 Spectral — blue

// ── Helper: dashed line via many short segments ───────────────────────────
// Three.js LineDashedMaterial needs computeLineDistances() and only works
// on Lines (not LineSegments for instanced use), so we fake dashes manually
// by emitting alternating filled / skipped segments.
function dashedLine(from, to, color, opacity, dashLen = 0.08, gapLen = 0.04) {
  const dir = new THREE.Vector3().subVectors(to, from);
  const totalLen = dir.length();
  dir.normalize();
  const pts = [];
  let cursor = 0;
  let drawing = true;
  while (cursor < totalLen) {
    const segLen = Math.min(drawing ? dashLen : gapLen, totalLen - cursor);
    if (drawing) {
      pts.push(
        from.clone().addScaledVector(dir, cursor),
        from.clone().addScaledVector(dir, cursor + segLen)
      );
    }
    cursor  += segLen;
    drawing  = !drawing;
  }
  if (!pts.length) return;
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  scene.add(new THREE.LineSegments(geo, mat));
}

// ── Helper: solid line ────────────────────────────────────────────────────
function solidLine(from, to, color, opacity) {
  const geo = new THREE.BufferGeometry().setFromPoints([from, to]);
  scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity })));
}

// ── Axes ──────────────────────────────────────────────────────────────────
// Solid centre portion (origin → tip), dashed negative half
const AXIS_LEN = 1.55;

// Positive halves — solid, fully opaque
solidLine(new THREE.Vector3(0,0,0), new THREE.Vector3(AXIS_LEN, 0, 0), AX_R, 0.80);
solidLine(new THREE.Vector3(0,0,0), new THREE.Vector3(0, AXIS_LEN, 0), AX_G, 0.80);
solidLine(new THREE.Vector3(0,0,0), new THREE.Vector3(0, 0, AXIS_LEN), AX_B, 0.80);

// Negative halves — dashed, dimmed
dashedLine(new THREE.Vector3(0,0,0), new THREE.Vector3(-AXIS_LEN, 0, 0), AX_R, 0.28);
dashedLine(new THREE.Vector3(0,0,0), new THREE.Vector3(0, -AXIS_LEN, 0), AX_G, 0.28);
dashedLine(new THREE.Vector3(0,0,0), new THREE.Vector3(0, 0, -AXIS_LEN), AX_B, 0.28);

// Arrow tips — small cones on the positive ends
function addArrowTip(pos, dir, color) {
  const geo = new THREE.ConeGeometry(0.022, 0.072, 8);
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.80 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(pos);
  // Rotate cone (default points +Y) to align with axis direction
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  scene.add(mesh);
}
addArrowTip(new THREE.Vector3(AXIS_LEN + 0.036, 0, 0), new THREE.Vector3(1, 0, 0), AX_R);
addArrowTip(new THREE.Vector3(0, AXIS_LEN + 0.036, 0), new THREE.Vector3(0, 1, 0), AX_G);
addArrowTip(new THREE.Vector3(0, 0, AXIS_LEN + 0.036), new THREE.Vector3(0, 0, 1), AX_B);

// ── Tick marks ────────────────────────────────────────────────────────────
// Larger ticks at round values, thinner style
const TICK_VALS  = [-1.2, -0.8, -0.4, 0.4, 0.8, 1.2];
const TICK_MAJOR = 0.05;  // half-length of a major tick

for (const v of TICK_VALS) {
  // X axis
  solidLine(
    new THREE.Vector3(v, -TICK_MAJOR, 0),
    new THREE.Vector3(v,  TICK_MAJOR, 0),
    AX_R, 0.35
  );
  // Y axis
  solidLine(
    new THREE.Vector3(-TICK_MAJOR, v, 0),
    new THREE.Vector3( TICK_MAJOR, v, 0),
    AX_G, 0.35
  );
  // Z axis
  solidLine(
    new THREE.Vector3(0, -TICK_MAJOR, v),
    new THREE.Vector3(0,  TICK_MAJOR, v),
    AX_B, 0.35
  );
}

// ── Axis labels (CSS2D) ───────────────────────────────────────────────────
// Redesigned: short human-readable name on top, full name below in muted text
function addAxisLabel(shortName, fullName, pos, hex) {
  const col = `#${hex.toString(16).padStart(6, '0')}`;
  const el  = document.createElement('div');
  el.style.cssText = [
    'display:flex',
    'flex-direction:column',
    'align-items:center',
    'pointer-events:none',
    'user-select:none',
    'line-height:1.25',
    'text-align:center',
  ].join(';');
  el.innerHTML = [
    `<span style="font-family:'Satoshi',system-ui,sans-serif;font-size:10px;font-weight:700;`,
    `letter-spacing:0.03em;color:${col};opacity:0.92;white-space:nowrap">${shortName}</span>`,
    `<span style="font-family:'Satoshi',system-ui,sans-serif;font-size:8px;font-weight:500;`,
    `letter-spacing:0.12em;text-transform:uppercase;color:${col};opacity:0.45;white-space:nowrap">${fullName}</span>`,
  ].join('');
  const obj = new CSS2DObject(el);
  obj.position.copy(pos);
  scene.add(obj);
}

addAxisLabel('Timbre',   'PC1', new THREE.Vector3(AXIS_LEN + 0.18, 0.04, 0),  AX_R);
addAxisLabel('Texture',  'PC2', new THREE.Vector3(0.04, AXIS_LEN + 0.18, 0),  AX_G);
addAxisLabel('Spectral', 'PC3', new THREE.Vector3(0.04, 0, AXIS_LEN + 0.18),  AX_B);

// ── Grid (XZ plane only — subtle, single plane) ──────────────────────────
// One clean XZ ground plane grid, tighter opacity, fewer lines
const GRID_STEPS = 6;
const GRID_RANGE = 1.4;
const gridMat = new THREE.LineBasicMaterial({ color: 0xb8b090, transparent: true, opacity: 0.09 });
for (let i = 0; i <= GRID_STEPS; i++) {
  const f = -GRID_RANGE + (2 * GRID_RANGE / GRID_STEPS) * i;
  const gx = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(f, 0, -GRID_RANGE),
    new THREE.Vector3(f, 0,  GRID_RANGE),
  ]);
  scene.add(new THREE.Line(gx, gridMat));
  const gz = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-GRID_RANGE, 0, f),
    new THREE.Vector3( GRID_RANGE, 0, f),
  ]);
  scene.add(new THREE.Line(gz, gridMat));
}

// ── Origin marker ─────────────────────────────────────────────────────────
// Small sphere at 0,0,0 so the convergence of axes is clear
const originGeo = new THREE.SphereGeometry(0.018, 12, 12);
const originMat = new THREE.MeshBasicMaterial({ color: 0x999070, transparent: true, opacity: 0.55 });
scene.add(new THREE.Mesh(originGeo, originMat));

// ═══════════════════════════════════════════════════════════════════════════
//  SLOT SYSTEM (unchanged logic, aesthetic tweaks only)
// ═══════════════════════════════════════════════════════════════════════════

const NODE_COUNT = 80;
const KNN_EDGES  = 3;
const TRAIL_HOPS = 20;

const _edgeCache = new WeakMap();
function buildEdgesMemo(nodePos, N, k) {
  if (_edgeCache.has(nodePos)) return _edgeCache.get(nodePos);
  const edges = [];
  for (let i = 0; i < N; i++) {
    const ax = nodePos[i*3], ay = nodePos[i*3+1], az = nodePos[i*3+2];
    const dists = [];
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = ax - nodePos[j*3], dy = ay - nodePos[j*3+1], dz = az - nodePos[j*3+2];
      dists.push([j, dx*dx + dy*dy + dz*dz]);
    }
    dists.sort((a, b) => a[1] - b[1]);
    for (let ki = 0; ki < Math.min(k, dists.length); ki++) {
      const j = dists[ki][0];
      if (j > i) edges.push([i, j]);
    }
  }
  _edgeCache.set(nodePos, edges);
  return edges;
}

function makeSlot() {
  return {
    manifold: null,
    audio: null, audioReady: false,
    nodes: null, nodeEnergy: null, nodeRawIdx: null,
    edges: null,
    edgeGeo: null, edgePos: null, edgeCol: null,
    cloud: null,
    labels: [],
    trail: [],
    trailGeo: null, trailPos: null, trailCol: null,
    halo: null, haloMat: null,
    objects: [], labelObjects: [],
    smoothIdx: 0,
    prevClockTime: -1,
    prevAni: -1,
    smoothScale: 1.0,
    smoothHaloOpacity: 0.0,
    smoothEnergy: 0.0,
  };
}

const primary   = makeSlot();
const secondary = makeSlot();

// ── Node geometry: rounded sphere instead of cube ─────────────────────────
// Spheres read more naturally as "data points" than cubes
const NODE_GEO = new THREE.SphereGeometry(0.012, 8, 8);
const HALO_GEO = new THREE.SphereGeometry(0.034, 16, 16);

function disposeSlot(slot) {
  for (const obj of slot.objects) {
    scene.remove(obj);
    obj.geometry?.dispose();
    if (obj.material) (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => m.dispose());
  }
  for (const obj of slot.labelObjects) scene.remove(obj);
  slot.objects = []; slot.labelObjects = []; slot.labels = [];
  slot.edgeGeo?.dispose(); slot.trailGeo?.dispose();
  slot.edgeGeo = null; slot.trailGeo = null;
  slot.manifold = null; slot.cloud = null; slot.halo = null;
  slot.trail = [];
  slot.smoothIdx = 0;
  slot.smoothScale = 1.0;
  slot.smoothHaloOpacity = 0.0;
  slot.smoothEnergy = 0.0;
  slot.prevClockTime = -1;
  slot.prevAni = -1;
}

function clearTrail(slot) {
  slot.trail = [];
  if (slot.trailGeo) {
    slot.trailGeo.setDrawRange(0, 0);
    slot.trailGeo.attributes.position.needsUpdate = true;
  }
}

const _pauseDummy = new THREE.Object3D();
const _pauseColor = new THREE.Color();
function resetNodeScales(slot) {
  if (!slot.cloud || !slot.nodes || !slot.nodeEnergy) return;
  for (let i = 0; i < NODE_COUNT; i++) {
    _pauseDummy.position.set(slot.nodes[i*3], slot.nodes[i*3+1], slot.nodes[i*3+2]);
    _pauseDummy.scale.set(1, 1, 1);
    _pauseDummy.updateMatrix();
    slot.cloud.setMatrixAt(i, _pauseDummy.matrix);
    _pauseColor.copy(heatColor(slot.nodeEnergy[i])).multiplyScalar(0.90);
    slot.cloud.setColorAt(i, _pauseColor);
  }
  slot.cloud.instanceMatrix.needsUpdate = true;
  slot.cloud.instanceColor.needsUpdate  = true;
  slot.prevAni = -1;
  slot.smoothScale = 1.0;
  slot.smoothHaloOpacity = 0.0;
  slot.smoothEnergy = 0.0;
}

function buildSlot(slot, manifold) {
  disposeSlot(slot);

  const rawPts  = manifold.xyz.map(([x,y,z]) => new THREE.Vector3(x,y,z));
  const rawE    = manifold.energy || [];
  const times   = manifold.t;
  const dur     = manifold.duration_s ?? times[times.length - 1] ?? 10;
  const spectral = manifold.spectral_centroid || [];

  const N    = NODE_COUNT;
  const step = (rawPts.length - 1) / (N - 1);
  const nodePos = new Float32Array(N * 3);
  const nodeE   = new Float32Array(N);
  const nodeRaw = new Int32Array(N);
  const nodeSC  = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    const ri = Math.round(i * step);
    const p  = rawPts[ri];
    nodePos[i*3]   = p.x;
    nodePos[i*3+1] = p.y;
    nodePos[i*3+2] = p.z;
    nodeE[i]   = rawE[ri] ?? 0;
    nodeRaw[i] = ri;
    nodeSC[i]  = spectral[ri] ?? 0;
  }

  slot.nodes      = nodePos;
  slot.nodeEnergy = nodeE;
  slot.nodeRawIdx = nodeRaw;

  const edges = buildEdgesMemo(nodePos, N, KNN_EDGES);
  slot.edges  = edges;

  const maxEdgePts = edges.length * 2;
  slot.edgePos = new Float32Array(maxEdgePts * 3);
  slot.edgeCol = new Float32Array(maxEdgePts * 3);
  slot.edgeGeo = new THREE.BufferGeometry();
  slot.edgeGeo.setAttribute('position', new THREE.BufferAttribute(slot.edgePos, 3));
  slot.edgeGeo.setAttribute('color',    new THREE.BufferAttribute(slot.edgeCol, 3));
  slot.edgeGeo.setDrawRange(0, maxEdgePts);

  for (let e = 0; e < edges.length; e++) {
    const [i, j] = edges[e];
    slot.edgePos[e*6+0] = nodePos[i*3];   slot.edgePos[e*6+1] = nodePos[i*3+1]; slot.edgePos[e*6+2] = nodePos[i*3+2];
    slot.edgePos[e*6+3] = nodePos[j*3];   slot.edgePos[e*6+4] = nodePos[j*3+1]; slot.edgePos[e*6+5] = nodePos[j*3+2];
    const e0 = Math.max(nodeE[i], nodeE[j]);
    const dc = heatColor(e0).multiplyScalar(0.85);
    for (let k = 0; k < 2; k++) {
      slot.edgeCol[(e*2+k)*3]   = dc.r;
      slot.edgeCol[(e*2+k)*3+1] = dc.g;
      slot.edgeCol[(e*2+k)*3+2] = dc.b;
    }
  }
  slot.edgeGeo.attributes.position.needsUpdate = true;
  slot.edgeGeo.attributes.color.needsUpdate    = true;

  const edgeLine = new THREE.LineSegments(slot.edgeGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0 }));
  scene.add(edgeLine);
  slot.objects.push(edgeLine);

  const cloudMat = new THREE.MeshBasicMaterial({ vertexColors: true });
  const cloud    = new THREE.InstancedMesh(NODE_GEO, cloudMat, N);
  cloud.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  cloud.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
  const dummy = new THREE.Object3D();
  const nc    = new THREE.Color();
  for (let i = 0; i < N; i++) {
    dummy.position.set(nodePos[i*3], nodePos[i*3+1], nodePos[i*3+2]);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    cloud.setMatrixAt(i, dummy.matrix);
    nc.copy(heatColor(nodeE[i])).multiplyScalar(0.90);
    cloud.setColorAt(i, nc);
  }
  cloud.instanceMatrix.needsUpdate = true;
  cloud.instanceColor.needsUpdate  = true;
  scene.add(cloud);
  slot.cloud = cloud;
  slot.objects.push(cloud);

  // ── Node labels: redesigned ───────────────────────────────────────────
  // Show every 10th node (less clutter), human-readable format:
  //   Energy value as a short percentage bar (▪▪▪▪▪░░░░░) + energy %
  //   Spectral centroid as "X.Xk Hz"
  const LABEL_EVERY = 10;
  for (let i = 0; i < N; i += LABEL_EVERY) {
    const e  = nodeE[i];
    const sc = nodeSC[i];

    // Progress bar: 5 filled + 5 empty blocks representing 0–100% energy
    const filled = Math.round(e * 5);
    const bar = '▪'.repeat(filled) + '░'.repeat(5 - filled);
    const energyPct = Math.round(e * 100);

    const el = document.createElement('div');
    el.className = 'node-label';

    // Energy line: bar + percentage
    const eLine = document.createElement('span');
    eLine.className = 'nl-e';
    // Colour shifts with energy: low=blue tint, high=amber tint
    const r = Math.round(40  + e * 180);
    const g = Math.round(40  + e * 80);
    const b = Math.round(160 - e * 140);
    eLine.style.color = `rgb(${r},${g},${b})`;
    eLine.textContent = `${bar} ${energyPct}%`;

    // Spectral centroid line (only if present)
    const scLine = document.createElement('span');
    scLine.className = 'nl-sc';
    if (sc > 0) scLine.textContent = `${(sc / 1000).toFixed(1)}k Hz`;

    el.appendChild(eLine);
    if (sc > 0) el.appendChild(scLine);

    const css = new CSS2DObject(el);
    css.position.set(nodePos[i*3], nodePos[i*3+1] + 0.028, nodePos[i*3+2]);
    scene.add(css);
    slot.labels.push({ css, el, nodeIdx: i });
    slot.labelObjects.push(css);
  }

  slot.haloMat = new THREE.MeshBasicMaterial({ color: 0xddccaa, transparent: true, opacity: 0, wireframe: true });
  slot.halo    = new THREE.Mesh(HALO_GEO, slot.haloMat);
  scene.add(slot.halo);
  slot.objects.push(slot.halo);

  // ── Trail geometry ────────────────────────────────────────────────────
  const trailPts = TRAIL_HOPS * 2;
  slot.trailPos  = new Float32Array(trailPts * 3);
  slot.trailCol  = new Float32Array(trailPts * 3);
  slot.trailGeo  = new THREE.BufferGeometry();
  slot.trailGeo.setAttribute('position', new THREE.BufferAttribute(slot.trailPos, 3));
  slot.trailGeo.setAttribute('color',    new THREE.BufferAttribute(slot.trailCol, 3));
  slot.trailGeo.setDrawRange(0, 0);
  const trailLine = new THREE.LineSegments(slot.trailGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1.0 }));
  scene.add(trailLine);
  slot.objects.push(trailLine);

  slot.manifold = { times, energy: rawE, rawPts, duration_s: dur, N };
  slot.smoothIdx = 0;
  slot.smoothScale = 1.0;
  slot.smoothHaloOpacity = 0.0;
  slot.smoothEnergy = 0.0;
  slot.prevClockTime = -1;
  slot.prevAni = -1;
}

// ── Clock ─────────────────────────────────────────────────────────────────
let clockTime = 0;
let lastRAF   = null;
let playing   = false;

const MAX_DELTA = 1 / 30;

function currentTime(slot) {
  if (slot.audioReady && slot.audio && !slot.audio.paused) return slot.audio.currentTime;
  return clockTime;
}

function fracNodeForTime(slot, ct) {
  const { times, duration_s } = slot.manifold;
  const dur = duration_s ?? times[times.length - 1] ?? 10;
  const tt  = dur > 0 ? ct % dur : ct;

  if (slot.prevClockTime >= 0) {
    const prevTT = dur > 0 ? slot.prevClockTime % dur : slot.prevClockTime;
    if (prevTT - tt > dur * 0.4) slot.smoothIdx = 0;
  }
  slot.prevClockTime = ct;

  let lo = 0, hi = times.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] < tt) lo = mid + 1; else hi = mid; }
  const rawIdx   = clamp(lo, 0, times.length - 1);
  const nodeStep = (times.length - 1) / (NODE_COUNT - 1);
  return clamp(rawIdx / nodeStep, 0, NODE_COUNT - 1);
}

// ── Per-frame update ──────────────────────────────────────────────────────
const _d = new THREE.Object3D();
const _c = new THREE.Color();

function lerpFactor(halfLifeSeconds, delta) {
  return 1.0 - Math.pow(0.5, delta / halfLifeSeconds);
}

function tickSlot(slot, delta) {
  if (!slot.manifold || !slot.cloud) return;
  const { nodes, nodeEnergy, edges } = slot;
  const N  = NODE_COUNT;
  const ct = currentTime(slot);

  const target = fracNodeForTime(slot, ct);

  const alpha = lerpFactor(0.040, delta);
  const prev  = slot.smoothIdx;

  if ((target - prev) < -(NODE_COUNT * 0.4)) {
    slot.smoothIdx = target;
  } else {
    slot.smoothIdx = prev + alpha * (target - prev);
  }

  const fracIdx = slot.smoothIdx;
  const ani     = clamp(Math.round(fracIdx), 0, N - 1);

  const nodeChanged = ani !== slot.prevAni;
  slot.prevAni = ani;

  const ae = nodeEnergy[ani];

  slot.smoothEnergy = slot.smoothEnergy + lerpFactor(0.060, delta) * (ae - slot.smoothEnergy);
  const se = slot.smoothEnergy;

  if (nodeChanged) {
    const neighbours = new Set();
    for (const [i, j] of edges) {
      if (i === ani) neighbours.add(j);
      if (j === ani) neighbours.add(i);
    }

    for (let i = 0; i < N; i++) {
      const e  = nodeEnergy[i];
      const isActive    = i === ani;
      const isNeighbour = neighbours.has(i);
      const bright = isActive    ? 1.6 + se * 0.8
                   : isNeighbour ? 1.1 + e  * 0.5
                   :               0.90 + e * 0.15;
      const scale  = isActive    ? 2.8 + se * 2.0
                   : isNeighbour ? 1.5 + e  * 0.6
                   :               0.85 + e  * 0.4;
      _d.position.set(nodes[i*3], nodes[i*3+1], nodes[i*3+2]);
      _d.scale.set(scale, scale, scale);
      _d.updateMatrix();
      slot.cloud.setMatrixAt(i, _d.matrix);
      _c.copy(heatColor(e)).multiplyScalar(bright);
      slot.cloud.setColorAt(i, _c);
    }
    slot.cloud.instanceMatrix.needsUpdate = true;
    slot.cloud.instanceColor.needsUpdate  = true;

    const { edgeCol, edgeGeo } = slot;
    for (let ei = 0; ei < edges.length; ei++) {
      const [i, j]   = edges[ei];
      const isActive = i === ani || j === ani;
      const bright   = isActive ? 1.3 + se * 0.6 : 0.85;
      const e        = Math.max(nodeEnergy[i], nodeEnergy[j]);
      const c2       = heatColor(e).multiplyScalar(bright);
      for (let k = 0; k < 2; k++) {
        edgeCol[(ei*2+k)*3]   = c2.r;
        edgeCol[(ei*2+k)*3+1] = c2.g;
        edgeCol[(ei*2+k)*3+2] = c2.b;
      }
    }
    edgeGeo.attributes.color.needsUpdate = true;

    // Labels: show only labels near active node, fade others
    for (const { el, nodeIdx } of slot.labels) {
      const dist   = Math.abs(nodeIdx - ani) / N;
      const active = dist < 0.08;   // slightly wider window than before
      el.style.opacity = active ? '1.0' : '0.18';  // others nearly invisible
      el.style.transform = active ? 'scale(1.08)' : 'scale(1)';
    }
  }

  // Smooth halo position (continuous sub-node interpolation)
  const loNode = Math.floor(fracIdx);
  const hiNode = Math.min(loNode + 1, N - 1);
  const t      = fracIdx - loNode;
  const hx = nodes[loNode*3]   + t * (nodes[hiNode*3]   - nodes[loNode*3]);
  const hy = nodes[loNode*3+1] + t * (nodes[hiNode*3+1] - nodes[loNode*3+1]);
  const hz = nodes[loNode*3+2] + t * (nodes[hiNode*3+2] - nodes[loNode*3+2]);
  slot.halo.position.set(hx, hy, hz);

  const targetScale   = 1.0 + se * 1.4;
  const targetOpacity = 0.28 + se * 0.42;
  slot.smoothScale       = slot.smoothScale       + lerpFactor(0.050, delta) * (targetScale   - slot.smoothScale);
  slot.smoothHaloOpacity = slot.smoothHaloOpacity + lerpFactor(0.050, delta) * (targetOpacity - slot.smoothHaloOpacity);
  slot.halo.scale.setScalar(slot.smoothScale);
  slot.haloMat.opacity = slot.smoothHaloOpacity;

  // ── Trail: wider line feel via brighter colour gradient ───────────────
  if (slot.trail[slot.trail.length - 1] !== ani) slot.trail.push(ani);
  if (slot.trail.length > TRAIL_HOPS + 1) slot.trail.shift();

  const tLen = slot.trail.length;
  const segs = Math.max(0, tLen - 1);
  const { trailPos, trailCol, trailGeo } = slot;
  for (let s = 0; s < segs; s++) {
    const a  = slot.trail[s], b = slot.trail[s + 1];
    const ea = nodeEnergy[a], eb = nodeEnergy[b];
    const f  = (s + 1) / segs;        // 0 = oldest, 1 = newest
    // Newer segments much brighter; older fade to near-zero
    const bright = Math.pow(f, 0.6) * 2.6;
    const alpha  = Math.pow(f, 1.2);  // additional fade encoded in colour brightness
    trailPos[s*6+0] = nodes[a*3];   trailPos[s*6+1] = nodes[a*3+1]; trailPos[s*6+2] = nodes[a*3+2];
    trailPos[s*6+3] = nodes[b*3];   trailPos[s*6+4] = nodes[b*3+1]; trailPos[s*6+5] = nodes[b*3+2];
    _c.copy(heatColor(ea)).multiplyScalar(bright * alpha);
    trailCol[s*6+0] = _c.r; trailCol[s*6+1] = _c.g; trailCol[s*6+2] = _c.b;
    _c.copy(heatColor(eb)).multiplyScalar(bright);
    trailCol[s*6+3] = _c.r; trailCol[s*6+4] = _c.g; trailCol[s*6+5] = _c.b;
  }
  trailGeo.setDrawRange(0, segs * 2);
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate    = true;
}

// ── Audio helpers ─────────────────────────────────────────────────────────
function loadAudioForSlot(slot, key) {
  if (slot.audio) { slot.audio.pause(); slot.audio.src = ''; }
  slot.audioReady = false;
  const audio = new Audio();
  audio.loop = true;
  slot.audio = audio;
  let idx = 0;
  const cands = [`./birds/${key}.ogg`, `./birds/${key}.mp3`];
  function tryNext() {
    if (idx >= cands.length) return;
    audio.src = cands[idx++]; audio.load();
  }
  audio.addEventListener('canplaythrough', () => {
    slot.audioReady = true;
    if (playing) audio.play().catch(() => {});
  }, { once: true });
  audio.addEventListener('error', tryNext);
  tryNext();
}

// ── Load data ─────────────────────────────────────────────────────────────
let allData = null;
try {
  const res = await fetch('./birdsong_data.json');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  allData = await res.json();
} catch (err) {
  showError(`Failed to load data: ${err.message}`);
  console.error('[birdsong] Data load failed:', err);
}

if (!allData) throw new DOMException('Data unavailable', 'AbortError');

const isSingle    = Array.isArray(allData?.t) && Array.isArray(allData?.xyz);
const speciesMap  = isSingle ? { [(allData.species || 'birdsong')]: allData } : allData;
const speciesKeys = Object.keys(speciesMap);

if (!speciesKeys.length) {
  showError('birdsong_data.json is empty or malformed.');
  throw new DOMException('Empty data', 'AbortError');
}

const knnIndex = buildIndex(speciesMap);

// ── Dropdown ──────────────────────────────────────────────────────────────
const select = document.getElementById('speciesSelect');
select.innerHTML = '';
for (const key of speciesKeys) {
  const opt = document.createElement('option');
  opt.value = key;
  opt.textContent = key.replace(/_/g, ' ');
  select.appendChild(opt);
}

function setTitleLabels(name) {
  const p = name.replace(/_/g, ' ');
  const titleEl = document.getElementById('titleSpecies');
  if (titleEl) titleEl.textContent = p;
  document.title = `${p} · Birdsong Acoustic Manifold`;
}

function updateManifoldLegend() {
  const el = document.getElementById('manifold-legend');
  el.classList.toggle('visible', !!secondary.manifold);
  const lp = document.getElementById('legend-primary');
  if (lp) lp.textContent = select.value.replace(/_/g, ' ');
}

function showKnnResults(results) {
  const panel = document.getElementById('knn-results');
  const list  = document.getElementById('knn-list');
  if (!panel || !list) return;
  list.innerHTML = '';
  results.forEach(({ species, distance, rank }) => {
    const pct = distToSimilarity(distance);
    const row = document.createElement('div'); row.className = 'knn-row';
    row.innerHTML = `<span class="knn-rank">${rank}</span><span class="knn-species">${species.replace(/_/g,' ')}</span><span class="knn-pct">${pct}%</span>`;
    const bar = document.createElement('div'); bar.className = 'knn-bar';
    bar.innerHTML = `<div class="knn-bar-fill" style="width:${pct}%"></div>`;
    list.appendChild(row); list.appendChild(bar);
  });
  panel.style.display = 'block';
}

function hideKnnResults() {
  const panel = document.getElementById('knn-results');
  if (panel) panel.style.display = 'none';
}

function loadSpecies(key) {
  const mRaw = speciesMap[key];
  if (!mRaw) return;
  const m = { ...mRaw, duration_s: mRaw.duration_s ?? mRaw.t[mRaw.t.length-1] ?? 10 };
  buildSlot(primary, m);
  loadAudioForSlot(primary, key);
  setTitleLabels(key);
  updateManifoldLegend();
}

loadSpecies(speciesKeys[0]);
select.value = speciesKeys[0];
select.addEventListener('change', () => {
  if (primary.audio) primary.audio.pause();
  loadSpecies(select.value);
});

// ── Overlay / Pause ───────────────────────────────────────────────────────
const overlay   = document.getElementById('overlay');
const playBtn   = document.getElementById('playBtn');

playBtn.style.display = 'block';
playBtn.textContent   = 'Play';

function startPlayback() {
  playing = true;
  playBtn.textContent = 'Pause';
  playBtn.classList.add('playing');
  lastRAF = null;
  primary.audio?.play().catch(() => {});
  if (secondary.audio) secondary.audio.play().catch(() => {});
}

function pausePlayback() {
  playing = false;
  playBtn.textContent = 'Play';
  playBtn.classList.remove('playing');
  lastRAF = null;
  primary.audio?.pause();
  secondary.audio?.pause();
  clearTrail(primary);
  clearTrail(secondary);
  if (primary.haloMat)   primary.haloMat.opacity = 0;
  if (secondary.haloMat) secondary.haloMat.opacity = 0;
  resetNodeScales(primary);
  resetNodeScales(secondary);
}

overlay.addEventListener('click', () => {
  overlay.classList.add('hidden');
  setTimeout(() => { overlay.style.display = 'none'; }, 900);
  startPlayback();
}, { once: true });

overlay.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') overlay.click();
});

playBtn.addEventListener('click', e => {
  e.stopPropagation();
  if (playing) pausePlayback(); else startPlayback();
});

// ── Upload ────────────────────────────────────────────────────────────────
initUpload({
  onManifold(m, file) {
    buildSlot(secondary, m);
    if (secondary.audio) { secondary.audio.pause(); secondary.audio.src = ''; }
    secondary.audioReady = false;
    const audio = new Audio(URL.createObjectURL(file));
    audio.loop = true; secondary.audio = audio;
    audio.addEventListener('canplaythrough', () => {
      secondary.audioReady = true;
      if (playing) audio.play().catch(() => {});
    }, { once: true });
    const mb = document.getElementById('mode-badge');
    if (mb) mb.textContent = m.species.replace(/_/g, ' ');
    const ls = document.getElementById('legend-secondary');
    if (ls) ls.textContent = m.species.replace(/_/g, ' ');
    updateManifoldLegend();
    showKnnResults(classify(m, knnIndex, 3));
  },
  onError(err) {
    console.error('Upload error:', err);
    const status = document.getElementById('upload-status');
    if (status) status.textContent = 'Error: ' + (err?.message || String(err));
    hideKnnResults();
  },
  onProgress(pct) {
    const bar = document.getElementById('upload-progress-bar');
    const bg  = document.getElementById('upload-progress-bg');
    if (bar) { bar.style.opacity = '1'; bar.style.width = pct + '%'; }
    if (bg)  bg.setAttribute('aria-valuenow', pct);
    if (pct >= 100) setTimeout(() => { if (bar) bar.style.opacity = '0'; }, 800);
  },
});

// ── Animation loop ────────────────────────────────────────────────────────
function animate(ts) {
  requestAnimationFrame(animate);
  controls.update();

  if (playing) {
    let delta = 0;
    if (lastRAF !== null) {
      delta = Math.min((ts - lastRAF) / 1000, MAX_DELTA);
      clockTime += delta;
    }
    lastRAF = ts;
    if (delta > 0) {
      tickSlot(primary, delta);
      if (secondary.manifold) tickSlot(secondary, delta);
    }
  }
  renderer.render(scene, camera);
  css2d.render(scene, camera);
}

animate(0);

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  css2d.setSize(innerWidth, innerHeight);
});
