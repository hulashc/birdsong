import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { initUpload } from './upload.js';
import { buildIndex, classify, distToSimilarity } from './knn.js';

// ── Renderers ──────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0xf4f1eb, 1);
document.body.appendChild(renderer.domElement);

const css2d = new CSS2DRenderer();
css2d.setSize(innerWidth, innerHeight);
css2d.domElement.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:5;';
document.body.appendChild(css2d.domElement);

// ── Scene / Camera ───────────────────────────────────
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

// ── Colour map (light theme) ─────────────────────────────
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

// ── Error display ──────────────────────────────────────
function showError(msg) {
  const el = document.getElementById('error-state');
  if (el) { el.querySelector('.error-msg').textContent = msg; el.style.display = 'flex'; }
  const ov = document.getElementById('overlay');
  if (ov) ov.style.display = 'none';
}

// ── Axes ────────────────────────────────────────────────
function addAxis(a, b, hex, op = 0.70) {
  const g = new THREE.BufferGeometry().setFromPoints([a, b]);
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: op })));
}

addAxis(new THREE.Vector3(-1.6,0,0), new THREE.Vector3(1.6,0,0), 0xc0230a, 0.75);
addAxis(new THREE.Vector3(0,-1.6,0), new THREE.Vector3(0,1.6,0), 0x1a7a40, 0.75);
addAxis(new THREE.Vector3(0,0,-1.6), new THREE.Vector3(0,0,1.6), 0x1a44a8, 0.75);

const TICK_STEP = 0.4;
const TICK_HALF = 0.04;
for (let v = -1.6; v <= 1.6; v += TICK_STEP) {
  if (Math.abs(v) < 0.01) continue;
  const xg = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(v, -TICK_HALF, 0), new THREE.Vector3(v, TICK_HALF, 0)
  ]);
  scene.add(new THREE.Line(xg, new THREE.LineBasicMaterial({ color: 0xc0230a, transparent: true, opacity: 0.45 })));
  const yg = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-TICK_HALF, v, 0), new THREE.Vector3(TICK_HALF, v, 0)
  ]);
  scene.add(new THREE.Line(yg, new THREE.LineBasicMaterial({ color: 0x1a7a40, transparent: true, opacity: 0.45 })));
  const zg = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, -TICK_HALF, v), new THREE.Vector3(0, TICK_HALF, v)
  ]);
  scene.add(new THREE.Line(zg, new THREE.LineBasicMaterial({ color: 0x1a44a8, transparent: true, opacity: 0.45 })));
}

function addAxisLabel(text, pos, col) {
  const el = document.createElement('div');
  el.textContent = text;
  el.style.cssText = `color:${col};font-size:8px;letter-spacing:.3em;font-weight:700;text-transform:uppercase;font-family:'Courier New',monospace;opacity:0.75;padding:1px 4px;`;
  const obj = new CSS2DObject(el);
  obj.position.copy(pos);
  scene.add(obj);
}
addAxisLabel('PC1 · Timbre',   new THREE.Vector3(1.68, 0.06, 0),  '#c0230a');
addAxisLabel('PC2 · Texture',  new THREE.Vector3(0.06, 1.68, 0),  '#1a7a40');
addAxisLabel('PC3 · Spectral', new THREE.Vector3(0.06, 0,   1.68),'#1a44a8');

const GRID_STEPS = 8;
const GRID_RANGE = 1.6;
const gridMat = new THREE.LineBasicMaterial({ color: 0x9e9880, transparent: true, opacity: 0.13 });
for (let i = 0; i <= GRID_STEPS; i++) {
  const f = -GRID_RANGE + (2 * GRID_RANGE / GRID_STEPS) * i;
  const gx = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(f, 0, -GRID_RANGE), new THREE.Vector3(f, 0, GRID_RANGE)
  ]);
  scene.add(new THREE.Line(gx, gridMat));
  const gz = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-GRID_RANGE, 0, f), new THREE.Vector3(GRID_RANGE, 0, f)
  ]);
  scene.add(new THREE.Line(gz, gridMat));
}

// ── Slot ──────────────────────────────────────────────────
const NODE_COUNT = 80;
const KNN_EDGES  = 3;
const TRAIL_HOPS = 12;

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
  };
}

const primary   = makeSlot();
const secondary = makeSlot();

const NODE_GEO = new THREE.BoxGeometry(0.016, 0.016, 0.016);
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
  slot.prevClockTime = -1;
}

// ── FIX: clear trail geometry draw range to 0 (hides frozen trail lines on pause)
function clearTrail(slot) {
  slot.trail = [];
  if (slot.trailGeo) {
    slot.trailGeo.setDrawRange(0, 0);
    slot.trailGeo.attributes.position.needsUpdate = true;
  }
}

function buildEdges(positions, N, k) {
  const edges = [];
  for (let i = 0; i < N; i++) {
    const ax = positions[i*3], ay = positions[i*3+1], az = positions[i*3+2];
    const dists = [];
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = ax - positions[j*3], dy = ay - positions[j*3+1], dz = az - positions[j*3+2];
      dists.push([j, dx*dx + dy*dy + dz*dz]);
    }
    dists.sort((a, b) => a[1] - b[1]);
    for (let ki = 0; ki < Math.min(k, dists.length); ki++) {
      const j = dists[ki][0];
      if (j > i) edges.push([i, j]);
    }
  }
  return edges;
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

  const edges = buildEdges(nodePos, N, KNN_EDGES);
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
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1 }));
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

  const LABEL_EVERY = 8;
  for (let i = 0; i < N; i += LABEL_EVERY) {
    const e  = nodeE[i];
    const sc = nodeSC[i];
    const el = document.createElement('div');
    el.className = 'node-label';
    el.innerHTML =
      `<span class="nl-e">${e.toFixed(4)}</span>` +
      `<span class="nl-sc">${sc > 0 ? (sc / 1000).toFixed(2) + 'k' : ''}</span>`;
    const css = new CSS2DObject(el);
    css.position.set(nodePos[i*3], nodePos[i*3+1] + 0.025, nodePos[i*3+2]);
    scene.add(css);
    slot.labels.push({ css, el, nodeIdx: i });
    slot.labelObjects.push(css);
  }

  slot.haloMat = new THREE.MeshBasicMaterial({ color: 0x1a1710, transparent: true, opacity: 0, wireframe: true });
  slot.halo    = new THREE.Mesh(HALO_GEO, slot.haloMat);
  scene.add(slot.halo);
  slot.objects.push(slot.halo);

  const trailPts = TRAIL_HOPS * 2;
  slot.trailPos  = new Float32Array(trailPts * 3);
  slot.trailCol  = new Float32Array(trailPts * 3);
  slot.trailGeo  = new THREE.BufferGeometry();
  slot.trailGeo.setAttribute('position', new THREE.BufferAttribute(slot.trailPos, 3));
  slot.trailGeo.setAttribute('color',    new THREE.BufferAttribute(slot.trailCol, 3));
  // FIX: start with draw range 0 — trail is invisible until playback advances
  slot.trailGeo.setDrawRange(0, 0);
  const trailLine = new THREE.LineSegments(slot.trailGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1.0 }));
  scene.add(trailLine);
  slot.objects.push(trailLine);

  slot.manifold = { times, energy: rawE, rawPts, duration_s: dur, N };
  slot.smoothIdx = 0;
  slot.prevClockTime = -1;
}

// ── Clock ────────────────────────────────────────────────
let clockTime = 0;
let lastRAF   = null;
let playing   = false;

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
    if (prevTT - tt > dur * 0.4) {
      slot.smoothIdx = 0;
    }
  }
  slot.prevClockTime = ct;

  let lo = 0, hi = times.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] < tt) lo = mid + 1; else hi = mid; }
  const rawIdx   = clamp(lo, 0, times.length - 1);
  const nodeStep = (times.length - 1) / (NODE_COUNT - 1);
  return clamp(rawIdx / nodeStep, 0, NODE_COUNT - 1);
}

// ── Per-frame update ─────────────────────────────────────────
const _d = new THREE.Object3D();
const _c = new THREE.Color();

function tickSlot(slot, delta) {
  if (!slot.manifold || !slot.cloud) return;
  const { nodes, nodeEnergy, edges } = slot;
  const N  = NODE_COUNT;
  const ct = currentTime(slot);

  const target = fracNodeForTime(slot, ct);

  // FIX: clamp lerp so smoothIdx can never overshoot target — eliminates inter-node jitter.
  // Use delta-scaled alpha so motion speed is frame-rate independent.
  const alpha = clamp(1.0 - Math.pow(0.04, delta), 0, 1);
  const prev  = slot.smoothIdx;
  slot.smoothIdx = prev + alpha * (target - prev);
  // Hard-clamp: if we've crossed the target integer, snap to it to prevent oscillation
  if ((target - prev) * (target - slot.smoothIdx) < 0) slot.smoothIdx = target;

  const ani = clamp(Math.round(slot.smoothIdx), 0, N - 1);
  const ae  = nodeEnergy[ani];

  const neighbours = new Set();
  for (const [i, j] of edges) {
    if (i === ani) neighbours.add(j);
    if (j === ani) neighbours.add(i);
  }

  for (let i = 0; i < N; i++) {
    const e  = nodeEnergy[i];
    const isActive    = i === ani;
    const isNeighbour = neighbours.has(i);
    const bright = isActive    ? 1.6 + ae * 0.8
                 : isNeighbour ? 1.1 + e  * 0.5
                 :               0.90 + e * 0.15;
    const scale  = isActive    ? 2.8 + ae * 2.0
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
    const [i, j]    = edges[ei];
    const isActive  = i === ani || j === ani;
    const bright    = isActive ? 1.3 + ae * 0.6 : 0.85;
    const e         = Math.max(nodeEnergy[i], nodeEnergy[j]);
    const c2        = heatColor(e);
    c2.multiplyScalar(bright);
    for (let k = 0; k < 2; k++) {
      edgeCol[(ei*2+k)*3]   = c2.r;
      edgeCol[(ei*2+k)*3+1] = c2.g;
      edgeCol[(ei*2+k)*3+2] = c2.b;
    }
  }
  edgeGeo.attributes.color.needsUpdate = true;

  slot.halo.position.set(nodes[ani*3], nodes[ani*3+1], nodes[ani*3+2]);
  const hs = 1.0 + ae * 1.4;
  slot.halo.scale.set(hs, hs, hs);
  slot.haloMat.opacity = 0.28 + ae * 0.42;

  for (const { el, nodeIdx } of slot.labels) {
    const dist   = Math.abs(nodeIdx - ani) / N;
    const active = dist < 0.06;
    el.style.opacity = active ? '1.0' : '0.32';
    const hc = heatColor(nodeEnergy[nodeIdx]);
    el.style.color = active
      ? `rgb(${Math.round(hc.r*180)},${Math.round(hc.g*80)},${Math.round(hc.b*20)})`
      : '#6a6050';
  }

  if (slot.trail[slot.trail.length - 1] !== ani) slot.trail.push(ani);
  if (slot.trail.length > TRAIL_HOPS + 1) slot.trail.shift();

  const tLen = slot.trail.length;
  const segs = Math.max(0, tLen - 1);
  const { trailPos, trailCol, trailGeo } = slot;
  for (let s = 0; s < segs; s++) {
    const a  = slot.trail[s], b = slot.trail[s + 1];
    const ea = nodeEnergy[a], eb = nodeEnergy[b];
    const f  = (s + 1) / segs;
    const bright = Math.pow(f, 1.0) * 1.8;
    trailPos[s*6+0] = nodes[a*3];   trailPos[s*6+1] = nodes[a*3+1]; trailPos[s*6+2] = nodes[a*3+2];
    trailPos[s*6+3] = nodes[b*3];   trailPos[s*6+4] = nodes[b*3+1]; trailPos[s*6+5] = nodes[b*3+2];
    _c.copy(heatColor(ea)).multiplyScalar(bright * (1 - f * 0.3));
    trailCol[s*6+0] = _c.r; trailCol[s*6+1] = _c.g; trailCol[s*6+2] = _c.b;
    _c.copy(heatColor(eb)).multiplyScalar(bright);
    trailCol[s*6+3] = _c.r; trailCol[s*6+4] = _c.g; trailCol[s*6+5] = _c.b;
  }
  trailGeo.setDrawRange(0, segs * 2);
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate    = true;
}

// ── Audio helpers ──────────────────────────────────────────
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

// ── Load data ──────────────────────────────────────────────
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
  console.error('[birdsong] Empty species map');
  throw new DOMException('Empty data', 'AbortError');
}

const knnIndex = buildIndex(speciesMap);

// ── Dropdown ───────────────────────────────────────────────
const select = document.getElementById('speciesSelect');
select.innerHTML = '';
for (const key of speciesKeys) {
  const opt = document.createElement('option'); opt.value = key;
  opt.textContent = key.replace(/_/g, ' '); select.appendChild(opt);
}

function setTitleLabels(name) {
  const p = name.replace(/_/g,' ').toUpperCase();
  document.getElementById('titleSpecies').textContent = p;
  document.getElementById('label').textContent = p + ' · Spatiotemporal Acoustic Manifold';
}

function updateManifoldLegend() {
  const el = document.getElementById('manifold-legend');
  el.classList.toggle('visible', !!secondary.manifold);
  document.getElementById('legend-primary').textContent = select.value.replace(/_/g,' ');
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

// ── Overlay / Pause ────────────────────────────────────────────
const overlay   = document.getElementById('overlay');
const playBtn   = document.getElementById('playBtn');
const modeBadge = document.getElementById('mode-badge');

playBtn.style.display = 'block';
playBtn.textContent = 'Play';

function startPlayback() {
  playing = true;
  playBtn.textContent = 'Pause';
  lastRAF = null;
  primary.audio?.play().catch(() => {});
  if (secondary.audio) secondary.audio.play().catch(() => {});
}

function pausePlayback() {
  playing = false;
  playBtn.textContent = 'Play';
  lastRAF = null;
  primary.audio?.pause();
  secondary.audio?.pause();
  // FIX: clear trail lines immediately so no frozen glow remains while paused
  clearTrail(primary);
  clearTrail(secondary);
  // FIX: hide halo while paused — no active node indicator when stopped
  if (primary.haloMat)   primary.haloMat.opacity   = 0;
  if (secondary.haloMat) secondary.haloMat.opacity = 0;
}

overlay.addEventListener('click', () => {
  overlay.classList.add('hidden');
  setTimeout(() => { overlay.style.display = 'none'; }, 900);
  startPlayback();
}, { once: true });

playBtn.addEventListener('click', e => {
  e.stopPropagation();
  if (playing) pausePlayback(); else startPlayback();
});

// ── Upload ──────────────────────────────────────────────────
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
    if (modeBadge) modeBadge.textContent = m.species.replace(/_/g,' ').toUpperCase();
    document.getElementById('legend-secondary').textContent = m.species.replace(/_/g,' ');
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
    if (bar) { bar.style.opacity = '1'; bar.style.width = pct + '%'; }
    if (pct >= 100) setTimeout(() => { if (bar) bar.style.opacity = '0'; }, 800);
  },
});

// ── Animation loop ─────────────────────────────────────────────
function animate(ts) {
  requestAnimationFrame(animate);
  controls.update();

  if (playing) {
    let delta = 0;
    if (lastRAF !== null) {
      delta = Math.min((ts - lastRAF) / 1000, 0.1); // cap at 100ms to avoid jump on tab restore
      clockTime += delta;
    }
    lastRAF = ts;
    // Only tick if delta > 0 — skips the first frame after resume to prevent smoothIdx jump
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
