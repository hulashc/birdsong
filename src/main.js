import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { initUpload } from './upload.js';
import { buildIndex, classify, distToSimilarity } from './knn.js';

// ── Renderers ──────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0a0a0f, 1);
document.body.appendChild(renderer.domElement);

const css2d = new CSS2DRenderer();
css2d.setSize(innerWidth, innerHeight);
css2d.domElement.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:5;';
document.body.appendChild(css2d.domElement);

// ── Scene / Camera ─────────────────────────────────────────────────────────
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.001, 200);
camera.position.set(2.2, 1.5, 3.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping   = true;
controls.dampingFactor   = 0.05;
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.3;
controls.minDistance     = 0.8;
controls.maxDistance     = 10;

// ── Helpers ────────────────────────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Neon palette: cool → hot as energy rises
// primary   = teal/cyan → electric white
// secondary = amber/gold → hot white
function energyToColor(e, tint = 'primary') {
  const t = clamp(e, 0, 1);
  const c = new THREE.Color();
  if (tint === 'secondary') {
    // amber → gold → hot white
    c.setRGB(0.6 + t * 0.4, 0.35 + t * 0.55, 0.0 + t * 0.9);
  } else {
    // deep blue → cyan → electric white
    c.setRGB(0.0 + t * 0.85, 0.4 + t * 0.55, 0.7 + t * 0.3);
  }
  return c;
}

function glowColor(e, tint, brightness) {
  const c = energyToColor(e, tint);
  c.multiplyScalar(brightness);
  return c;
}

function smoothPoints(pts, divisions = 6) {
  if (pts.length < 2) return pts.map(p => p.clone());
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5);
  return curve.getPoints(pts.length * divisions);
}

// ── Error display ──────────────────────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById('error-state');
  if (el) { el.querySelector('.error-msg').textContent = msg; el.style.display = 'flex'; }
  const ov = document.getElementById('overlay');
  if (ov) ov.style.display = 'none';
}

// ── Axes ───────────────────────────────────────────────────────────────────
function addAxis(a, b, hex, opacity = 0.10) {
  const g = new THREE.BufferGeometry().setFromPoints([a, b]);
  const m = new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity });
  scene.add(new THREE.Line(g, m));
}
addAxis(new THREE.Vector3(-1.5,0,0), new THREE.Vector3(1.5,0,0), 0x441111, 0.18);
addAxis(new THREE.Vector3(0,-1.5,0), new THREE.Vector3(0,1.5,0), 0x114411, 0.18);
addAxis(new THREE.Vector3(0,0,-1.5), new THREE.Vector3(0,0,1.5), 0x111144, 0.18);
for (const v of [-1, -0.5, 0.5, 1]) {
  const t = 0.016;
  addAxis(new THREE.Vector3(v,-t,0), new THREE.Vector3(v,t,0), 0x441111, 0.12);
  addAxis(new THREE.Vector3(-t,v,0), new THREE.Vector3(t,v,0), 0x114411, 0.12);
  addAxis(new THREE.Vector3(0,-t,v), new THREE.Vector3(0,t,v), 0x111144, 0.12);
}

function addAxisLabel(text, pos, col) {
  const el = document.createElement('div');
  el.textContent = text;
  el.style.cssText = [
    `color:${col}`, 'font-size:7px', 'letter-spacing:.3em',
    'font-weight:700', 'text-transform:uppercase',
    'font-family:Helvetica Neue,sans-serif', 'opacity:0.3', 'padding:1px 4px',
  ].join(';');
  const obj = new CSS2DObject(el);
  obj.position.copy(pos);
  scene.add(obj);
}
addAxisLabel('PC1', new THREE.Vector3(1.58, 0.04, 0), '#ff4444');
addAxisLabel('PC2', new THREE.Vector3(0.04, 1.58, 0), '#44ff88');
addAxisLabel('PC3', new THREE.Vector3(0.04, 0, 1.58), '#4488ff');

// ── Slot structure ─────────────────────────────────────────────────────────
// WINDOW: fraction of smoothed path visible at once (behind + ahead of traveller)
const WINDOW_BACK  = 0.06;  // 6% of path shown behind
const WINDOW_AHEAD = 0.04;  // 4% of path shown ahead
const TAIL_LEN     = 55;    // comet tail history frames
const NODE_COUNT   = 72;    // point cloud nodes

function makeSlot(tint) {
  return {
    tint, manifold: null,
    audio: null, audioReady: false,
    // Traveller sphere
    dot: null, dotMat: null,
    // Point cloud (InstancedMesh)
    cloud: null,
    // Rolling window arc (BufferGeometry line, updated each frame)
    arcGeo: null, arcPos: null, arcCol: null, arcLine: null,
    // Comet tail
    tailGeo: null, tPos: null, tCol: null, tRaw: null,
    objects: [],
  };
}

const primary   = makeSlot('primary');
const secondary = makeSlot('secondary');

const SPHERE_GEO = new THREE.SphereGeometry(0.008, 8, 8);

function disposeSlot(slot) {
  for (const obj of slot.objects) {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
      else obj.material.dispose();
    }
  }
  slot.objects = [];
  if (slot.tailGeo) { slot.tailGeo.dispose(); slot.tailGeo = null; }
  if (slot.arcGeo)  { slot.arcGeo.dispose();  slot.arcGeo  = null; }
  slot.manifold = null;
  slot.cloud = null; slot.arcLine = null;
}

function buildSlot(slot, manifold) {
  disposeSlot(slot);
  const rawPts = manifold.xyz.map(([x,y,z]) => new THREE.Vector3(x,y,z));
  const points = smoothPoints(rawPts, 6);
  const energy = manifold.energy || [];
  const times  = manifold.t;
  const N      = points.length;

  // ── 1. Point cloud ──
  const cloudMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 });
  // We'll set per-instance colour via instanceColor
  cloudMat.vertexColors = true;
  cloudMat.opacity = 1; // controlled per-instance via colour brightness
  const cloud = new THREE.InstancedMesh(SPHERE_GEO, cloudMat, NODE_COUNT);
  cloud.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  if (!cloud.instanceColor) {
    cloud.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NODE_COUNT * 3), 3);
  }

  const dummy     = new THREE.Object3D();
  const nodeStep  = Math.floor(rawPts.length / NODE_COUNT);
  const nodeColor = new THREE.Color();

  for (let i = 0; i < NODE_COUNT; i++) {
    const ri = Math.min(i * nodeStep, rawPts.length - 1);
    const p  = rawPts[ri];
    const e  = energy[ri] ?? 0;
    dummy.position.copy(p);
    dummy.scale.set(1, 1, 1);
    dummy.updateMatrix();
    cloud.setMatrixAt(i, dummy.matrix);
    nodeColor.copy(glowColor(e, slot.tint, 0.08)); // very dim by default
    cloud.setColorAt(i, nodeColor);
  }
  cloud.instanceMatrix.needsUpdate = true;
  cloud.instanceColor.needsUpdate  = true;
  scene.add(cloud);
  slot.cloud = cloud;
  slot.objects.push(cloud);

  // ── 2. Rolling arc line ── (MAX_ARC_PTS covers WINDOW_BACK + WINDOW_AHEAD)
  const MAX_ARC = Math.ceil((WINDOW_BACK + WINDOW_AHEAD) * N) + 4;
  slot.arcPos = new Float32Array(MAX_ARC * 3);
  slot.arcCol = new Float32Array(MAX_ARC * 3);
  slot.arcGeo = new THREE.BufferGeometry();
  slot.arcGeo.setAttribute('position', new THREE.BufferAttribute(slot.arcPos, 3));
  slot.arcGeo.setAttribute('color',    new THREE.BufferAttribute(slot.arcCol, 3));
  slot.arcGeo.setDrawRange(0, 0);
  const arcMat  = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
  const arcLine = new THREE.Line(slot.arcGeo, arcMat);
  scene.add(arcLine);
  slot.arcLine = arcLine;
  slot.objects.push(arcLine);

  // ── 3. Traveller dot ──
  const dotGeo = new THREE.SphereGeometry(0.018, 20, 20);
  slot.dotMat  = new THREE.MeshBasicMaterial({ color: 0xffffff });
  slot.dot     = new THREE.Mesh(dotGeo, slot.dotMat);
  scene.add(slot.dot);
  slot.objects.push(slot.dot);

  // ── 4. Comet tail ──
  slot.tPos = new Float32Array(TAIL_LEN * 3);
  slot.tCol = new Float32Array(TAIL_LEN * 3);
  slot.tRaw = new Float32Array(TAIL_LEN * 3);
  slot.tailGeo = new THREE.BufferGeometry();
  slot.tailGeo.setAttribute('position', new THREE.BufferAttribute(slot.tPos, 3));
  slot.tailGeo.setAttribute('color',    new THREE.BufferAttribute(slot.tCol, 3));
  const tailLine = new THREE.Line(slot.tailGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }));
  scene.add(tailLine);
  slot.objects.push(tailLine);

  slot.manifold = { times, energy, points, rawPts, duration_s: manifold.duration_s, N };
}

// ── Arc update (called each frame) ────────────────────────────────────────
function updateArc(slot, headIdx) {
  const { manifold, arcPos, arcCol, arcGeo, tint } = slot;
  const { points, energy, N, rawPts } = manifold;

  const back  = Math.ceil(WINDOW_BACK  * N);
  const ahead = Math.ceil(WINDOW_AHEAD * N);

  const startIdx = Math.max(0, headIdx - back);
  const endIdx   = Math.min(N - 1, headIdx + ahead);
  const count    = endIdx - startIdx + 1;

  for (let j = 0; j < count; j++) {
    const idx = startIdx + j;
    const p   = points[idx];
    arcPos[j*3]   = p.x;
    arcPos[j*3+1] = p.y;
    arcPos[j*3+2] = p.z;

    // Distance from head: 0 = head, positive = behind, negative = ahead
    const dist    = headIdx - idx;           // + behind, - ahead
    const normDist = Math.abs(dist) / Math.max(back, ahead);

    // Behind: bright centre → dim tail; Ahead: fainter ghost
    let brightness;
    if (dist >= 0) {
      // behind: sharp falloff
      brightness = Math.pow(1 - normDist, 2.2) * 0.95;
    } else {
      // ahead: very faint — just enough to show where we're going
      brightness = (1 - normDist) * 0.18;
    }

    const ri = Math.floor((idx / N) * rawPts.length);
    const e  = energy[ri] ?? 0;
    const c  = glowColor(e, tint, brightness);
    arcCol[j*3]   = c.r;
    arcCol[j*3+1] = c.g;
    arcCol[j*3+2] = c.b;
  }

  arcGeo.setDrawRange(0, count);
  arcGeo.attributes.position.needsUpdate = true;
  arcGeo.attributes.color.needsUpdate    = true;
}

// ── Comet tail ─────────────────────────────────────────────────────────────
function pushTail(slot, pos, e) {
  const { tPos, tCol, tRaw, tailGeo, tint } = slot;
  // Shift history back
  for (let j = TAIL_LEN - 1; j > 0; j--) {
    tPos[j*3]   = tPos[(j-1)*3];   tPos[j*3+1] = tPos[(j-1)*3+1]; tPos[j*3+2] = tPos[(j-1)*3+2];
    tRaw[j*3]   = tRaw[(j-1)*3];   tRaw[j*3+1] = tRaw[(j-1)*3+1]; tRaw[j*3+2] = tRaw[(j-1)*3+2];
  }
  tPos[0] = pos.x; tPos[1] = pos.y; tPos[2] = pos.z;
  const c = glowColor(e, tint, 1.2);
  tRaw[0] = c.r; tRaw[1] = c.g; tRaw[2] = c.b;
  for (let j = 0; j < TAIL_LEN; j++) {
    // Sharp exponential fade
    const f = Math.pow(1 - j / TAIL_LEN, 2.8);
    tCol[j*3] = tRaw[j*3]*f; tCol[j*3+1] = tRaw[j*3+1]*f; tCol[j*3+2] = tRaw[j*3+2]*f;
  }
  tailGeo.attributes.position.needsUpdate = true;
  tailGeo.attributes.color.needsUpdate    = true;
}

// ── Point cloud pulse ──────────────────────────────────────────────────────
const _dummy = new THREE.Object3D();
const _nc    = new THREE.Color();

function pulseCloud(slot, headRawIdx, e) {
  const { cloud, manifold, tint } = slot;
  if (!cloud) return;
  const rawLen  = manifold.rawPts.length;
  const nodeStep = Math.floor(rawLen / NODE_COUNT);

  for (let n = 0; n < NODE_COUNT; n++) {
    const ni       = Math.min(n * nodeStep, rawLen - 1);
    const ne       = manifold.energy[ni] ?? 0;
    const nodeDist = Math.abs(ni - headRawIdx) / rawLen; // 0–1

    // Nodes within ~8% of current position glow; rest are near-invisible
    const proximity = Math.max(0, 1 - nodeDist * 12);
    const bright    = proximity > 0
      ? 0.08 + proximity * (1.6 + e * 1.2)  // glow up to ~2.8x
      : 0.04;                                 // almost invisible otherwise

    const nodeSize = 0.8 + ne * 0.6 + (proximity > 0.5 ? proximity * 0.8 : 0);

    cloud.getMatrixAt(n, _dummy.matrix);
    _dummy.matrix.decompose(_dummy.position, _dummy.quaternion, _dummy.scale);
    _dummy.scale.set(nodeSize, nodeSize, nodeSize);
    _dummy.updateMatrix();
    cloud.setMatrixAt(n, _dummy.matrix);

    _nc.copy(glowColor(ne, tint, bright));
    cloud.setColorAt(n, _nc);
  }
  cloud.instanceMatrix.needsUpdate = true;
  cloud.instanceColor.needsUpdate  = true;
}

// ── Timing: use audio.currentTime directly, fallback to clock ─────────────
let clockTime = 0, lastTS = null;

function currentTime(slot) {
  if (slot.audioReady && slot.audio && !slot.audio.paused) {
    return slot.audio.currentTime;
  }
  return clockTime;
}

function indexForTime(slot, ct) {
  if (!slot.manifold) return 0;
  const { times, N, duration_s } = slot.manifold;
  const dur = duration_s ?? times[times.length - 1] ?? 10;
  const tt  = dur > 0 ? ct % dur : ct;

  // Binary search on times array
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < tt) lo = mid + 1; else hi = mid;
  }
  const rawIdx = clamp(lo, 0, times.length - 1);
  // Map raw frame → smoothed point index
  return Math.floor((rawIdx / times.length) * N);
}

// ── Audio helpers ──────────────────────────────────────────────────────────
function loadAudioForSlot(slot, key) {
  if (slot.audio) { slot.audio.pause(); slot.audio.src = ''; }
  slot.audioReady = false;
  const candidates = [`./birds/${key}.ogg`, `./birds/${key}.mp3`];
  let idx = 0;
  const audio = new Audio();
  audio.loop = true;
  slot.audio = audio;
  function tryNext() {
    if (idx >= candidates.length) { console.warn(`[birdsong] no audio for "${key}"`); return; }
    audio.src = candidates[idx++];
    audio.load();
  }
  audio.addEventListener('canplaythrough', () => {
    slot.audioReady = true;
    if (started && !paused) audio.play().catch(() => {});
  }, { once: true });
  audio.addEventListener('error', tryNext);
  tryNext();
}

// ── Load data ──────────────────────────────────────────────────────────────
let allData;
try {
  const res = await fetch('./birdsong_data.json');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  allData = await res.json();
} catch (err) {
  showError(`Failed to load data: ${err.message}`);
  throw err;
}

const isSingle    = Array.isArray(allData?.t) && Array.isArray(allData?.xyz);
const speciesMap  = isSingle ? { [(allData.species || 'birdsong')]: allData } : allData;
const speciesKeys = Object.keys(speciesMap);
if (!speciesKeys.length) { showError('birdsong_data.json is empty.'); throw new Error('Empty'); }

const knnIndex = buildIndex(speciesMap);

// ── Dropdown ───────────────────────────────────────────────────────────────
const select = document.getElementById('speciesSelect');
select.innerHTML = '';
for (const key of speciesKeys) {
  const opt = document.createElement('option');
  opt.value = key;
  opt.textContent = key.replace(/_/g, ' ');
  select.appendChild(opt);
}

function setTitleLabels(name) {
  const pretty = name.replace(/_/g, ' ').toUpperCase();
  document.getElementById('titleSpecies').textContent = pretty;
  document.getElementById('label').textContent = pretty + ' · Spatiotemporal Acoustic Manifold';
}

function updateManifoldLegend() {
  const el = document.getElementById('manifold-legend');
  el.classList.toggle('visible', !!secondary.manifold);
  document.getElementById('legend-primary').textContent = select.value.replace(/_/g,' ');
}

// ── K-NN panel ─────────────────────────────────────────────────────────────
function showKnnResults(results) {
  const panel = document.getElementById('knn-results');
  const list  = document.getElementById('knn-list');
  if (!panel || !list) return;
  list.innerHTML = '';
  results.forEach(({ species, distance, rank }) => {
    const pct = distToSimilarity(distance);
    const row = document.createElement('div');
    row.className = 'knn-row';
    row.innerHTML = `
      <span class="knn-rank">${rank}</span>
      <span class="knn-species">${species.replace(/_/g, ' ')}</span>
      <span class="knn-pct">${pct}%</span>
    `;
    const bar = document.createElement('div');
    bar.className = 'knn-bar';
    bar.innerHTML = `<div class="knn-bar-fill" style="width:${pct}%"></div>`;
    list.appendChild(row);
    list.appendChild(bar);
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

// ── Overlay / Pause ────────────────────────────────────────────────────────
let started = false, paused = false;

const overlay   = document.getElementById('overlay');
const playBtn   = document.getElementById('playBtn');
const modeBadge = document.getElementById('mode-badge');

overlay.addEventListener('click', () => {
  overlay.classList.add('hidden');
  setTimeout(() => overlay.style.display = 'none', 900);
  playBtn.style.display = 'block';
  started = true; paused = false;
  playBtn.textContent = 'Pause';
  primary.audio?.play().catch(() => {});
  if (secondary.audio) secondary.audio.play().catch(() => {});
}, { once: true });

playBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!paused) {
    primary.audio?.pause();
    secondary.audio?.pause();
    paused = true; playBtn.textContent = 'Resume';
  } else {
    primary.audio?.play().catch(() => {});
    if (secondary.audio) secondary.audio.play().catch(() => {});
    paused = false; playBtn.textContent = 'Pause';
  }
});

// ── Upload ─────────────────────────────────────────────────────────────────
initUpload({
  onManifold(m, file) {
    buildSlot(secondary, m);
    if (secondary.audio) { secondary.audio.pause(); secondary.audio.src = ''; }
    secondary.audioReady = false;
    const audio = new Audio(URL.createObjectURL(file));
    audio.loop = true;
    secondary.audio = audio;
    audio.addEventListener('canplaythrough', () => {
      secondary.audioReady = true;
      if (started && !paused) audio.play().catch(() => {});
    }, { once: true });
    if (modeBadge) modeBadge.textContent = m.species.replace(/_/g,' ').toUpperCase();
    document.getElementById('legend-secondary').textContent = m.species.replace(/_/g,' ');
    updateManifoldLegend();
    showKnnResults(classify(m, knnIndex, 3));
  },
  onError(err) { console.error('Upload:', err); hideKnnResults(); },
  onProgress() {},
});

// ── Animation loop ─────────────────────────────────────────────────────────
function tickSlot(slot) {
  if (!slot.manifold) return;
  const { points, energy, rawPts, N } = slot.manifold;

  const ct  = currentTime(slot);
  const i   = indexForTime(slot, ct);          // smoothed index
  const ri  = Math.floor((i / N) * rawPts.length); // raw index for energy
  const e   = energy[ri] ?? 0;
  const pos = points[i];

  // Traveller
  slot.dot.position.copy(pos);
  slot.dotMat.color = glowColor(e, slot.tint, 2.2 + e * 1.5); // always bright
  const ds = 1.0 + e * 1.8;
  slot.dot.scale.set(ds, ds, ds);

  // Rolling arc
  updateArc(slot, i);

  // Point cloud pulse
  pulseCloud(slot, ri, e);

  // Comet tail
  pushTail(slot, pos, e);
}

function animate(ts) {
  requestAnimationFrame(animate);
  controls.update();
  if (started && !paused) {
    if (lastTS !== null) clockTime += (ts - lastTS) / 1000;
    lastTS = ts;
    tickSlot(primary);
    if (secondary.manifold) tickSlot(secondary);
  } else {
    lastTS = null;
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
