import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { initUpload } from './upload.js';

// ── Renderers ──────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x050508, 1);
document.body.appendChild(renderer.domElement);

const css2d = new CSS2DRenderer();
css2d.setSize(innerWidth, innerHeight);
css2d.domElement.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:5;';
document.body.appendChild(css2d.domElement);

// ── Scene / Camera ──────────────────────────────────────
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.001, 200);
camera.position.set(2, 1.4, 3);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping   = true;
controls.dampingFactor   = 0.06;
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.35;
controls.minDistance     = 1;
controls.maxDistance     = 10;

// ── Helpers ─────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Two tint sets: primary (cool blue→orange, original palette) and
// secondary (warm gold tones) for the uploaded/comparison manifold.
function energyToColor(e, tint = 'primary') {
  const c = new THREE.Color();
  if (tint === 'secondary') {
    // Warm gold palette for uploaded / comparison manifold
    c.setHSL(0.12 - clamp(e, 0, 1) * 0.08, 0.9, 0.55);
  } else {
    c.setHSL(0.62 - clamp(e, 0, 1) * 0.62, 0.95, 0.58);
  }
  return c;
}

function smoothPoints(pts, divisions = 4) {
  if (pts.length < 2) return pts;
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  return curve.getPoints(pts.length * divisions);
}

// ── Error display ────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById('error-state');
  if (el) { el.querySelector('.error-msg').textContent = msg; el.style.display = 'flex'; }
  const overlay = document.getElementById('overlay');
  if (overlay) overlay.style.display = 'none';
}

// ── Axes ─────────────────────────────────────────────────
function addAxis(a, b, hex) {
  const g = new THREE.BufferGeometry().setFromPoints([a, b]);
  const m = new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.3 });
  scene.add(new THREE.Line(g, m));
}
addAxis(new THREE.Vector3(-1.4,0,0), new THREE.Vector3(1.4,0,0), 0xff3333);
addAxis(new THREE.Vector3(0,-1.4,0), new THREE.Vector3(0,1.4,0), 0x44ff88);
addAxis(new THREE.Vector3(0,0,-1.4), new THREE.Vector3(0,0,1.4), 0x3a6fff);

for (const v of [-1, -0.5, 0.5, 1]) {
  const t = 0.025;
  addAxis(new THREE.Vector3(v,-t,0), new THREE.Vector3(v,t,0), 0xff3333);
  addAxis(new THREE.Vector3(-t,v,0), new THREE.Vector3(t,v,0), 0x44ff88);
  addAxis(new THREE.Vector3(0,-t,v), new THREE.Vector3(0,t,v), 0x3a6fff);
}

function addAxisLabel(text, pos, col) {
  const el = document.createElement('div');
  el.textContent = text;
  el.style.cssText = `color:${col};font-size:9px;letter-spacing:.15em;
    padding:2px 5px;background:rgba(5,5,8,.5);border-radius:3px;
    text-transform:uppercase;font-family:Helvetica Neue,sans-serif;`;
  const obj = new CSS2DObject(el);
  obj.position.copy(pos);
  scene.add(obj);
}
addAxisLabel('PC1', new THREE.Vector3(1.48, 0.05, 0), '#ff5555');
addAxisLabel('PC2', new THREE.Vector3(0.05, 1.48, 0), '#55ff88');
addAxisLabel('PC3', new THREE.Vector3(0.05, 0, 1.48), '#5599ff');

// ── Manifold slots ────────────────────────────────────────
// Slot 0 = primary (species selector), Slot 1 = secondary (upload)

const TAIL = 280;

function makeSlot(tint) {
  return {
    tint,
    manifold: null,
    audio: null,
    dot: null, dotMat: null,
    tailGeo: null, tPos: null, tCol: null, tRaw: null,
    inkGeo: null, inkPos: null, inkCol: null,
    inkHead: 0, lastIdx: -1,
    ghostLine: null,
    objects: [],   // all scene objects to remove on swap
  };
}

const primary   = makeSlot('primary');    // species dropdown
const secondary = makeSlot('secondary'); // upload

function disposeSlot(slot) {
  for (const obj of slot.objects) {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }
  slot.objects = [];
  if (slot.tailGeo) { slot.tailGeo.dispose(); slot.tailGeo = null; }
  if (slot.inkGeo)  { slot.inkGeo.dispose();  slot.inkGeo  = null; }
  slot.manifold = null;
  slot.inkHead  = 0;
  slot.lastIdx  = -1;
}

function buildSlot(slot, manifold) {
  disposeSlot(slot);

  const rawPts = manifold.xyz.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const points = smoothPoints(rawPts, 3);
  const energy = manifold.energy || [];
  const times  = manifold.t;

  // Ghost path
  {
    const positions = new Float32Array(points.length * 3);
    const colors    = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      positions[i*3] = p.x; positions[i*3+1] = p.y; positions[i*3+2] = p.z;
      const ri = Math.floor((i / points.length) * rawPts.length);
      const c  = energyToColor((energy[ri] ?? 0) * 0.4, slot.tint);
      colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
    const line = new THREE.Line(geo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.07 }));
    scene.add(line);
    slot.ghostLine = line;
    slot.objects.push(line);
  }

  // Dot
  slot.dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  slot.dot    = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 16), slot.dotMat);
  scene.add(slot.dot);
  slot.objects.push(slot.dot);

  // Tail
  slot.tPos = new Float32Array(TAIL * 3);
  slot.tCol = new Float32Array(TAIL * 3);
  slot.tRaw = new Float32Array(TAIL * 3);
  slot.tailGeo = new THREE.BufferGeometry();
  slot.tailGeo.setAttribute('position', new THREE.BufferAttribute(slot.tPos, 3));
  slot.tailGeo.setAttribute('color',    new THREE.BufferAttribute(slot.tCol, 3));
  const tailLine = new THREE.Line(slot.tailGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1 }));
  scene.add(tailLine);
  slot.objects.push(tailLine);

  // Ink path
  slot.inkPos  = new Float32Array(points.length * 3);
  slot.inkCol  = new Float32Array(points.length * 3);
  slot.inkHead = 0; slot.lastIdx = -1;
  slot.inkGeo  = new THREE.BufferGeometry();
  slot.inkGeo.setAttribute('position', new THREE.BufferAttribute(slot.inkPos, 3));
  slot.inkGeo.setAttribute('color',    new THREE.BufferAttribute(slot.inkCol, 3));
  slot.inkGeo.setDrawRange(0, 0);
  const inkLine = new THREE.Line(slot.inkGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.45 }));
  scene.add(inkLine);
  slot.objects.push(inkLine);

  slot.manifold = { times, energy, points, rawPts, duration_s: manifold.duration_s };
}

function pushTail(slot, pos, e) {
  const { tPos, tCol, tRaw, tailGeo, tint } = slot;
  for (let j = TAIL - 1; j > 0; j--) {
    tPos[j*3]   = tPos[(j-1)*3];
    tPos[j*3+1] = tPos[(j-1)*3+1];
    tPos[j*3+2] = tPos[(j-1)*3+2];
    tRaw[j*3]   = tRaw[(j-1)*3];
    tRaw[j*3+1] = tRaw[(j-1)*3+1];
    tRaw[j*3+2] = tRaw[(j-1)*3+2];
  }
  tPos[0] = pos.x; tPos[1] = pos.y; tPos[2] = pos.z;
  const c = energyToColor(e, tint);
  tRaw[0] = c.r; tRaw[1] = c.g; tRaw[2] = c.b;
  for (let j = 0; j < TAIL; j++) {
    const f = Math.pow(1 - j / TAIL, 1.8);
    tCol[j*3]   = tRaw[j*3]   * f;
    tCol[j*3+1] = tRaw[j*3+1] * f;
    tCol[j*3+2] = tRaw[j*3+2] * f;
  }
  tailGeo.attributes.position.needsUpdate = true;
  tailGeo.attributes.color.needsUpdate    = true;
}

function inkPath(slot, i, e) {
  const { manifold, inkGeo, inkPos, inkCol, tint } = slot;
  if (!manifold || i === slot.lastIdx || slot.inkHead >= manifold.points.length) return;
  slot.lastIdx = i;
  const pos = manifold.points[i];
  const c   = energyToColor(e, tint);
  inkPos[slot.inkHead*3]   = pos.x;
  inkPos[slot.inkHead*3+1] = pos.y;
  inkPos[slot.inkHead*3+2] = pos.z;
  inkCol[slot.inkHead*3]   = c.r;
  inkCol[slot.inkHead*3+1] = c.g;
  inkCol[slot.inkHead*3+2] = c.b;
  slot.inkHead++;
  inkGeo.setDrawRange(0, slot.inkHead);
  inkGeo.attributes.position.needsUpdate = true;
  inkGeo.attributes.color.needsUpdate    = true;
}

function indexForTime(slot, ct) {
  if (!slot.manifold) return 0;
  const { times, points, duration_s } = slot.manifold;
  const manifoldDur = duration_s ?? times[times.length - 1] ?? 10;
  const tt = manifoldDur > 0 ? ct % manifoldDur : ct;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < tt) lo = mid + 1; else hi = mid;
  }
  const rawIdx = clamp(lo, 0, times.length - 1);
  return Math.floor((rawIdx / times.length) * points.length);
}

// ── Audio helpers ─────────────────────────────────────────
function loadAudioForSlot(slot, src) {
  if (slot.audio) { slot.audio.pause(); slot.audio.src = ''; }
  const audio = new Audio(src);
  audio.loop  = true;
  slot.audio  = audio;
  return audio;
}

// ── Load JSON ─────────────────────────────────────────────
let allData;
try {
  const res = await fetch('./birdsong_data.json');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  allData = await res.json();
} catch (err) {
  showError(`Failed to load data: ${err.message}`);
  throw err;
}

// Normalise: support both single-species and multi-species JSON
const isSingle = Array.isArray(allData?.t) && Array.isArray(allData?.xyz);
const speciesMap = isSingle
  ? { [(allData.species || 'birdsong')]: allData }
  : allData;

const speciesKeys = Object.keys(speciesMap);
if (!speciesKeys.length) {
  showError('birdsong_data.json is empty — run process_birds.py first.');
  throw new Error('Empty JSON');
}

// ── Populate dropdown ─────────────────────────────────────
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

// Update manifold-legend dots
function updateManifoldLegend() {
  const legendEl = document.getElementById('manifold-legend');
  const hasSecondary = !!secondary.manifold;
  legendEl.classList.toggle('visible', hasSecondary);

  const primaryName = select.value.replace(/_/g, ' ');
  document.getElementById('legend-primary').textContent = primaryName;

  // colour of primary dot: mid-energy primary tint
  const c = new THREE.Color();
  c.setHSL(0.62 - 0.31, 0.95, 0.58); // ~cyan
  document.getElementById('dot-primary').style.background =
    `rgb(${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)})`;
}

// ── Load a species into the primary slot ─────────────────
function loadSpecies(key) {
  const mRaw = speciesMap[key];
  if (!mRaw) return;
  const m = { ...mRaw, duration_s: mRaw.duration_s ?? mRaw.t[mRaw.t.length - 1] ?? 10 };

  buildSlot(primary, m);
  loadAudioForSlot(primary, `./birds/${key}.mp3`);
  setTitleLabels(key);
  updateManifoldLegend();

  if (started) {
    primary.audio.play().catch(() => {});
    paused = false;
    playBtn.textContent = 'Pause';
  }
}

// Initial load: first species
loadSpecies(speciesKeys[0]);
select.value = speciesKeys[0];

// Dropdown change
select.addEventListener('change', () => {
  // Fade: stop primary audio, switch manifold
  if (primary.audio) primary.audio.pause();
  loadSpecies(select.value);
});

// ── Overlay / play button ─────────────────────────────────
let started = false;
let paused  = false;

const overlay = document.getElementById('overlay');
const playBtn = document.getElementById('playBtn');
const modeBadge = document.getElementById('mode-badge');

overlay.addEventListener('click', async () => {
  try {
    await primary.audio?.play();
    // also start secondary if present
    if (secondary.audio) secondary.audio.play().catch(() => {});
    overlay.classList.add('hidden');
    setTimeout(() => overlay.style.display = 'none', 900);
    playBtn.style.display = 'block';
    started = true;
  } catch (err) {
    showError(`Could not play audio: ${err.message}`);
  }
}, { once: true });

playBtn.addEventListener('click', async () => {
  if (!paused) {
    primary.audio?.pause();
    secondary.audio?.pause();
    paused = true; playBtn.textContent = 'Resume';
  } else {
    try {
      await primary.audio?.play();
      if (secondary.audio) secondary.audio.play().catch(() => {});
      paused = false; playBtn.textContent = 'Pause';
    } catch (err) {
      showError(`Could not resume: ${err.message}`);
    }
  }
});

// ── Upload integration ────────────────────────────────────
initUpload({
  onManifold(uploadedManifold, file) {
    buildSlot(secondary, uploadedManifold);
    loadAudioForSlot(secondary, URL.createObjectURL(file));

    const name = uploadedManifold.species.replace(/_/g, ' ').toUpperCase();
    if (modeBadge) modeBadge.textContent = name;
    document.getElementById('legend-secondary').textContent = uploadedManifold.species.replace(/_/g, ' ');
    updateManifoldLegend();

    if (started) {
      secondary.audio.play().catch(() => {});
      paused = false;
      playBtn.textContent = 'Pause';
    }
  },
  onError(err) { console.error('Upload error:', err); },
  onProgress() {}
});

// ── Animate ───────────────────────────────────────────────
function tickSlot(slot) {
  if (!slot.manifold || !slot.audio) return;
  const i   = indexForTime(slot, slot.audio.currentTime);
  const pos = slot.manifold.points[i];
  const ri  = Math.floor((i / slot.manifold.points.length) * slot.manifold.rawPts.length);
  const e   = slot.manifold.energy[ri] ?? 0;

  slot.dot.position.copy(pos);
  slot.dotMat.color = energyToColor(e, slot.tint);
  const s = 1 + e * 2.0;
  slot.dot.scale.set(s, s, s);

  pushTail(slot, pos, e);
  inkPath(slot, i, e);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (started && !paused) {
    tickSlot(primary);
    if (secondary.manifold) tickSlot(secondary);
  }

  renderer.render(scene, camera);
  css2d.render(scene, camera);
}

animate();

// ── Resize ────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  css2d.setSize(innerWidth, innerHeight);
});
