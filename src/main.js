import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { initUpload } from './upload.js';

// ── Renderers ────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0e0e0e, 1);
document.body.appendChild(renderer.domElement);

const css2d = new CSS2DRenderer();
css2d.setSize(innerWidth, innerHeight);
css2d.domElement.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:5;';
document.body.appendChild(css2d.domElement);

// ── Scene / Camera ───────────────────────────────────────
const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.001, 200);
camera.position.set(2.2, 1.5, 3.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping   = true;
controls.dampingFactor   = 0.05;
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.28;
controls.minDistance     = 1;
controls.maxDistance     = 12;

// ── Helpers ──────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/**
 * Lucio Arese palette:
 * silence  → deep cool blue-grey  (dark, desaturated)
 * mid      → near-white / silver
 * peak     → warm cream/ivory
 * Secondary slot gets a subtle warm-gold tint.
 */
function energyToColor(e, tint = 'primary') {
  const t = clamp(e, 0, 1);
  const c = new THREE.Color();
  if (tint === 'secondary') {
    // warm gold thread for uploaded audio
    c.setRGB(
      0.55 + t * 0.40,
      0.42 + t * 0.38,
      0.10 + t * 0.05
    );
  } else {
    // cool blue-grey → silver-white → warm ivory
    if (t < 0.5) {
      const u = t / 0.5;
      c.setRGB(
        0.18 + u * 0.52,  // 0.18 → 0.70
        0.20 + u * 0.55,  // 0.20 → 0.75
        0.35 + u * 0.45   // 0.35 → 0.80
      );
    } else {
      const u = (t - 0.5) / 0.5;
      c.setRGB(
        0.70 + u * 0.22,  // 0.70 → 0.92
        0.75 + u * 0.16,  // 0.75 → 0.91
        0.80 - u * 0.16   // 0.80 → 0.64  (warm cream at peak)
      );
    }
  }
  return c;
}

function smoothPoints(pts, divisions = 5) {
  if (pts.length < 2) return pts;
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.5);
  return curve.getPoints(pts.length * divisions);
}

// ── Error display ────────────────────────────────────────
function showError(msg) {
  const el = document.getElementById('error-state');
  if (el) { el.querySelector('.error-msg').textContent = msg; el.style.display = 'flex'; }
  const ov = document.getElementById('overlay');
  if (ov) ov.style.display = 'none';
}

// ── Axes — extremely faint, Arese style ─────────────────
function addAxis(a, b, hex, opacity = 0.12) {
  const g = new THREE.BufferGeometry().setFromPoints([a, b]);
  const m = new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity });
  scene.add(new THREE.Line(g, m));
}
addAxis(new THREE.Vector3(-1.5,0,0), new THREE.Vector3(1.5,0,0), 0xcc3333, 0.14);
addAxis(new THREE.Vector3(0,-1.5,0), new THREE.Vector3(0,1.5,0), 0x33aa55, 0.14);
addAxis(new THREE.Vector3(0,0,-1.5), new THREE.Vector3(0,0,1.5), 0x3366cc, 0.14);

// Fine tick marks
for (const v of [-1, -0.5, 0.5, 1]) {
  const t = 0.018;
  addAxis(new THREE.Vector3(v,-t,0), new THREE.Vector3(v,t,0), 0xcc3333, 0.10);
  addAxis(new THREE.Vector3(-t,v,0), new THREE.Vector3(t,v,0), 0x33aa55, 0.10);
  addAxis(new THREE.Vector3(0,-t,v), new THREE.Vector3(0,t,v), 0x3366cc, 0.10);
}

function addAxisLabel(text, pos, col) {
  const el = document.createElement('div');
  el.textContent = text;
  el.style.cssText = [
    `color:${col}`,
    'font-size:8px',
    'letter-spacing:.2em',
    'font-weight:700',
    'text-transform:uppercase',
    'font-family:Helvetica Neue,sans-serif',
    'opacity:0.45',
    'padding:1px 4px',
  ].join(';');
  const obj = new CSS2DObject(el);
  obj.position.copy(pos);
  scene.add(obj);
}
addAxisLabel('PC1', new THREE.Vector3(1.58, 0.04, 0), '#cc5555');
addAxisLabel('PC2', new THREE.Vector3(0.04, 1.58, 0), '#44bb66');
addAxisLabel('PC3', new THREE.Vector3(0.04, 0, 1.58), '#4477cc');

// ── Manifold slots ───────────────────────────────────────
const TAIL = 320;

function makeSlot(tint) {
  return {
    tint, manifold: null,
    audio: null, audioReady: false,
    dot: null, dotMat: null,
    tailGeo: null, tPos: null, tCol: null, tRaw: null,
    inkGeo: null, inkPos: null, inkCol: null,
    inkHead: 0, lastIdx: -1,
    ghostLine: null, objects: [],
  };
}

const primary   = makeSlot('primary');
const secondary = makeSlot('secondary');

function disposeSlot(slot) {
  for (const obj of slot.objects) {
    scene.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) obj.material.dispose();
  }
  slot.objects = [];
  if (slot.tailGeo) { slot.tailGeo.dispose(); slot.tailGeo = null; }
  if (slot.inkGeo)  { slot.inkGeo.dispose();  slot.inkGeo  = null; }
  slot.manifold = null; slot.inkHead = 0; slot.lastIdx = -1;
}

function buildSlot(slot, manifold) {
  disposeSlot(slot);

  const rawPts = manifold.xyz.map(([x,y,z]) => new THREE.Vector3(x,y,z));
  const points = smoothPoints(rawPts, 4);
  const energy = manifold.energy || [];
  const times  = manifold.t;

  // ── Ghost path — very faint full trajectory
  {
    const positions = new Float32Array(points.length * 3);
    const colors    = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      positions[i*3]   = p.x;
      positions[i*3+1] = p.y;
      positions[i*3+2] = p.z;
      const ri = Math.floor((i / points.length) * rawPts.length);
      const c  = energyToColor((energy[ri] ?? 0), slot.tint);
      colors[i*3]   = c.r * 0.28;
      colors[i*3+1] = c.g * 0.28;
      colors[i*3+2] = c.b * 0.28;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors,    3));
    const line = new THREE.Line(geo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }));
    scene.add(line);
    slot.ghostLine = line;
    slot.objects.push(line);
  }

  // ── Dot — small, clean sphere
  slot.dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  slot.dot    = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 16), slot.dotMat);
  scene.add(slot.dot);
  slot.objects.push(slot.dot);

  // ── Comet tail
  slot.tPos = new Float32Array(TAIL * 3);
  slot.tCol = new Float32Array(TAIL * 3);
  slot.tRaw = new Float32Array(TAIL * 3);
  slot.tailGeo = new THREE.BufferGeometry();
  slot.tailGeo.setAttribute('position', new THREE.BufferAttribute(slot.tPos, 3));
  slot.tailGeo.setAttribute('color',    new THREE.BufferAttribute(slot.tCol, 3));
  const tailLine = new THREE.Line(slot.tailGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
  scene.add(tailLine);
  slot.objects.push(tailLine);

  // ── Ink path — builds up as audio plays
  slot.inkPos  = new Float32Array(points.length * 3);
  slot.inkCol  = new Float32Array(points.length * 3);
  slot.inkHead = 0; slot.lastIdx = -1;
  slot.inkGeo  = new THREE.BufferGeometry();
  slot.inkGeo.setAttribute('position', new THREE.BufferAttribute(slot.inkPos, 3));
  slot.inkGeo.setAttribute('color',    new THREE.BufferAttribute(slot.inkCol, 3));
  slot.inkGeo.setDrawRange(0, 0);
  const inkLine = new THREE.Line(slot.inkGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }));
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
    const f = Math.pow(1 - j / TAIL, 1.6);
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
  inkCol[slot.inkHead*3]   = c.r * 0.7;
  inkCol[slot.inkHead*3+1] = c.g * 0.7;
  inkCol[slot.inkHead*3+2] = c.b * 0.7;
  slot.inkHead++;
  inkGeo.setDrawRange(0, slot.inkHead);
  inkGeo.attributes.position.needsUpdate = true;
  inkGeo.attributes.color.needsUpdate    = true;
}

// ── Clock fallback ───────────────────────────────────────
let clockTime = 0, lastTimestamp = null;

function currentTime(slot) {
  if (slot.audioReady && slot.audio && !slot.audio.paused) return slot.audio.currentTime;
  return clockTime;
}

function indexForTime(slot, ct) {
  if (!slot.manifold) return 0;
  const { times, points, duration_s } = slot.manifold;
  const dur = duration_s ?? times[times.length - 1] ?? 10;
  const tt  = dur > 0 ? ct % dur : ct;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < tt) lo = mid + 1; else hi = mid;
  }
  const rawIdx = clamp(lo, 0, times.length - 1);
  return Math.floor((rawIdx / times.length) * points.length);
}

// ── Audio helpers ────────────────────────────────────────
function loadAudioForSlot(slot, key) {
  if (slot.audio) { slot.audio.pause(); slot.audio.src = ''; }
  slot.audioReady = false;
  const candidates = [`./birds/${key}.ogg`, `./birds/${key}.mp3`];
  let idx = 0;
  const audio = new Audio();
  audio.loop = true;
  slot.audio = audio;
  function tryNext() {
    if (idx >= candidates.length) {
      console.warn(`[birdsong] No audio for "${key}" — clock fallback.`);
      return;
    }
    audio.src = candidates[idx++];
    audio.load();
  }
  audio.addEventListener('canplaythrough', () => {
    slot.audioReady = true;
    if (started && !paused) audio.play().catch(() => {});
  }, { once: true });
  audio.addEventListener('error', tryNext);
  tryNext();
  return audio;
}

// ── Load JSON ────────────────────────────────────────────
let allData;
try {
  const res = await fetch('./birdsong_data.json');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  allData = await res.json();
} catch (err) {
  showError(`Failed to load data: ${err.message}`);
  throw err;
}

const isSingle  = Array.isArray(allData?.t) && Array.isArray(allData?.xyz);
const speciesMap = isSingle ? { [(allData.species || 'birdsong')]: allData } : allData;
const speciesKeys = Object.keys(speciesMap);
if (!speciesKeys.length) { showError('birdsong_data.json is empty.'); throw new Error('Empty'); }

// ── Dropdown ─────────────────────────────────────────────
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
  document.getElementById('label').textContent = pretty + ' \u00b7 Spatiotemporal Acoustic Manifold';
}

function updateManifoldLegend() {
  const el = document.getElementById('manifold-legend');
  el.classList.toggle('visible', !!secondary.manifold);
  document.getElementById('legend-primary').textContent = select.value.replace(/_/g,' ');
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

// ── Overlay / Pause  — FIX: playBtn pointer-events handled here ──
let started = false, paused = false;

const overlay   = document.getElementById('overlay');
const playBtn   = document.getElementById('playBtn');
const modeBadge = document.getElementById('mode-badge');

overlay.addEventListener('click', () => {
  overlay.classList.add('hidden');
  setTimeout(() => overlay.style.display = 'none', 900);
  playBtn.style.display = 'block';
  started = true;
  paused  = false;
  playBtn.textContent = 'Pause';
  primary.audio?.play().catch(() => {});
  if (secondary.audio) secondary.audio.play().catch(() => {});
}, { once: true });

// Pause button — pointer-events:all is set on .tb-right in CSS;
// this listener fires correctly because the button itself is interactive.
playBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  if (!paused) {
    primary.audio?.pause();
    secondary.audio?.pause();
    paused = true;
    playBtn.textContent = 'Resume';
  } else {
    primary.audio?.play().catch(() => {});
    if (secondary.audio) secondary.audio.play().catch(() => {});
    paused = false;
    playBtn.textContent = 'Pause';
  }
});

// ── Upload ───────────────────────────────────────────────
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
  },
  onError(err) { console.error('Upload:', err); },
  onProgress() {}
});

// ── Animation ────────────────────────────────────────────
function tickSlot(slot) {
  if (!slot.manifold) return;
  const ct  = currentTime(slot);
  const i   = indexForTime(slot, ct);
  const pos = slot.manifold.points[i];
  const ri  = Math.floor((i / slot.manifold.points.length) * slot.manifold.rawPts.length);
  const e   = slot.manifold.energy[ri] ?? 0;

  slot.dot.position.copy(pos);
  slot.dotMat.color = energyToColor(e, slot.tint);
  // Dot size: subtle — 1× to 2.2× based on energy
  const s = 1 + e * 1.4;
  slot.dot.scale.set(s, s, s);

  pushTail(slot, pos, e);
  inkPath(slot, i, e);
}

function animate(ts) {
  requestAnimationFrame(animate);
  controls.update();
  if (started && !paused) {
    if (lastTimestamp !== null) clockTime += (ts - lastTimestamp) / 1000;
    lastTimestamp = ts;
    tickSlot(primary);
    if (secondary.manifold) tickSlot(secondary);
  } else {
    lastTimestamp = null;
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
