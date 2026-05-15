import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { initUpload } from './upload.js';

// ── URL param ──────────────────────────────────────────
const soundParam = new URLSearchParams(location.search)
  .get('sound')?.trim().toLowerCase() ?? '';

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

function energyToColor(e) {
  const c = new THREE.Color();
  c.setHSL(0.62 - clamp(e, 0, 1) * 0.62, 0.95, 0.58);
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
  const m = new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.35 });
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
  el.style.cssText = `color:${col};font-size:10px;letter-spacing:.15em;
    padding:2px 5px;background:rgba(5,5,8,.5);border-radius:3px;
    text-transform:uppercase;font-family:Helvetica Neue,sans-serif;`;
  const obj = new CSS2DObject(el);
  obj.position.copy(pos);
  scene.add(obj);
}
addAxisLabel('PC1', new THREE.Vector3(1.48, 0.05, 0), '#ff5555');
addAxisLabel('PC2', new THREE.Vector3(0.05, 1.48, 0), '#55ff88');
addAxisLabel('PC3', new THREE.Vector3(0.05, 0, 1.48), '#5599ff');

// ── Manifold scene state ──────────────────────────────────
let currentManifold = null;
let currentAudio    = null;
let started = false;
let paused  = false;

let dot, dotMat;
let tailGeo, tPos, tCol, tRaw;
let inkGeo, inkPos, inkCol;
let inkHead = 0, lastIdx = -1;
const TAIL = 280;
let ghostLine = null;

// ── Build Three.js geometry from a manifold ────────────────
function buildManifoldScene(manifold) {
  if (dot)      { scene.remove(dot); dot.geometry.dispose(); }
  if (ghostLine){ scene.remove(ghostLine); ghostLine.geometry.dispose(); }

  // Remove old tail/ink lines by traversing scene
  const toRemove = [];
  scene.traverse(obj => {
    if (obj.isLine && obj !== ghostLine) {
      if (obj.geometry === tailGeo || obj.geometry === inkGeo) toRemove.push(obj);
    }
  });
  toRemove.forEach(obj => { scene.remove(obj); obj.geometry.dispose(); });
  if (tailGeo) tailGeo.dispose();
  if (inkGeo)  inkGeo.dispose();

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
      const c  = energyToColor((energy[ri] ?? 0) * 0.4);
      colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));
    ghostLine = new THREE.Line(geo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.08 }));
    scene.add(ghostLine);
  }

  dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  dot    = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 16), dotMat);
  scene.add(dot);

  tPos = new Float32Array(TAIL * 3);
  tCol = new Float32Array(TAIL * 3);
  tRaw = new Float32Array(TAIL * 3);
  tailGeo = new THREE.BufferGeometry();
  tailGeo.setAttribute('position', new THREE.BufferAttribute(tPos, 3));
  tailGeo.setAttribute('color',    new THREE.BufferAttribute(tCol, 3));
  const tailLine = new THREE.Line(tailGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1 }));
  scene.add(tailLine);

  inkPos  = new Float32Array(points.length * 3);
  inkCol  = new Float32Array(points.length * 3);
  inkHead = 0; lastIdx = -1;
  inkGeo  = new THREE.BufferGeometry();
  inkGeo.setAttribute('position', new THREE.BufferAttribute(inkPos, 3));
  inkGeo.setAttribute('color',    new THREE.BufferAttribute(inkCol, 3));
  inkGeo.setDrawRange(0, 0);
  const inkLine = new THREE.Line(inkGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5 }));
  scene.add(inkLine);

  currentManifold = { times, energy, points, rawPts, duration_s: manifold.duration_s };
}

function pushTail(pos, e) {
  for (let j = TAIL - 1; j > 0; j--) {
    tPos[j*3]   = tPos[(j-1)*3];
    tPos[j*3+1] = tPos[(j-1)*3+1];
    tPos[j*3+2] = tPos[(j-1)*3+2];
    tRaw[j*3]   = tRaw[(j-1)*3];
    tRaw[j*3+1] = tRaw[(j-1)*3+1];
    tRaw[j*3+2] = tRaw[(j-1)*3+2];
  }
  tPos[0] = pos.x; tPos[1] = pos.y; tPos[2] = pos.z;
  const c = energyToColor(e);
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

function inkPath(i, e) {
  if (!currentManifold || i === lastIdx || inkHead >= currentManifold.points.length) return;
  lastIdx = i;
  const pos = currentManifold.points[i];
  const c   = energyToColor(e);
  inkPos[inkHead*3]   = pos.x;
  inkPos[inkHead*3+1] = pos.y;
  inkPos[inkHead*3+2] = pos.z;
  inkCol[inkHead*3]   = c.r;
  inkCol[inkHead*3+1] = c.g;
  inkCol[inkHead*3+2] = c.b;
  inkHead++;
  inkGeo.setDrawRange(0, inkHead);
  inkGeo.attributes.position.needsUpdate = true;
  inkGeo.attributes.color.needsUpdate    = true;
}

function indexForTime(ct) {
  if (!currentManifold) return 0;
  const { times, points, duration_s } = currentManifold;
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

// ── Load JSON ─────────────────────────────────────────────
let all, manifold, soundName;
try {
  const res = await fetch('./birdsong_data.json');
  if (!res.ok) throw new Error(`HTTP ${res.status} — could not load birdsong_data.json`);
  all = await res.json();
} catch (err) {
  showError(`Failed to load data: ${err.message}`);
  throw err;
}

const isSingle = Array.isArray(all?.t) && Array.isArray(all?.xyz);
if (isSingle) {
  manifold  = all;
  soundName = (all.species || soundParam || 'birdsong').toLowerCase();
} else {
  const keys = Object.keys(all);
  if (!keys.length) {
    showError('birdsong_data.json is empty — run process_birds.py first.');
    throw new Error('Empty JSON');
  }
  soundName = (soundParam && all[soundParam]) ? soundParam : keys[0];
  manifold  = all[soundName];
}

manifold.duration_s = manifold.duration_s ?? manifold.t[manifold.t.length - 1] ?? 10;

document.getElementById('titleSpecies').textContent =
  soundName.replace(/_/g, ' ').toUpperCase();
document.getElementById('label').textContent =
  soundName.replace(/_/g, ' ').toUpperCase() + ' · Spatiotemporal Acoustic Manifold';

buildManifoldScene(manifold);

// ── Audio ─────────────────────────────────────────────────
function loadAudio(src) {
  if (currentAudio) { currentAudio.pause(); currentAudio.src = ''; }
  const audio = new Audio(src);
  audio.loop  = true;
  audio.addEventListener('error', () => showError(`Audio not found: ${src}`));
  currentAudio = audio;
  return audio;
}

let audio = loadAudio(`./birds/${soundName}.mp3`);

const overlay = document.getElementById('overlay');
const playBtn = document.getElementById('playBtn');
const modeBadge = document.getElementById('mode-badge');

overlay.addEventListener('click', async () => {
  try {
    await audio.play();
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
    currentAudio?.pause(); paused = true; playBtn.textContent = 'Resume';
  } else {
    try {
      await currentAudio?.play(); paused = false; playBtn.textContent = 'Pause';
    } catch (err) {
      showError(`Could not resume audio: ${err.message}`);
    }
  }
});

// ── Upload integration ────────────────────────────────────
initUpload({
  onManifold(uploadedManifold, file) {
    inkHead = 0; lastIdx = -1;
    buildManifoldScene(uploadedManifold);

    const name = uploadedManifold.species.replace(/_/g, ' ').toUpperCase();
    document.getElementById('titleSpecies').textContent = name;
    document.getElementById('label').textContent = name + ' · Spatiotemporal Acoustic Manifold';
    if (modeBadge) modeBadge.textContent = 'Uploaded audio';

    const url = URL.createObjectURL(file);
    audio = loadAudio(url);

    if (started) {
      audio.play().catch(err => {
        if (statusEl) statusEl.textContent = 'Tap play to start: ' + err.message;
      });
      paused = false;
      playBtn.textContent = 'Pause';
    }
  },
  onError(err) { console.error('Upload error:', err); },
  onProgress(stage) {
    if (stage === 'done' && modeBadge) modeBadge.textContent = 'Uploaded audio';
  }
});

// ── Animate ───────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (started && !paused && currentAudio && currentManifold) {
    const i   = indexForTime(currentAudio.currentTime);
    const pos = currentManifold.points[i];
    const ri  = Math.floor((i / currentManifold.points.length) * currentManifold.rawPts.length);
    const e   = currentManifold.energy[ri] ?? 0;

    dot.position.copy(pos);
    dotMat.color = energyToColor(e);
    const s = 1 + e * 2.0;
    dot.scale.set(s, s, s);

    pushTail(pos, e);
    inkPath(i, e);
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
