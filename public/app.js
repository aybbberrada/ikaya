// Ikaya Gaze — webcam eye tracking in the browser.
//
// Pipeline:
//   front camera → MediaPipe Face Landmarker (iris landmarks + eye blendshapes + head pose)
//   → per-frame feature vector → ridge regression fitted during calibration
//   → One-Euro smoothing → drawn on a full-screen canvas.

import {
  FaceLandmarker,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const STORE_KEY = 'ikaya.calibration.v1';

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const preview = $('preview');
const pctx = preview.getContext('2d');
const target = $('target');
const statusEl = $('status');

// ---------- State ----------
let landmarker = null;
let lastVideoTime = -1;
let current = null; // latest {features, blink, landmarks}
let collecting = null; // {fx, fy, tag, list} while recording calibration samples
let calibrating = false;
let samples = []; // {f: number[], x: fraction, y: fraction, tag: 'cal'|'tap'}
let model = null; // {mu, sd, wx, wy, err}
let mode = 'dot';
let gaze = null; // smoothed {x, y} in CSS px
let trail = []; // {x, y, t}
let showPreview = false;
let fps = 0;
let lastFaceTime = 0;

const heat = document.createElement('canvas');
const hctx = heat.getContext('2d');

// =====================================================================
// Camera + model
// =====================================================================
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  });
  video.srcObject = stream;
  await video.play();
  await new Promise((r) => (video.readyState >= 2 ? r() : (video.onloadeddata = r)));
}

async function loadLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch (e) {
    console.warn('GPU delegate failed, using CPU', e);
    return await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}

// =====================================================================
// Features
// =====================================================================
// Landmark indices (MediaPipe face mesh, 478 points with irises).
// Corner pairs are ordered so both eye axes point the same way in the image.
const EYES = [
  { a: 33, b: 133, top: 159, bottom: 145, iris: 468 }, // subject's right eye
  { a: 362, b: 263, top: 386, bottom: 374, iris: 473 }, // subject's left eye
];

function eyeFeature(lm, e, W, H) {
  const P = (i) => [lm[i].x * W, lm[i].y * H];
  const A = P(e.a), B = P(e.b), I = P(e.iris), T = P(e.top), Bo = P(e.bottom);
  const ux = B[0] - A[0], uy = B[1] - A[1];
  const w = Math.hypot(ux, uy) || 1;
  const nx = ux / w, ny = uy / w;
  const mx = (A[0] + B[0]) / 2, my = (A[1] + B[1]) / 2;
  const dx = I[0] - mx, dy = I[1] - my;
  return {
    x: (dx * nx + dy * ny) / w, // along the eye axis
    y: (-dx * ny + dy * nx) / w, // perpendicular to it
    open: Math.hypot(T[0] - Bo[0], T[1] - Bo[1]) / w,
  };
}

function extractFeatures(res) {
  if (!res.faceLandmarks || !res.faceLandmarks.length) return null;
  const lm = res.faceLandmarks[0];
  if (lm.length < 478) return null;
  const W = video.videoWidth, H = video.videoHeight;
  const r = eyeFeature(lm, EYES[0], W, H);
  const l = eyeFeature(lm, EYES[1], W, H);

  const bs = {};
  const cats = res.faceBlendshapes?.[0]?.categories || [];
  for (const c of cats) bs[c.categoryName] = c.score;
  const g = (k) => bs[k] || 0;
  // Horizontal / vertical look from blendshapes (sign convention is irrelevant to regression).
  const h = (g('eyeLookOutLeft') + g('eyeLookInRight') - g('eyeLookInLeft') - g('eyeLookOutRight')) / 2;
  const v = (g('eyeLookUpLeft') + g('eyeLookUpRight') - g('eyeLookDownLeft') - g('eyeLookDownRight')) / 2;
  const blink = Math.max(g('eyeBlinkLeft'), g('eyeBlinkRight'));

  // Head pose: forward axis + translation from the facial transformation matrix (column-major 4x4).
  const m = res.facialTransformationMatrixes?.[0]?.data;
  const fx = m ? m[8] : 0, fy = m ? m[9] : 0;
  const tx = m ? m[12] / 10 : 0, ty = m ? m[13] / 10 : 0, tz = m ? m[14] / 10 : 0;

  const ex = (r.x + l.x) / 2, ey = (r.y + l.y) / 2, open = (r.open + l.open) / 2;
  const features = [ex, ey, open, h, v, fx, fy, tx, ty, tz, ex * ex, ey * ey, ex * ey, ex * fx, ey * fy];
  return { features, blink, landmarks: lm };
}

// =====================================================================
// Ridge regression (closed form, standardized features)
// =====================================================================
function solve(A, b) {
  // Gaussian elimination with partial pivoting; A is n×n, modified in place.
  const n = b.length;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
    [A[i], A[p]] = [A[p], A[i]];
    [b[i], b[p]] = [b[p], b[i]];
    const d = A[i][i] || 1e-12;
    for (let k = i + 1; k < n; k++) {
      const f = A[k][i] / d;
      if (!f) continue;
      for (let j = i; j < n; j++) A[k][j] -= f * A[i][j];
      b[k] -= f * b[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / (A[i][i] || 1e-12);
  }
  return x;
}

function fit(data, lambda = 2) {
  const d = data[0].f.length;
  const mu = new Array(d).fill(0), sd = new Array(d).fill(0);
  for (const s of data) s.f.forEach((v, i) => (mu[i] += v / data.length));
  for (const s of data) s.f.forEach((v, i) => (sd[i] += (v - mu[i]) ** 2 / data.length));
  for (let i = 0; i < d; i++) sd[i] = Math.sqrt(sd[i]) || 1;

  const n = d + 1; // + bias
  const XtX = Array.from({ length: n }, () => new Array(n).fill(0));
  const Xty = new Array(n).fill(0), Xtx = new Array(n).fill(0);
  for (const s of data) {
    const row = [1, ...s.f.map((v, i) => (v - mu[i]) / sd[i])];
    const wgt = s.tag === 'tap' ? 3 : 1; // fine-tune taps count a bit more
    for (let i = 0; i < n; i++) {
      Xtx[i] += wgt * row[i] * s.x;
      Xty[i] += wgt * row[i] * s.y;
      for (let j = 0; j < n; j++) XtX[i][j] += wgt * row[i] * row[j];
    }
  }
  for (let i = 1; i < n; i++) XtX[i][i] += lambda;
  const wx = solve(XtX.map((r) => r.slice()), Xtx);
  const wy = solve(XtX.map((r) => r.slice()), Xty);
  return { mu, sd, wx, wy };
}

function predict(m, f) {
  let x = m.wx[0], y = m.wy[0];
  for (let i = 0; i < f.length; i++) {
    const z = (f[i] - m.mu[i]) / m.sd[i];
    x += m.wx[i + 1] * z;
    y += m.wy[i + 1] * z;
  }
  return { x, y }; // fractions of screen width / height
}

// Honest accuracy estimate: leave one calibration point out, predict it from the rest.
function crossValidate(data) {
  const keys = [...new Set(data.map((s) => `${s.x.toFixed(4)},${s.y.toFixed(4)}`))];
  if (keys.length < 6) return null;
  const W = innerWidth, H = innerHeight;
  let sum = 0, cnt = 0;
  for (const k of keys) {
    const test = data.filter((s) => `${s.x.toFixed(4)},${s.y.toFixed(4)}` === k);
    const train = data.filter((s) => `${s.x.toFixed(4)},${s.y.toFixed(4)}` !== k);
    const m = fit(train);
    let px = 0, py = 0;
    for (const s of test) {
      const p = predict(m, s.f);
      px += p.x / test.length;
      py += p.y / test.length;
    }
    sum += Math.hypot((px - test[0].x) * W, (py - test[0].y) * H);
    cnt++;
  }
  return sum / cnt;
}

function refit() {
  if (samples.length < 20) return;
  const m = fit(samples);
  m.err = crossValidate(samples.filter((s) => s.tag === 'cal'));
  model = m;
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ samples, w: innerWidth, h: innerHeight }));
  } catch {}
}

// =====================================================================
// One-Euro filter (smooth when still, responsive when moving)
// =====================================================================
class LowPass {
  constructor() { this.y = null; }
  run(x, a) { this.y = this.y === null ? x : a * x + (1 - a) * this.y; return this.y; }
}
class OneEuro {
  constructor() { this.x = new LowPass(); this.dx = new LowPass(); this.t = null; this.prev = null; }
  alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  run(v, t, minCutoff, beta, dCutoff = 1) {
    if (this.t === null) { this.t = t; this.prev = v; return this.x.run(v, 1); }
    const dt = Math.max((t - this.t) / 1000, 1e-3);
    this.t = t;
    const d = this.dx.run((v - this.prev) / dt, this.alpha(dCutoff, dt));
    this.prev = v;
    return this.x.run(v, this.alpha(minCutoff + beta * Math.abs(d), dt));
  }
  reset() { this.x = new LowPass(); this.dx = new LowPass(); this.t = null; }
}
const filt = { x: new OneEuro(), y: new OneEuro() };

function smoothParams() {
  const s = Number($('smooth').value); // 0 = raw, 10 = very smooth
  return { minCutoff: 4 * Math.pow(0.68, s), beta: 0.006 };
}

// =====================================================================
// Main loop
// =====================================================================
let frames = 0, fpsT = performance.now();

function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();

  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    let res;
    try {
      res = landmarker.detectForVideo(video, now);
    } catch (e) {
      console.error(e);
      return;
    }
    current = extractFeatures(res);
    frames++;
    if (current) lastFaceTime = now;

    if (current && current.blink < 0.45) {
      if (collecting) collecting.list.push(current.features.slice());
      if (model && !calibrating) {
        const p = predict(model, current.features);
        const { minCutoff, beta } = smoothParams();
        const x = filt.x.run(p.x * innerWidth, now, minCutoff, beta);
        const y = filt.y.run(p.y * innerHeight, now, minCutoff, beta);
        gaze = {
          x: Math.min(innerWidth, Math.max(0, x)),
          y: Math.min(innerHeight, Math.max(0, y)),
        };
        trail.push({ x: gaze.x, y: gaze.y, t: now });
      }
    }
  }

  if (now - fpsT > 1000) {
    fps = Math.round((frames * 1000) / (now - fpsT));
    frames = 0;
    fpsT = now;
  }

  draw(now);
  if (showPreview) drawPreview();
  updateStatus(now);
}

function updateStatus(now) {
  if (!landmarker) return;
  const faceOk = now - lastFaceTime < 500;
  let msg;
  if (!faceOk) msg = 'Face not found — move into view';
  else if (calibrating) msg = 'Follow the dot with your eyes';
  else if (!model) msg = 'Face found · tap Calibrate';
  else msg = `${fps} fps${model.err ? ` · ~${Math.round(model.err)} px avg error` : ''}`;
  if (statusEl.textContent !== msg) statusEl.textContent = msg;
}

// =====================================================================
// Drawing
// =====================================================================
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  heat.width = Math.round(innerWidth / 2);
  heat.height = Math.round(innerHeight / 2);
}
addEventListener('resize', resize);
resize();

function drawDot(x, y, r, alpha = 1) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2.2);
  g.addColorStop(0, `rgba(124,242,200,${0.55 * alpha})`);
  g.addColorStop(1, 'rgba(124,242,200,0)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(x, y, r * 2.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = `rgba(124,242,200,${alpha})`;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = `rgba(255,255,255,${alpha})`;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
}

function draw(now) {
  ctx.clearRect(0, 0, innerWidth, innerHeight);
  // Keep the trail to the last few seconds.
  const keep = mode === 'trail' ? 4000 : 300;
  while (trail.length && now - trail[0].t > keep) trail.shift();

  if (!gaze || calibrating) return;
  const stale = now - lastFaceTime > 500;
  const alpha = stale ? 0.3 : 1;

  if (mode === 'trail') {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i - 1], b = trail[i];
      const age = (now - b.t) / keep;
      ctx.strokeStyle = `hsla(${(b.t / 12) % 360}, 90%, 65%, ${(1 - age) * 0.9})`;
      ctx.lineWidth = 2 + 8 * (1 - age);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    drawDot(gaze.x, gaze.y, 10, alpha);
  } else if (mode === 'heat') {
    // Slowly fade, then splat the current gaze point.
    hctx.globalCompositeOperation = 'destination-out';
    hctx.fillStyle = 'rgba(0,0,0,0.012)';
    hctx.fillRect(0, 0, heat.width, heat.height);
    if (!stale) {
      hctx.globalCompositeOperation = 'lighter';
      const hx = gaze.x / 2, hy = gaze.y / 2, r = 40;
      const g = hctx.createRadialGradient(hx, hy, 0, hx, hy, r);
      g.addColorStop(0, 'rgba(255,90,40,0.10)');
      g.addColorStop(0.5, 'rgba(255,180,40,0.04)');
      g.addColorStop(1, 'rgba(255,200,40,0)');
      hctx.fillStyle = g;
      hctx.beginPath();
      hctx.arc(hx, hy, r, 0, Math.PI * 2);
      hctx.fill();
    }
    ctx.drawImage(heat, 0, 0, innerWidth, innerHeight);
    drawDot(gaze.x, gaze.y, 8, alpha);
  } else {
    const r = Math.max(18, Math.min(60, (model?.err || 40) * 0.6));
    drawDot(gaze.x, gaze.y, r, alpha);
  }
}

function drawPreview() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return;
  const w = 264, h = Math.round((w * vh) / vw);
  if (preview.width !== w || preview.height !== h) {
    preview.width = w;
    preview.height = h;
  }
  pctx.drawImage(video, 0, 0, w, h);
  const lm = current?.landmarks;
  if (!lm) return;
  pctx.fillStyle = '#7cf2c8';
  for (const e of EYES) {
    for (const i of [e.a, e.b, e.top, e.bottom]) {
      pctx.fillRect(lm[i].x * w - 1.5, lm[i].y * h - 1.5, 3, 3);
    }
    pctx.beginPath();
    pctx.arc(lm[e.iris].x * w, lm[e.iris].y * h, 3, 0, Math.PI * 2);
    pctx.fillStyle = '#ff5a8a';
    pctx.fill();
    pctx.fillStyle = '#7cf2c8';
  }
}

// =====================================================================
// Calibration
// =====================================================================
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function showTarget(px, py, dur) {
  target.classList.remove('hidden');
  target.style.left = px + 'px';
  target.style.top = py + 'px';
  const ring = target.querySelector('.ring');
  ring.style.setProperty('--dur', dur + 'ms');
  ring.style.animation = 'none';
  void ring.offsetWidth; // restart animation
  ring.style.animation = '';
}

async function recordAt(fx, fy, tag, settleMs, recordMs) {
  showTarget(fx * innerWidth, fy * innerHeight, settleMs + recordMs);
  await sleep(settleMs);
  collecting = { list: [] };
  await sleep(recordMs);
  const list = collecting.list;
  collecting = null;
  // Drop the noisiest frames: keep those closest to the per-point median.
  if (list.length >= 6) {
    const med = list[0].map((_, i) => {
      const s = list.map((f) => f[i]).sort((a, b) => a - b);
      return s[s.length >> 1];
    });
    list.sort((a, b) => dist(a, med) - dist(b, med));
    list.length = Math.ceil(list.length * 0.75);
  }
  for (const f of list) samples.push({ f, x: fx, y: fy, tag });
  return list.length;
}
const dist = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);

async function calibrate() {
  if (calibrating) return;
  calibrating = true;
  setUI('calibrating');

  // Wait for a face.
  statusEl.textContent = 'Looking for your face…';
  while (performance.now() - lastFaceTime > 300) await sleep(100);

  const m = 0.07;
  const pts = [
    [0.5, 0.5],
    [m, m], [0.5, m], [1 - m, m],
    [m, 0.5], [1 - m, 0.5],
    [m, 1 - m], [0.5, 1 - m], [1 - m, 1 - m],
    [0.28, 0.28], [0.72, 0.28], [0.28, 0.72], [0.72, 0.72],
    [0.5, 0.25], [0.5, 0.75],
  ];
  // Center first, then a shuffled order so the eye has to jump.
  const rest = pts.slice(1).sort(() => Math.random() - 0.5);
  samples = [];
  for (const [x, y] of [pts[0], ...rest]) {
    await recordAt(x, y, 'cal', 750, 1000);
  }
  target.classList.add('hidden');

  if (samples.length < 40) {
    statusEl.textContent = 'Not enough data — check lighting and try again';
    calibrating = false;
    setUI('tracking');
    return;
  }
  refit();
  filt.x.reset();
  filt.y.reset();
  hctx.clearRect(0, 0, heat.width, heat.height);
  trail = [];
  calibrating = false;
  setUI('tracking');
}

// Tap anywhere while looking at that spot to add a fine-tuning point.
let tapping = false;
canvas.addEventListener('pointerdown', async (e) => {
  if (!model || calibrating || tapping) return;
  tapping = true;
  const fx = e.clientX / innerWidth, fy = e.clientY / innerHeight;
  // Keep only the most recent fine-tune points so old ones don't drag the fit.
  const tapPts = [...new Set(samples.filter((s) => s.tag === 'tap').map((s) => `${s.x},${s.y}`))];
  if (tapPts.length >= 12) {
    const drop = tapPts[0];
    samples = samples.filter((s) => !(s.tag === 'tap' && `${s.x},${s.y}` === drop));
  }
  await recordAt(fx, fy, 'tap', 250, 600);
  target.classList.add('hidden');
  refit();
  tapping = false;
});

// =====================================================================
// UI wiring
// =====================================================================
function setUI(state) {
  $('intro').classList.toggle('hidden', state !== 'intro');
  $('toolbar').classList.toggle('hidden', state !== 'tracking');
  $('hint').classList.toggle('hidden', state !== 'tracking' || !model);
  statusEl.classList.toggle('hidden', state === 'intro');
  preview.classList.toggle('hidden', !showPreview || state === 'intro');
}

async function boot(useSaved) {
  const msg = $('introMsg');
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera needs HTTPS. Open the https:// address printed by the server.');
    }
    $('startBtn').disabled = true;
    msg.textContent = 'Starting camera…';
    await startCamera();
    msg.textContent = 'Loading face model (first time takes a few seconds)…';
    landmarker = await loadLandmarker();
    requestAnimationFrame(loop);
    if (useSaved && savedSamples) {
      samples = savedSamples;
      refit();
      setUI('tracking');
    } else {
      setUI('tracking');
      calibrate();
    }
  } catch (e) {
    console.error(e);
    msg.textContent = e.name === 'NotAllowedError'
      ? 'Camera permission denied. Allow camera access for this site in Settings → Safari → Camera.'
      : `Error: ${e.message || e}`;
    $('startBtn').disabled = false;
  }
}

let savedSamples = null;
try {
  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
  if (saved?.samples?.length) {
    savedSamples = saved.samples;
    $('resumeBtn').classList.remove('hidden');
  }
} catch {}

$('startBtn').onclick = () => boot(false);
$('resumeBtn').onclick = () => boot(true);
$('calBtn').onclick = () => calibrate();
$('clearBtn').onclick = () => {
  trail = [];
  hctx.clearRect(0, 0, heat.width, heat.height);
};
$('previewBtn').onclick = () => {
  showPreview = !showPreview;
  preview.classList.toggle('hidden', !showPreview);
};
$('smooth').oninput = () => { filt.x.reset(); filt.y.reset(); };
for (const b of document.querySelectorAll('#modeSeg button')) {
  b.onclick = () => {
    mode = b.dataset.mode;
    for (const o of document.querySelectorAll('#modeSeg button')) o.classList.toggle('on', o === b);
  };
}
// Toolbar taps shouldn't count as fine-tune taps.
$('toolbar').addEventListener('pointerdown', (e) => e.stopPropagation());

setUI('intro');
