// Dependency-free MLP trainer for the rock-paper-scissors hand-landmark CSV.
// Input: 42 features (21 landmarks x,y normalized) -> 3 classes.
const fs = require("fs");
const path = require("path");

const SEED = 42;
const HIDDEN1 = 64;
const HIDDEN2 = 32;
const EPOCHS = parseInt(process.env.EPOCHS || "25", 10);
const LR = 0.01;
const BATCH = 64;

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
function randn() {
  // Box-Muller
  let u = 0,
    v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---- data ----
function loadCSV(file) {
  const text = fs.readFileSync(file, "utf8");
  const lines = text.split("\n");
  const header = lines[0].split(",");
  const labelIndex = header.indexOf("Category");
  if (labelIndex < 0) throw new Error("no Category column found");
  const featCount = labelIndex;
  const X = [];
  const Y = [];
  const classes = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line || line.length < 10) continue;
    const parts = line.split(",");
    if (parts.length !== header.length) continue;
    const feat = new Float32Array(featCount);
    for (let f = 0; f < featCount; f++) feat[f] = parseFloat(parts[f]);
    if (feat.some((v) => !isFinite(v))) continue;
    const label = parts[labelIndex].trim();
    let y = classes.indexOf(label);
    if (y < 0) {
      y = classes.length;
      classes.push(label);
    }
    X.push(feat);
    Y.push(y);
  }
  return { X, Y, classes, featCount };
}

const { X, Y, classes, featCount } = loadCSV(path.join(__dirname, "data.csv"));
console.log(
  `loaded ${X.length} samples, ${featCount} features, classes: ${classes.join(", ")}`,
);

// ---- standardization ----
const mean = new Float64Array(featCount);
const std = new Float64Array(featCount);
for (const x of X) for (let j = 0; j < featCount; j++) mean[j] += x[j];
for (let j = 0; j < featCount; j++) mean[j] /= X.length;
for (const x of X)
  for (let j = 0; j < featCount; j++) std[j] += (x[j] - mean[j]) ** 2;
for (let j = 0; j < featCount; j++) std[j] = Math.sqrt(std[j] / X.length) || 1;
const Xn = X.map((x) => {
  const z = new Float32Array(featCount);
  for (let j = 0; j < featCount; j++) z[j] = (x[j] - mean[j]) / std[j];
  return z;
});

// ---- shuffle + split ----
const idx = Xn.map((_, i) => i);
for (let i = idx.length - 1; i > 0; i--) {
  const j = Math.floor(rand() * (i + 1));
  [idx[i], idx[j]] = [idx[j], idx[i]];
}
const split = Math.floor(idx.length * 0.8);
const trainIdx = idx.slice(0, split);
const testIdx = idx.slice(split);
const nOut = classes.length;

// ---- model ----
function initWeights(nIn, nOut, scale) {
  const W = new Float32Array(nIn * nOut);
  const b = new Float32Array(nOut);
  for (let i = 0; i < W.length; i++) W[i] = randn() * scale;
  return { W, b };
}
const L1 = initWeights(featCount, HIDDEN1, Math.sqrt(2 / featCount));
const L2 = initWeights(HIDDEN1, HIDDEN2, Math.sqrt(2 / HIDDEN1));
const L3 = initWeights(HIDDEN2, nOut, Math.sqrt(2 / HIDDEN2));
L1.out = HIDDEN1;
L2.out = HIDDEN2;
L3.out = nOut;

function zeros(n) {
  return new Float32Array(n);
}
// Adam state
const mW = [zeros(L1.W.length), zeros(L2.W.length), zeros(L3.W.length)];
const vW = [zeros(L1.W.length), zeros(L2.W.length), zeros(L3.W.length)];
const mb = [zeros(L1.b.length), zeros(L2.b.length), zeros(L3.b.length)];
const vb = [zeros(L1.b.length), zeros(L2.b.length), zeros(L3.b.length)];
let step = 0;

const layers = [L1, L2, L3];

function forward(x) {
  const acts = [x];
  let a = x;
  for (let li = 0; li < 3; li++) {
    const { W, b } = layers[li];
    const nIn = li === 0 ? featCount : layers[li - 1].out;
    const out = li === 2 ? nOut : layers[li].out;
    const z = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = b[o];
      for (let i = 0; i < nIn; i++) s += a[i] * W[i * out + o];
      z[o] = s;
    }
    const next =
      li === 2
        ? softmax(z)
        : (() => {
            const t = new Float32Array(out);
            for (let o = 0; o < out; o++) t[o] = relu(z[o]);
            return t;
          })();
    acts.push({ z, next, nIn, out });
    a = next;
  }
  return a; // softmax probabilities
}
function relu(v) {
  return v > 0 ? v : 0;
}
function softmax(z) {
  const o = new Float32Array(z.length);
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  let s = 0;
  for (let i = 0; i < z.length; i++) {
    o[i] = Math.exp(z[i] - m);
    s += o[i];
  }
  for (let i = 0; i < z.length; i++) o[i] /= s;
  return o;
}

function backward(x, target) {
  const acts = [x];
  let a = x;
  for (let li = 0; li < 3; li++) {
    const { W, b } = layers[li];
    const nIn = li === 0 ? featCount : layers[li - 1].out;
    const out = li === 2 ? nOut : layers[li].out;
    const z = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = b[o];
      for (let i = 0; i < nIn; i++) s += a[i] * W[i * out + o];
      z[o] = s;
    }
    const next =
      li === 2
        ? softmax(z)
        : (() => {
            const t = new Float32Array(out);
            for (let o = 0; o < out; o++) t[o] = relu(z[o]);
            return t;
          })();
    acts.push({ z, next, nIn, out });
    a = next;
  }
  // output delta (acts[3].next holds the softmax probabilities)
  const d3 = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++)
    d3[o] = acts[3].next[o] - (o === target ? 1 : 0);

  const grads = [];
  // layer 3 (output layer)
  grads.push(gradFor(acts[2].next, d3, acts[2].nIn, nOut));
  // d for hidden2 = W3^T * d3, then relu'(z2)
  const d2 = new Float32Array(HIDDEN2);
  for (let i = 0; i < HIDDEN2; i++) {
    let s = 0;
    for (let o = 0; o < nOut; o++) s += L3.W[i * nOut + o] * d3[o];
    d2[i] = s * (acts[2].z[i] > 0 ? 1 : 0);
  }
  // layer 2
  grads.push(gradFor(acts[1].next, d2, acts[1].nIn, HIDDEN2));
  const d1 = new Float32Array(HIDDEN1);
  for (let i = 0; i < HIDDEN1; i++) {
    let s = 0;
    for (let o = 0; o < HIDDEN2; o++) s += L2.W[i * HIDDEN2 + o] * d2[o];
    d1[i] = s * (acts[1].z[i] > 0 ? 1 : 0);
  }
  // layer 1
  grads.push(gradFor(acts[0], d1, featCount, HIDDEN1));
  return { probs: acts[3].next, grads };
}

function gradFor(a, delta, nIn, nOut) {
  const gW = new Float32Array(nIn * nOut);
  const gb = new Float32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    gb[o] = delta[o];
    for (let i = 0; i < nIn; i++) gW[i * nOut + o] += a[i] * delta[o];
  }
  return { gW, gb };
}

const B1 = 0.9,
  B2 = 0.999,
  EPS = 1e-8;
function adamUpdate() {
  step++;
  for (let li = 0; li < 3; li++) {
    const { W, b } = layers[li];
    const g = G[li];
    for (let i = 0; i < W.length; i++) {
      mW[li][i] = B1 * mW[li][i] + (1 - B1) * g.gW[i];
      vW[li][i] = B2 * vW[li][i] + (1 - B2) * g.gW[i] * g.gW[i];
      W[i] -=
        (LR * (mW[li][i] / (1 - Math.pow(B1, step)))) /
        (Math.sqrt(vW[li][i] / (1 - Math.pow(B2, step))) + EPS);
    }
    for (let i = 0; i < b.length; i++) {
      mb[li][i] = B1 * mb[li][i] + (1 - B1) * g.gb[i];
      vb[li][i] = B2 * vb[li][i] + (1 - B2) * g.gb[i] * g.gb[i];
      b[i] -=
        (LR * (mb[li][i] / (1 - Math.pow(B1, step)))) /
        (Math.sqrt(vb[li][i] / (1 - Math.pow(B2, step))) + EPS);
    }
  }
}

let G = null;
function accumulate(indices) {
  G = [
    { gW: zeros(L1.W.length), gb: zeros(L1.b.length) },
    { gW: zeros(L2.W.length), gb: zeros(L2.b.length) },
    { gW: zeros(L3.W.length), gb: zeros(L3.b.length) },
  ];
  for (const i of indices) {
    const { grads } = backward(Xn[i], Y[i]);
    // grads come back output-layer-first; store them in layer order
    for (let li = 0; li < 3; li++) {
      const g = grads[2 - li],
        acc = G[li];
      for (let k = 0; k < acc.gW.length; k++) acc.gW[k] += g.gW[k];
      for (let k = 0; k < acc.gb.length; k++) acc.gb[k] += g.gb[k];
    }
  }
  const n = indices.length || 1;
  for (const acc of G) {
    for (let k = 0; k < acc.gW.length; k++) acc.gW[k] /= n;
    for (let k = 0; k < acc.gb.length; k++) acc.gb[k] /= n;
  }
}

function predict(x) {
  const probs = forward(x);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { cls: best, conf: probs[best] };
}

function evaluate(indices) {
  let correct = 0,
    loss = 0;
  const conf = new Array(nOut).fill(0).map(() => new Array(nOut).fill(0));
  for (const i of indices) {
    const { probs } = backward(Xn[i], Y[i]);
    loss -= Math.log(Math.max(probs[Y[i]], 1e-9));
    let best = 0;
    for (let k = 1; k < nOut; k++) if (probs[k] > probs[best]) best = k;
    if (best === Y[i]) correct++;
    conf[Y[i]][best]++;
  }
  return { acc: correct / indices.length, loss: loss / indices.length, conf };
}

function printConfusion(conf) {
  const order = [2, 1, 0]; // map to Paper/Scissor/Stone
  console.log("  confusion (rows=true, cols=pred):");
  for (let i = 0; i < nOut; i++) {
    console.log(
      `    ${classes[i].padEnd(8)} ${order.map((j) => String(conf[i][j]).padStart(6)).join("")}`,
    );
  }
}

console.log(`training on ${trainIdx.length}, testing on ${testIdx.length}`);
const t0 = Date.now();
for (let epoch = 0; epoch < EPOCHS; epoch++) {
  // shuffle train each epoch
  for (let i = trainIdx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [trainIdx[i], trainIdx[j]] = [trainIdx[j], trainIdx[i]];
  }
  for (let s = 0; s < trainIdx.length; s += BATCH) {
    accumulate(trainIdx.slice(s, s + BATCH));
    adamUpdate();
  }
  const tr = evaluate(trainIdx);
  const te = evaluate(testIdx);
  console.log(
    `epoch ${String(epoch + 1).padStart(2)}  train acc ${(tr.acc * 100).toFixed(2)}%  loss ${tr.loss.toFixed(4)}  |  test acc ${(te.acc * 100).toFixed(2)}%  loss ${te.loss.toFixed(4)}`,
  );
}
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
const final = evaluate(testIdx);
printConfusion(final.conf);

const model = {
  classes,
  featCount,
  layers: [
    { W: Array.from(L1.W), b: Array.from(L1.b) },
    { W: Array.from(L2.W), b: Array.from(L2.b) },
    { W: Array.from(L3.W), b: Array.from(L3.b) },
  ],
  mean: Array.from(mean),
  std: Array.from(std),
  testAcc: final.acc,
};
fs.writeFileSync(path.join(__dirname, "model.json"), JSON.stringify(model));
console.log(`wrote model.json (test acc ${(final.acc * 100).toFixed(2)}%)`);
