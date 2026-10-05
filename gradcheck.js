// Numeric gradient check: compare analytic backprop against finite differences.
const fs = require("fs");
const path = require("path");

// --- replicate the exact model math from train.js on a tiny random net ---
const F = 4, H1 = 5, H2 = 3, NOUT = 3;
const layers = [
  { W: Float32Array.from({ length: F * H1 }, () => Math.random() * 0.5 - 0.25), b: Float32Array.from({ length: H1 }, () => Math.random() * 0.1) },
  { W: Float32Array.from({ length: H1 * H2 }, () => Math.random() * 0.5 - 0.25), b: Float32Array.from({ length: H2 }, () => Math.random() * 0.1) },
  { W: Float32Array.from({ length: H2 * NOUT }, () => Math.random() * 0.5 - 0.25), b: Float32Array.from({ length: NOUT }, () => Math.random() * 0.1) },
];
layers[0].out = H1; layers[1].out = H2; layers[2].out = NOUT;
const featCount = F;

function relu(v) { return v > 0 ? v : 0; }
function softmax(z) {
  const o = new Float32Array(z.length);
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  let s = 0;
  for (let i = 0; i < z.length; i++) { o[i] = Math.exp(z[i] - m); s += o[i]; }
  for (let i = 0; i < z.length; i++) o[i] /= s;
  return o;
}
function forward(x) {
  const acts = [x];
  let a = x;
  for (let li = 0; li < 3; li++) {
    const { W, b } = layers[li];
    const nIn = li === 0 ? featCount : layers[li - 1].out;
    const out = li === 2 ? NOUT : layers[li].out;
    const z = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = b[o];
      for (let i = 0; i < nIn; i++) s += a[i] * W[i * out + o];
      z[o] = s;
    }
    const next = li === 2 ? softmax(z) : (() => {
      const t = new Float32Array(out);
      for (let o = 0; o < out; o++) t[o] = relu(z[o]);
      return t;
    })();
    acts.push({ z, next, nIn, out });
    a = next;
  }
  return a;
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
function backward(x, target) {
  const acts = [x];
  let a = x;
  for (let li = 0; li < 3; li++) {
    const { W, b } = layers[li];
    const nIn = li === 0 ? featCount : layers[li - 1].out;
    const out = li === 2 ? NOUT : layers[li].out;
    const z = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = b[o];
      for (let i = 0; i < nIn; i++) s += a[i] * W[i * out + o];
      z[o] = s;
    }
    const next = li === 2 ? softmax(z) : (() => {
      const t = new Float32Array(out);
      for (let o = 0; o < out; o++) t[o] = relu(z[o]);
      return t;
    })();
    acts.push({ z, next, nIn, out });
    a = next;
  }
  const d3 = new Float32Array(NOUT);
  for (let o = 0; o < NOUT; o++) d3[o] = acts[3].next[o] - (o === target ? 1 : 0);
  const grads = [];
  grads.push(gradFor(acts[2].next, d3, acts[2].nIn, NOUT));
  const d2 = new Float32Array(H2);
  for (let i = 0; i < H2; i++) {
    let s = 0;
    for (let o = 0; o < NOUT; o++) s += layers[2].W[i * NOUT + o] * d3[o];
    d2[i] = s * (acts[2].z[i] > 0 ? 1 : 0);
  }
  grads.push(gradFor(acts[1].next, d2, acts[1].nIn, H2));
  const d1 = new Float32Array(H1);
  for (let i = 0; i < H1; i++) {
    let s = 0;
    for (let o = 0; o < H2; o++) s += layers[1].W[i * H2 + o] * d2[o];
    d1[i] = s * (acts[1].z[i] > 0 ? 1 : 0);
  }
  grads.push(gradFor(acts[0], d1, featCount, H1));
  return { probs: acts[3].next, grads };
}

function lossOf(x, target) {
  const p = forward(x);
  return -Math.log(Math.max(p[target], 1e-12));
}

const x = Float32Array.from({ length: F }, () => Math.random());
const target = 1;
const { grads } = backward(x, target);

// grads[0] = layer3, grads[1] = layer2, grads[2] = layer1
const eps = 1e-3;
console.log("layer | max|analytic grad| | max error vs numeric |");
for (let li = 0; li < 3; li++) {
  const g = grads[2 - li];
  const { W } = layers[li];
  let maxA = 0;
  for (const v of g.gW) maxA = Math.max(maxA, Math.abs(v));
  let maxErr = 0;
  for (let trial = 0; trial < 6; trial++) {
    const k = Math.floor(Math.random() * W.length);
    const orig = W[k];
    W[k] = orig + eps;
    const lp = lossOf(x, target);
    W[k] = orig - eps;
    const lm = lossOf(x, target);
    W[k] = orig;
    const numeric = (lp - lm) / (2 * eps);
    maxErr = Math.max(maxErr, Math.abs(numeric - g.gW[k]));
    if (trial === 0)
      console.log(`  L${li + 1} idx ${k}: analytic ${g.gW[k].toFixed(6)} numeric ${numeric.toFixed(6)}`);
  }
  console.log(`L${li + 1}: max|analytic|=${maxA.toFixed(6)}  maxErr=${maxErr.toFixed(6)}\n`);
}
