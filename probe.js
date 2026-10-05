// Does the training loop actually change weights? Probe a real mini-batch.
const fs = require("fs");
const path = require("path");
const src = fs.readFileSync(path.join(__dirname, "train.js"), "utf8");

// Instrument: inject a probe right after the model is built and run 1 epoch.
let code = src.replace(
  "console.log(`training on ${trainIdx.length}",
  `
  // PROBE
  {
    const w0 = L1.W[0], b0 = L1.b[0];
    const g = (() => { accumulate([0,1,2,3]); return G; })();
    console.log("PROBE grad L1.gW[0] =", g[0].gW[0], " grad L1.gb[0] =", g[0].gb[0]);
    console.log("PROBE grad L3.gW[0] =", g[2].gW[0], " grad L3.gb[0] =", g[2].gb[0]);
    console.log("PROBE L1.W[0] before =", w0);
    adamUpdate();
    console.log("PROBE L1.W[0] after  =", L1.W[0], " delta =", L1.W[0] - w0);
    console.log("PROBE L1.b[0] delta =", L1.b[0] - b0);
    // what does a single sample's softmax look like?
    const p = forward(Xn[0]);
    console.log("PROBE sample0 probs =", Array.from(p), " label =", Y[0]);
    const act1 = forward(Xn[0]);
  }
  process.exit(0);
  console.log(\`training on \${trainIdx.length}`,
);
fs.writeFileSync(path.join(__dirname, "_probe.js"), code);
require(path.join(__dirname, "_probe.js"));
