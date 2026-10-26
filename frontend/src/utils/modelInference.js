// Helper functions for normalization and inference

export function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 42) return null;

  const wristX = landmarks[0];
  const wristY = landmarks[1];

  const centered = new Float32Array(42);
  let sumSq = 0;

  for (let i = 0; i < 21; i++) {
    const x = landmarks[2 * i] - wristX;
    const y = landmarks[2 * i + 1] - wristY;
    centered[2 * i] = x;
    centered[2 * i + 1] = y;
    sumSq += x * x + y * y;
  }

  const norm = Math.sqrt(sumSq);
  if (norm > 0) {
    for (let i = 0; i < 42; i++) {
      centered[i] /= norm;
    }
  }

  return centered;
}

// In-Browser Neural Net Inference (MLP: 42 -> 128 -> 64 -> 3)
export function predictInBrowser(landmarks, weightsData) {
  const normInput = normalizeLandmarks(landmarks);
  if (!normInput || !weightsData) return null;

  const { coefs, intercepts, labels } = weightsData;

  // Layer 1: 42 -> 128 (ReLU)
  const l1_w = coefs[0];
  const l1_b = intercepts[0];
  const l1_out = new Float32Array(128);

  for (let j = 0; j < 128; j++) {
    let sum = l1_b[j];
    for (let i = 0; i < 42; i++) {
      sum += normInput[i] * l1_w[i][j];
    }
    l1_out[j] = Math.max(0, sum); // ReLU
  }

  // Layer 2: 128 -> 64 (ReLU)
  const l2_w = coefs[1];
  const l2_b = intercepts[1];
  const l2_out = new Float32Array(64);

  for (let j = 0; j < 64; j++) {
    let sum = l2_b[j];
    for (let i = 0; i < 128; i++) {
      sum += l1_out[i] * l2_w[i][j];
    }
    l2_out[j] = Math.max(0, sum); // ReLU
  }

  // Layer 3: 64 -> 3 (Softmax)
  const l3_w = coefs[2];
  const l3_b = intercepts[2];
  const logits = new Float32Array(3);

  let maxLogit = -Infinity;
  for (let j = 0; j < 3; j++) {
    let sum = l3_b[j];
    for (let i = 0; i < 64; i++) {
      sum += l2_out[i] * l3_w[i][j];
    }
    logits[j] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }

  // Softmax
  let expSum = 0;
  const probs = new Float32Array(3);
  for (let j = 0; j < 3; j++) {
    probs[j] = Math.exp(logits[j] - maxLogit);
    expSum += probs[j];
  }
  for (let j = 0; j < 3; j++) {
    probs[j] /= expSum;
  }

  // Find max class
  let maxIdx = 0;
  for (let j = 1; j < 3; j++) {
    if (probs[j] > probs[maxIdx]) maxIdx = j;
  }

  const rawLabel = labels[maxIdx];
  const labelMap = {
    Stone: 'Rock',
    Paper: 'Paper',
    Scissor: 'Scissors'
  };

  const probabilities = {};
  labels.forEach((lbl, idx) => {
    const moveName = labelMap[lbl] || lbl;
    probabilities[moveName] = probs[idx];
  });

  return {
    prediction: labelMap[rawLabel] || rawLabel,
    raw_label: rawLabel,
    confidence: probs[maxIdx],
    probabilities
  };
}

const BACKEND_URL = 'http://localhost:8000';

export async function predictWithBackend(landmarks) {
  try {
    const response = await fetch(`${BACKEND_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks })
    });
    if (!response.ok) throw new Error('Backend error');
    return await response.json();
  } catch (err) {
    console.warn('Backend unavailable, using fallback', err);
    return null;
  }
}

export async function playRoundWithBackend(landmarks, playerMove, difficulty = 'normal') {
  try {
    const response = await fetch(`${BACKEND_URL}/api/play`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks, player_move: playerMove, difficulty })
    });
    if (!response.ok) throw new Error('Backend error');
    return await response.json();
  } catch (err) {
    console.warn('Backend unavailable for play endpoint', err);
    return null;
  }
}

export async function fetchModelInfo() {
  try {
    const res = await fetch(`${BACKEND_URL}/api/info`);
    if (res.ok) return await res.json();
  } catch (e) {
    console.warn('Could not fetch backend info', e);
  }
  return null;
}
