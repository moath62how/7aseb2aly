import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import {
  Trophy,
  Play,
  RotateCcw,
  BarChart2,
  Brain,
  Sparkles,
  Shield,
  Zap,
  Flame,
  Volume2,
  VolumeX,
  Radio
} from 'lucide-react';
import CameraFeed from './components/CameraFeed';
import StatsModal from './components/StatsModal';
import {
  predictInBrowser,
  predictWithBackend,
  playRoundWithBackend,
  fetchModelInfo
} from './utils/modelInference';

const MOVES = {
  Rock: { emoji: '🪨', color: '#f59e0b', beats: 'Scissors' },
  Paper: { emoji: '📄', color: '#06b6d4', beats: 'Rock' },
  Scissors: { emoji: '✂️', color: '#10b981', beats: 'Paper' }
};

export default function App() {
  const [landmarks, setLandmarks] = useState(null);
  const [weightsData, setWeightsData] = useState(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  
  // Game State
  const [scores, setScores] = useState({ player: 0, ai: 0, ties: 0 });
  const [streak, setStreak] = useState(0);
  const [difficulty, setDifficulty] = useState('normal'); // 'normal', 'smart', 'impossible'
  const [difficultyOpen, setDifficultyOpen] = useState(false);

  // Live and Round State
  const [livePrediction, setLivePrediction] = useState(null);
  const [roundResult, setRoundResult] = useState(null); // { player_move, ai_move, result, message, confidence }
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [statsModalOpen, setStatsModalOpen] = useState(false);

  const landmarksRef = useRef(null);
  landmarksRef.current = landmarks;

  // Load model weights & check backend on startup
  useEffect(() => {
    fetch('/model_weights.json')
      .then((res) => res.json())
      .then((data) => setWeightsData(data))
      .catch((err) => console.error('Failed to load local model weights:', err));

    fetchModelInfo().then((info) => {
      if (info) {
        setBackendOnline(true);
        setModelInfo(info);
      }
    });
  }, []);

  // Real-time live prediction on landmark update
  useEffect(() => {
    if (!landmarks) {
      setLivePrediction(null);
      return;
    }

    let isMounted = true;

    // Try in-browser fast predictor first for instant UX
    if (weightsData) {
      const fastPred = predictInBrowser(landmarks, weightsData);
      if (fastPred && isMounted) {
        setLivePrediction(fastPred);
      }
    }

    // Optionally sync with Python backend
    predictWithBackend(landmarks).then((res) => {
      if (res && isMounted) {
        setLivePrediction(res);
        setBackendOnline(true);
      }
    });

    return () => { isMounted = false; };
  }, [landmarks, weightsData]);

  // Audio effects
  const playSound = (type) => {
    if (!soundEnabled) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);

      if (type === 'beep') {
        osc.frequency.value = 440;
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15);
        osc.start();
        osc.stop(ctx.currentTime + 0.15);
      } else if (type === 'shoot') {
        osc.frequency.setValueAtTime(300, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(800, ctx.currentTime + 0.25);
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
        osc.start();
        osc.stop(ctx.currentTime + 0.25);
      } else if (type === 'win') {
        osc.frequency.setValueAtTime(523.25, ctx.currentTime);
        osc.frequency.setValueAtTime(659.25, ctx.currentTime + 0.1);
        osc.frequency.setValueAtTime(783.99, ctx.currentTime + 0.2);
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
        osc.start();
        osc.stop(ctx.currentTime + 0.4);
      }
    } catch (e) {
      // Audio fallback silent ignore
    }
  };

  // Start Play Round (3..2..1.. SHOOT!)
  const startRound = () => {
    if (isCountingDown) return;
    setIsCountingDown(true);
    setRoundResult(null);
    setCountdown(3);
    playSound('beep');

    let count = 3;
    const timer = setInterval(() => {
      count -= 1;
      if (count > 0) {
        setCountdown(count);
        playSound('beep');
      } else if (count === 0) {
        setCountdown('SHOOT!');
        playSound('shoot');
        executeShoot();
        clearInterval(timer);
      }
    }, 900);
  };

  const executeShoot = async () => {
    setTimeout(async () => {
      const currentLandmarks = landmarksRef.current;
      let detectedMove = 'Rock';
      let confidence = 0.9;
      let probabilities = {};

      if (currentLandmarks && weightsData) {
        const localPred = predictInBrowser(currentLandmarks, weightsData);
        if (localPred) {
          detectedMove = localPred.prediction;
          confidence = localPred.confidence;
          probabilities = localPred.probabilities;
        }
      }

      // Try Backend Play Endpoint
      let resultData = null;
      if (backendOnline) {
        resultData = await playRoundWithBackend(currentLandmarks, detectedMove, difficulty);
      }

      if (!resultData) {
        // Fallback local play logic
        const moves = ['Rock', 'Paper', 'Scissors'];
        let aiMove = moves[Math.floor(Math.random() * moves.length)];
        
        if (difficulty === 'impossible') {
          aiMove = MOVES[detectedMove].beats === 'Scissors' ? 'Paper' : (MOVES[detectedMove].beats === 'Paper' ? 'Scissors' : 'Rock');
        }

        let res = 'draw';
        let msg = "It's a Tie!";
        if (detectedMove === aiMove) {
          res = 'draw';
        } else if (MOVES[detectedMove].beats === aiMove) {
          res = 'win';
          msg = `${detectedMove} beats ${aiMove}. You Win!`;
        } else {
          res = 'lose';
          msg = `${aiMove} beats ${detectedMove}. AI Wins!`;
        }

        resultData = {
          player_move: detectedMove,
          ai_move: aiMove,
          result: res,
          message: msg,
          confidence,
          probabilities
        };
      }

      setRoundResult(resultData);
      setIsCountingDown(false);

      // Update Scores & Streaks
      if (resultData.result === 'win') {
        setScores((prev) => ({ ...prev, player: prev.player + 1 }));
        setStreak((prev) => prev + 1);
        playSound('win');
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 }
        });
      } else if (resultData.result === 'lose') {
        setScores((prev) => ({ ...prev, ai: prev.ai + 1 }));
        setStreak(0);
      } else {
        setScores((prev) => ({ ...prev, ties: prev.ties + 1 }));
      }
    }, 400);
  };

  const resetScores = () => {
    setScores({ player: 0, ai: 0, ties: 0 });
    setStreak(0);
    setRoundResult(null);
  };

  return (
    <div style={{ maxWidth: '1180px', margin: '0 auto', padding: '24px 20px 40px' }}>
      {/* Top Navbar */}
      <header
        className="glass-panel"
        style={{
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '24px'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: 'var(--radius-sm)',
              background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              fontWeight: 800,
              fontSize: '1.2rem',
              boxShadow: '0 4px 12px rgba(6,182,212,0.4)'
            }}
          >
            RPS
          </div>
          <div>
            <h1 style={{ fontSize: '1.2rem', fontWeight: 800, letterSpacing: '0.04em' }}>
              ROCK <span style={{ color: 'var(--accent-cyan)' }}>•</span> PAPER <span style={{ color: 'var(--accent-emerald)' }}>•</span> SCISSORS
            </h1>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              MediaPipe Hand Landmarks & ML Classifier
            </p>
          </div>
        </div>

        {/* Status Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid var(--bg-card-border)',
              borderRadius: 'var(--radius-full)',
              padding: '6px 14px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '0.82rem'
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: backendOnline ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                boxShadow: backendOnline ? '0 0 8px var(--accent-emerald)' : 'none'
              }}
            />
            <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>
              {backendOnline ? 'Python FastAPI Active' : 'In-Browser JS Model'}
            </span>
          </div>

          <button
            className="btn btn-secondary"
            onClick={() => setStatsModalOpen(true)}
            style={{ padding: '8px 14px', fontSize: '0.84rem' }}
          >
            <BarChart2 size={16} />
            <span>98.87% Accuracy</span>
          </button>

          <button
            className="btn btn-secondary"
            onClick={() => setSoundEnabled(!soundEnabled)}
            style={{ padding: '8px', borderRadius: '50%' }}
            title={soundEnabled ? 'Mute Sound' : 'Enable Sound'}
          >
            {soundEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.15fr 1fr', gap: '24px' }}>
        {/* Left Column: Live Camera & Hand Tracker */}
        <CameraFeed
          onLandmarksDetected={setLandmarks}
          livePrediction={livePrediction}
          countdown={countdown}
          isCountingDown={isCountingDown}
        />

        {/* Right Column: Battle Arena & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Scoreboard Card */}
          <div className="glass-panel" style={{ padding: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Trophy size={18} style={{ color: 'var(--accent-amber)' }} />
                <span style={{ fontWeight: 700, fontSize: '0.95rem' }}>SCOREBOARD</span>
              </div>
              {streak > 1 && (
                <div
                  style={{
                    background: 'rgba(245, 158, 11, 0.15)',
                    border: '1px solid rgba(245, 158, 11, 0.3)',
                    color: 'var(--accent-amber)',
                    borderRadius: 'var(--radius-full)',
                    padding: '3px 10px',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  <Flame size={14} /> {streak} Win Streak!
                </div>
              )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', textAlign: 'center' }}>
              <div style={{ background: 'rgba(6, 182, 212, 0.08)', border: '1px solid rgba(6, 182, 212, 0.2)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px' }}>YOU</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>{scores.player}</div>
              </div>
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid var(--bg-card-border)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px' }}>TIES</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-main)' }}>{scores.ties}</div>
              </div>
              <div style={{ background: 'rgba(244, 63, 94, 0.08)', border: '1px solid rgba(244, 63, 94, 0.2)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px' }}>AI</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-rose)' }}>{scores.ai}</div>
              </div>
            </div>
          </div>

          {/* Battle Result & Move Cards */}
          <div className="glass-panel" style={{ padding: '24px', flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {roundResult ? (
              <div style={{ textAlign: 'center' }}>
                <div
                  style={{
                    fontSize: '1.1rem',
                    fontWeight: 800,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    marginBottom: '20px',
                    color:
                      roundResult.result === 'win'
                        ? 'var(--accent-emerald)'
                        : roundResult.result === 'lose'
                        ? 'var(--accent-rose)'
                        : 'var(--accent-amber)'
                  }}
                >
                  {roundResult.message}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-around', margin: '20px 0' }}>
                  {/* Player Move */}
                  <div style={{ textAlign: 'center' }}>
                    <div
                      style={{
                        width: '76px',
                        height: '76px',
                        borderRadius: '50%',
                        background: 'rgba(6, 182, 212, 0.15)',
                        border: '2px solid var(--accent-cyan)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '2.5rem',
                        margin: '0 auto 8px',
                        boxShadow: '0 0 20px rgba(6,182,212,0.3)'
                      }}
                    >
                      {MOVES[roundResult.player_move]?.emoji || '✋'}
                    </div>
                    <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{roundResult.player_move}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Player</div>
                  </div>

                  <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-dim)' }}>VS</div>

                  {/* AI Move */}
                  <div style={{ textAlign: 'center' }}>
                    <div
                      style={{
                        width: '76px',
                        height: '76px',
                        borderRadius: '50%',
                        background: 'rgba(244, 63, 94, 0.15)',
                        border: '2px solid var(--accent-rose)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '2.5rem',
                        margin: '0 auto 8px',
                        boxShadow: '0 0 20px rgba(244,63,94,0.3)'
                      }}
                    >
                      {MOVES[roundResult.ai_move]?.emoji || '🤖'}
                    </div>
                    <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{roundResult.ai_move}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>AI Opponent</div>
                  </div>
                </div>

                {/* Probability Distribution */}
                {roundResult.probabilities && (
                  <div style={{ marginTop: '20px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--bg-card-border)' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Model Confidence Breakdown
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                      {Object.entries(roundResult.probabilities).map(([move, prob]) => (
                        <div key={move} style={{ fontSize: '0.78rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>{move}: </span>
                          <span className="mono" style={{ fontWeight: 600, color: move === roundResult.player_move ? 'var(--accent-cyan)' : 'var(--text-main)' }}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '30px 10px' }}>
                <Brain size={48} style={{ color: 'var(--accent-cyan)', opacity: 0.6, marginBottom: '14px' }} />
                <h3 style={{ color: 'var(--text-main)', fontWeight: 700, marginBottom: '6px' }}>Ready to Battle?</h3>
                <p style={{ fontSize: '0.88rem', maxWidth: '300px', margin: '0 auto' }}>
                  Press <strong style={{ color: 'var(--accent-cyan)' }}>PLAY ROUND</strong>, hold up your hand, and show your move on SHOOT!
                </p>
              </div>
            )}
          </div>

          {/* Action Bar */}
          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              className="btn btn-primary"
              onClick={startRound}
              disabled={isCountingDown}
              style={{ flex: 2, padding: '14px 24px', fontSize: '1rem' }}
            >
              <Play size={18} />
              <span>{isCountingDown ? 'Counting Down...' : 'PLAY ROUND'}</span>
            </button>

            {/* Difficulty Selector */}
            <div style={{ relative: 'relative', flex: 1 }}>
              <button
                className="btn btn-secondary"
                onClick={() => setDifficultyOpen(!difficultyOpen)}
                style={{ width: '100%', padding: '14px', textTransform: 'capitalize' }}
              >
                <Zap size={16} style={{ color: difficulty === 'impossible' ? 'var(--accent-rose)' : 'var(--accent-amber)' }} />
                <span>{difficulty}</span>
              </button>

              {difficultyOpen && (
                <div
                  className="glass-panel"
                  style={{
                    position: 'absolute',
                    bottom: '60px',
                    right: '120px',
                    width: '160px',
                    padding: '6px',
                    zIndex: 20,
                    boxShadow: '0 10px 30px rgba(0,0,0,0.6)'
                  }}
                >
                  {['normal', 'smart', 'impossible'].map((diff) => (
                    <button
                      key={diff}
                      style={{
                        width: '100%',
                        padding: '8px 12px',
                        background: difficulty === diff ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                        border: 'none',
                        color: 'var(--text-main)',
                        borderRadius: 'var(--radius-sm)',
                        textAlign: 'left',
                        cursor: 'pointer',
                        textTransform: 'capitalize',
                        fontSize: '0.85rem',
                        fontWeight: 500
                      }}
                      onClick={() => {
                        setDifficulty(diff);
                        setDifficultyOpen(false);
                      }}
                    >
                      {diff} AI
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button
              className="btn btn-secondary"
              onClick={resetScores}
              style={{ padding: '14px' }}
              title="Reset Scores"
            >
              <RotateCcw size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Model Stats Modal */}
      <StatsModal
        isOpen={statsModalOpen}
        onClose={() => setStatsModalOpen(false)}
        modelInfo={modelInfo}
      />
    </div>
  );
}
