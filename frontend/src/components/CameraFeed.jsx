import React, { useEffect, useRef, useState } from 'react';
import { RefreshCw, AlertCircle } from 'lucide-react';

export default function CameraFeed({ onLandmarksDetected, livePrediction, countdown, isCountingDown }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const cameraObjRef = useRef(null);

  useEffect(() => {
    let hands;
    let camera;
    let isCancelled = false;

    async function initCamera() {
      try {
        // Wait for MediaPipe scripts loaded in window if necessary
        let attempts = 0;
        while (!window.Hands && attempts < 30) {
          await new Promise((r) => setTimeout(r, 100));
          attempts++;
        }

        if (!window.Hands || !window.Camera) {
          throw new Error('MediaPipe Hands or Camera script failed to load.');
        }

        const HandsClass = window.Hands;
        const CameraClass = window.Camera;
        const HandConnections = window.HAND_CONNECTIONS || [
          [0, 1], [1, 2], [2, 3], [3, 4],
          [0, 5], [5, 6], [6, 7], [7, 8],
          [5, 9], [9, 10], [10, 11], [11, 12],
          [9, 13], [13, 14], [14, 15], [15, 16],
          [13, 17], [17, 18], [18, 19], [19, 20],
          [0, 17]
        ];

        hands = new HandsClass({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6
        });

        hands.onResults((results) => {
          if (isCancelled) return;
          const canvas = canvasRef.current;
          const video = videoRef.current;
          if (!canvas || !video) return;

          const ctx = canvas.getContext('2d');
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;

          ctx.save();
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // Mirror canvas to match video stream
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);

          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            const flatLandmarks = [];
            landmarks.forEach((lm) => {
              flatLandmarks.push(lm.x, lm.y);
            });

            onLandmarksDetected(flatLandmarks);
            drawHandSkeleton(ctx, landmarks, HandConnections, canvas.width, canvas.height);
          } else {
            onLandmarksDetected(null);
          }

          ctx.restore();
        });

        if (videoRef.current) {
          camera = new CameraClass(videoRef.current, {
            onFrame: async () => {
              if (videoRef.current && hands) {
                await hands.send({ image: videoRef.current });
              }
            },
            width: 640,
            height: 480
          });
          await camera.start();
          cameraObjRef.current = camera;
          if (!isCancelled) setCameraActive(true);
        }
      } catch (err) {
        console.error('Camera initialization error:', err);
        if (!isCancelled) setErrorMsg('Camera access error. Please grant permission.');
      }
    }

    initCamera();

    return () => {
      isCancelled = true;
      if (cameraObjRef.current) cameraObjRef.current.stop();
      if (hands) hands.close();
    };
  }, []);

  function drawHandSkeleton(ctx, landmarks, connections, width, height) {
    // Draw bones
    connections.forEach(([start, end]) => {
      const p1 = landmarks[start];
      const p2 = landmarks[end];
      if (p1 && p2) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.85)';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    });

    // Draw joint nodes
    landmarks.forEach((p, idx) => {
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, idx === 0 ? 7 : 4, 0, 2 * Math.PI);
      ctx.fillStyle = idx === 0 ? '#10b981' : '#06b6d4';
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 10;
      ctx.fill();
    });
  }

  const getEmoji = (move) => {
    if (move === 'Rock') return '🪨';
    if (move === 'Paper') return '📄';
    if (move === 'Scissors') return '✂️';
    return '✋';
  };

  return (
    <div className="glass-panel" style={{ position: 'relative', overflow: 'hidden', minHeight: '380px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ position: 'relative', width: '100%', flex: 1, backgroundColor: '#050811', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <video
          ref={videoRef}
          playsInline
          muted
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scaleX(-1)',
            display: cameraActive ? 'block' : 'none'
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none'
          }}
        />

        {!cameraActive && !errorMsg && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px' }}>
            <RefreshCw className="pulse" size={36} style={{ color: 'var(--accent-cyan)', marginBottom: '12px' }} />
            <p style={{ fontWeight: 500 }}>Initializing Camera & MediaPipe...</p>
          </div>
        )}

        {errorMsg && (
          <div style={{ textAlign: 'center', color: 'var(--accent-rose)', padding: '20px' }}>
            <AlertCircle size={36} style={{ marginBottom: '8px' }} />
            <p style={{ fontWeight: 500 }}>{errorMsg}</p>
          </div>
        )}

        {cameraActive && livePrediction && !isCountingDown && (
          <div
            style={{
              position: 'absolute',
              top: '16px',
              left: '16px',
              background: 'rgba(9, 13, 22, 0.85)',
              backdropFilter: 'blur(8px)',
              border: '1px solid rgba(6, 182, 212, 0.4)',
              borderRadius: 'var(--radius-full)',
              padding: '6px 14px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '0.88rem',
              fontWeight: 600,
              color: '#fff',
              boxShadow: '0 4px 12px rgba(0,0,0,0.4)'
            }}
          >
            <span style={{ fontSize: '1.2rem' }}>{getEmoji(livePrediction.prediction)}</span>
            <span>{livePrediction.prediction}</span>
            <span style={{ color: 'var(--accent-cyan)', fontSize: '0.8rem', opacity: 0.9 }}>
              ({(livePrediction.confidence * 100).toFixed(1)}%)
            </span>
          </div>
        )}

        {isCountingDown && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(9, 13, 22, 0.75)',
              backdropFilter: 'blur(6px)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 10
            }}
          >
            <div
              className="animate-pop"
              key={countdown}
              style={{
                fontSize: countdown === 'SHOOT!' ? '3.5rem' : '6rem',
                fontWeight: 800,
                color: countdown === 'SHOOT!' ? 'var(--accent-emerald)' : 'var(--accent-cyan)',
                textShadow: '0 0 30px currentColor'
              }}
            >
              {countdown}
            </div>
            <p style={{ marginTop: '12px', color: 'var(--text-muted)', fontSize: '1rem' }}>
              Hold your hand up to the camera!
            </p>
          </div>
        )}
      </div>

      <div style={{ padding: '12px 18px', borderTop: '1px solid var(--bg-card-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: cameraActive ? 'var(--accent-emerald)' : 'var(--accent-rose)' }} />
          <span>{cameraActive ? 'MediaPipe Hands Active' : 'Camera Ready'}</span>
        </div>
        <div>21 Hand Landmarks Tracked</div>
      </div>
    </div>
  );
}
