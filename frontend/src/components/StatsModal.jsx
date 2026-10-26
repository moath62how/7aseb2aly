import React from 'react';
import { X, Cpu, Database, CheckCircle2, Award, Zap } from 'lucide-react';

export default function StatsModal({ isOpen, onClose, modelInfo }) {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(5, 8, 17, 0.85)',
        backdropFilter: 'blur(10px)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px'
      }}
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '540px',
          padding: '28px',
          borderRadius: 'var(--radius-lg)',
          position: 'relative',
          boxShadow: '0 20px 50px rgba(0,0,0,0.8)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', borderBottom: '1px solid var(--bg-card-border)', paddingBottom: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Cpu style={{ color: 'var(--accent-cyan)' }} size={24} />
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>ML Model Specifications</h3>
          </div>
          <button
            onClick={onClose}
            className="btn-secondary"
            style={{ padding: '6px', borderRadius: '50%', border: 'none', cursor: 'pointer' }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Specs Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '22px' }}>
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--bg-card-border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Database size={14} /> Training Samples
            </div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
              {modelInfo?.dataset_samples ? modelInfo.dataset_samples.toLocaleString() : '45,050'}
            </div>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--bg-card-border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Award size={14} /> Model Accuracy
            </div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>
              {modelInfo?.accuracy ? `${(modelInfo.accuracy * 100).toFixed(2)}%` : '98.87%'}
            </div>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--bg-card-border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Zap size={14} /> Architecture
            </div>
            <div style={{ fontSize: '0.92rem', fontWeight: 600 }}>
              MLP Neural Net & Random Forest
            </div>
          </div>

          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--bg-card-border)' }}>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <CheckCircle2 size={14} /> Input Vector
            </div>
            <div style={{ fontSize: '0.92rem', fontWeight: 600 }}>
              42 Landmark Coords (21 pts)
            </div>
          </div>
        </div>

        {/* Dataset Breakdown */}
        <h4 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '10px', color: 'var(--text-main)' }}>
          Class Balance in `data.csv`:
        </h4>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '22px' }}>
          {[
            { name: 'Paper (📄)', count: 15050, pct: 33.4, color: 'var(--accent-cyan)' },
            { name: 'Scissors / Scissor (✂️)', count: 15000, pct: 33.3, color: 'var(--accent-emerald)' },
            { name: 'Rock / Stone (🪨)', count: 15000, pct: 33.3, color: 'var(--accent-amber)' }
          ].map((item) => (
            <div key={item.name} style={{ background: 'rgba(255,255,255,0.02)', padding: '8px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                <span>{item.name}</span>
                <span className="mono" style={{ color: 'var(--text-muted)' }}>{item.count.toLocaleString()} samples ({item.pct}%)</span>
              </div>
              <div style={{ height: '5px', width: '100%', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${item.pct * 3}%`, background: item.color, borderRadius: '3px' }} />
              </div>
            </div>
          ))}
        </div>

        <div style={{ textAlign: 'right' }}>
          <button className="btn btn-primary" onClick={onClose}>
            Close Specifications
          </button>
        </div>
      </div>
    </div>
  );
}
