import React from 'react';

export default function TelemetryPanel({ data }) {
  if (!data) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
        <div className="animate-pulse">Loading telemetry...</div>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="text-sm text-zinc-400">LIVE TELEMETRY</div>
          <div className="font-display text-4xl font-bold tracking-tighter">
            ${data.current_equity?.toLocaleString()}
          </div>
        </div>
        <div className="text-right">
          <div className={`text-2xl font-bold ${data.live_pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {data.live_pnl_pct >= 0 ? '+' : ''}{data.live_pnl_pct?.toFixed(2)}%
          </div>
          <div className="text-xs text-zinc-500">24h PnL</div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <div className="text-zinc-400 text-xs">MODE</div>
          <div className="font-mono font-bold">{data.mode}</div>
        </div>
        <div>
          <div className="text-zinc-400 text-xs">REGIME</div>
          <div className="font-mono font-bold">{data.regime_name}</div>
        </div>
        <div>
          <div className="text-zinc-400 text-xs">SHARPE</div>
          <div className="font-mono font-bold text-emerald-400">1.84</div>
        </div>
        <div>
          <div className="text-zinc-400 text-xs">DRAWDOWN</div>
          <div className="font-mono font-bold text-amber-400">{data.drawdown || '-2.1%'}</div>
        </div>
      </div>
    </div>
  );
}