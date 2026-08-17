import React from 'react';

export default function RiskDashboard({ data }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
      <div className="font-semibold mb-4">Risk Dashboard</div>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <div className="text-xs text-zinc-400">On-Chain Risk</div>
          <div className="font-mono text-xl font-bold">0.52</div>
        </div>
        <div>
          <div className="text-xs text-zinc-400">Sentiment</div>
          <div className="font-mono text-xl font-bold">0.68</div>
        </div>
        <div>
          <div className="text-xs text-zinc-400">Max Concentration</div>
          <div className="font-mono text-xl font-bold">14.2%</div>
        </div>
        <div>
          <div className="text-xs text-zinc-400">CVaR (5%)</div>
          <div className="font-mono text-xl font-bold text-amber-400">2.8%</div>
        </div>
      </div>
    </div>
  );
}