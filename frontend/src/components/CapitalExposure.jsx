import React from 'react';

export default function CapitalExposure({ exposure = 68 }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
      <div className="font-semibold mb-4">Capital Exposure (LOT 50)</div>
      
      <div className="flex items-end justify-between mb-2">
        <div>
          <div className="font-display text-5xl font-bold tracking-tighter">{exposure}</div>
          <div className="text-xs text-zinc-400 -mt-1">PERCENT</div>
        </div>
        <div className="text-emerald-400 text-sm font-bold">OPTIMAL</div>
      </div>

      <div className="h-3 bg-zinc-800 rounded-2xl overflow-hidden">
        <div 
          className="h-3 bg-gradient-to-r from-emerald-400 to-cyan-400 transition-all duration-500"
          style={{ width: `${exposure}%` }}
        />
      </div>

      <div className="flex justify-between text-xs mt-1.5 text-zinc-400">
        <div>Min 28%</div>
        <div>Max 92%</div>
      </div>
    </div>
  );
}