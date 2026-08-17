import React, { useEffect, useState } from 'react';

export default function StrategyAttribution() {
  const [attribution, setAttribution] = useState({});

  useEffect(() => {
    fetch('/api/attribution')
      .then(res => res.json())
      .then(setAttribution)
      .catch(() => {});
  }, []);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
      <div className="font-semibold mb-4">Strategy Attribution (30 trades)</div>
      
      <div className="space-y-3 text-sm">
        {Object.keys(attribution).length > 0 ? (
          Object.entries(attribution).map(([name, stats]) => (
            <div key={name} className="flex justify-between items-center">
              <div>
                <div className="font-medium">{name}</div>
                <div className="text-xs text-zinc-400">{stats.trades} trades</div>
              </div>
              <div className="text-right">
                <div className="font-mono text-emerald-400">{stats.avg_score?.toFixed(3)}</div>
                <div className="text-xs text-zinc-400">{(stats.weight * 100).toFixed(0)}% weight</div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-zinc-400 text-sm">Loading attribution...</div>
        )}
      </div>
    </div>
  );
}