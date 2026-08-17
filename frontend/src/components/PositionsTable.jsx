import React from 'react';

export default function PositionsTable({ positions = [] }) {
  if (positions.length === 0) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6 text-center text-zinc-400">
        No open positions
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
      <div className="font-semibold mb-4 flex items-center justify-between">
        <span>Open Positions</span>
        <span className="text-xs text-zinc-400">{positions.length} assets</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-zinc-400 border-b border-zinc-700">
              <th className="text-left py-3 font-normal">Symbol</th>
              <th className="text-right py-3 font-normal">Qty</th>
              <th className="text-right py-3 font-normal">Avg Price</th>
              <th className="text-right py-3 font-normal">Current</th>
              <th className="text-right py-3 font-normal">Unrealized PnL</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos, index) => {
              const unrealized = ((pos.price - pos.avg) / pos.avg * 100);
              const pnlColor = pos.pnl >= 0 ? 'text-emerald-400' : 'text-red-400';
              
              return (
                <tr key={index} className="border-b border-zinc-800 last:border-0">
                  <td className="py-3 font-bold">{pos.symbol}</td>
                  <td className="py-3 text-right font-mono">{pos.qty}</td>
                  <td className="py-3 text-right font-mono text-zinc-400">${pos.avg}</td>
                  <td className="py-3 text-right font-mono">${pos.price}</td>
                  <td className={`py-3 text-right font-mono font-bold ${pnlColor}`}>
                    {pos.pnl >= 0 ? '+' : ''}${pos.pnl} <span className="text-xs">({unrealized.toFixed(1)}%)</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}