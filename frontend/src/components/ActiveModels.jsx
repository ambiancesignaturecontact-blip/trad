import React from 'react';

export default function ActiveModels({ models = [] }) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
      <div className="font-semibold mb-4 flex items-center justify-between">
        <span>Active Models (LOT 46)</span>
        <span className="text-xs px-3 py-1 bg-cyan-500/10 text-cyan-400 rounded-2xl">
          {models.length} active
        </span>
      </div>

      {models.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {models.map((model, index) => (
            <div 
              key={index} 
              className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-2xl text-sm flex items-center gap-x-2 transition-colors"
            >
              <div className="w-2 h-2 rounded-full bg-emerald-400"></div>
              <span>{model}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-zinc-400 text-sm">No active models</div>
      )}
    </div>
  );
}