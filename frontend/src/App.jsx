import React, { useState, useEffect } from 'react';
import { connectWebSocket } from './ws';
import TelemetryPanel from './components/TelemetryPanel';
import PositionsTable from './components/PositionsTable';
import ActiveModels from './components/ActiveModels';
import CapitalExposure from './components/CapitalExposure';
import EquityChart from './components/EquityChart';
import RiskDashboard from './components/RiskDashboard';
import StrategyAttribution from './components/StrategyAttribution';

function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [activeModels, setActiveModels] = useState([]);
  const [exposure, setExposure] = useState(68);
  const [positions, setPositions] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    connectWebSocket((data) => {
      setTelemetry(data);
      setIsConnected(true);

      if (data.positions) setPositions(data.positions);
      if (data.active_models) setActiveModels(data.active_models);
      if (data.capital_exposure) setExposure(data.capital_exposure);
    });
  }, []);

  const handlePause = async () => {
    await fetch('/api/toggle-bot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_running: false })
    });
  };

  const handleKillSwitch = async () => {
    if (!confirm('⚠️ KILL SWITCH ? Toutes les positions seront liquidées.')) return;
    await fetch('/api/kill-switch', { method: 'POST' });
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-x-4">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
              <i className="fa-solid fa-robot text-3xl text-white"></i>
            </div>
            <div>
              <div className="font-display text-4xl font-bold tracking-tighter">Q-Bot</div>
              <div className="text-xs text-zinc-500 -mt-1">Institutional v4.2</div>
            </div>
          </div>

          <div className="flex items-center gap-x-4">
            <div className={`px-4 py-1.5 rounded-3xl text-sm font-bold flex items-center gap-x-2 
              ${isConnected ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-red-500/10 text-red-400'}`}>
              <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`}></div>
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Telemetry + Chart */}
          <div className="lg:col-span-7 space-y-6">
            <TelemetryPanel data={telemetry} />
            <EquityChart />
          </div>

          {/* Right Column */}
          <div className="lg:col-span-5 space-y-6">
            <ActiveModels models={activeModels} />
            <CapitalExposure exposure={exposure} />
            <RiskDashboard data={telemetry} />
          </div>

          {/* Positions */}
          <div className="lg:col-span-12">
            <PositionsTable positions={positions} />
          </div>

          {/* Strategy Attribution */}
          <div className="lg:col-span-6">
            <StrategyAttribution />
          </div>

          {/* Quick Actions */}
          <div className="lg:col-span-6">
            <div className="bg-zinc-900 border border-zinc-800 rounded-3xl p-6">
              <div className="font-semibold mb-4 text-lg">Quick Actions</div>
              <div className="grid grid-cols-2 gap-4">
                <button 
                  onClick={handlePause}
                  className="flex items-center justify-center gap-x-3 px-6 py-4 rounded-2xl bg-zinc-800 hover:bg-zinc-700 active:scale-[0.985] transition-all font-semibold"
                >
                  <i className="fa-solid fa-pause"></i>
                  <span>PAUSE BOT</span>
                </button>
                <button 
                  onClick={handleKillSwitch}
                  className="flex items-center justify-center gap-x-3 px-6 py-4 rounded-2xl bg-red-500/90 hover:bg-red-600 active:scale-[0.985] transition-all font-bold text-white"
                >
                  <i className="fa-solid fa-skull"></i>
                  <span>KILL SWITCH</span>
                </button>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;