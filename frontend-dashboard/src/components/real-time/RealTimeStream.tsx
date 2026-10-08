'use client';

import React, { useState, useEffect } from 'react';

export const RealTimeStream: React.FC = () => {
  const [messages, setMessages] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
    let ws: WebSocket;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setConnected(true);
        ws.send('ping');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMessages((prev) => [data, ...prev.slice(0, 19)]);
        } catch {
          // Plain text message
          setMessages((prev) => [{ type: 'raw', text: event.data }, ...prev.slice(0, 19)]);
        }
      };

      ws.onclose = () => setConnected(false);
      ws.onerror = () => setConnected(false);
    } catch {
      setConnected(false);
    }

    return () => {
      if (ws) ws.close();
    };
  }, []);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 shadow-xl">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h2 className="text-lg font-bold">Real-time WebSocket Stream</h2>
          <p className="text-xs text-slate-400">Pembaruan langsung proses training & simulasi</p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`}
          />
          <span className="text-xs text-slate-400">{connected ? 'Terhubung' : 'Terputus'}</span>
        </div>
      </div>

      <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-300 h-48 overflow-y-auto space-y-1.5">
        {messages.length === 0 ? (
          <span className="text-slate-500">[WS] Menunggu pesan siaran dari backend...</span>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className="text-slate-300">
              <span className="text-slate-500">[{msg.type || 'INFO'}]</span>{' '}
              {msg.type === 'training_progress'
                ? `Epoch ${msg.epoch}/${msg.total_epochs} - Loss: ${msg.loss} - Acc: ${(msg.accuracy * 100).toFixed(1)}%`
                : msg.type === 'simulation_started'
                ? `Simulasi dimulai: ${msg.simulation_type} (${msg.num_agents} agen)`
                : msg.type === 'simulation_completed'
                ? `Simulasi selesai. Tingkat deteksi: ${(msg.detection_rate * 100).toFixed(1)}%`
                : JSON.stringify(msg)}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
