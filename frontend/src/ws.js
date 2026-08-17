let socket = null;

/**
 * Connects to the backend WebSocket using a RELATIVE URL so the dashboard
 * works from any host/port (dev server, Railway, Telegram Mini App proxy...).
 */
export const connectWebSocket = (onMessage) => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${proto}://${window.location.host}/ws`);

  socket.onopen = () => console.log("[WS] Connected to backend");

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error("WS parse error", e);
    }
  };

  socket.onclose = () => console.log("[WS] Disconnected");
  socket.onerror = (err) => console.error("[WS] Error", err);
};

export const disconnectWebSocket = () => {
  if (socket) {
    socket.close();
    socket = null;
  }
};
