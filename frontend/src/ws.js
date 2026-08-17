let socket = null;

export const connectWebSocket = (onMessage) => {
  socket = new WebSocket("ws://localhost:8000/ws");

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