import { useEffect, useRef } from "react";
import { useRiskStore } from "../store/useRiskStore";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);

  const {
    connectionStatus,
    isStreaming,
    setConnectionStatus,
    setIsStreaming,
    updateFromWebSocket,
    setStressResult,
  } = useRiskStore();

  const connect = (
    currency: string,
    modelName: string,
    parameters: Record<string, number>,
    interval: number = 1.0
  ) => {
    // Disconnect existing if any
    disconnect();

    setConnectionStatus("connecting");

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/risk";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus("connected");
      // Send subscription request
      const subPayload = {
        action: "subscribe",
        currency,
        model_name: modelName,
        parameters,
        interval,
      };
      ws.send(JSON.stringify(subPayload));
    };

    ws.onmessage = async (event) => {
      try {
        let rawText = "";
        if (event.data instanceof Blob) {
          rawText = await event.data.text();
        } else if (event.data instanceof ArrayBuffer) {
          rawText = new TextDecoder("utf-8").decode(event.data);
        } else {
          rawText = event.data;
        }

        const data = JSON.parse(rawText);

        if (data.type === "update") {
          updateFromWebSocket(data);
        } else if (data.type === "stress_result") {
          setStressResult(data.greeks, data.spot);
        } else if (data.type === "subscribed") {
          setIsStreaming(true);
        } else if (data.type === "unsubscribed") {
          setIsStreaming(false);
        } else if (data.type === "error") {
          console.error("WebSocket server error:", data.message);
        }
      } catch (err) {
        console.error("Failed to parse WebSocket message:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error occurred:", err);
      ws.close();
    };

    ws.onclose = () => {
      setConnectionStatus("disconnected");
      setIsStreaming(false);
      wsRef.current = null;
    };
  };

  const disconnect = () => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        // Send unsubscribe message first
        wsRef.current.send(JSON.stringify({ action: "unsubscribe" }));
        wsRef.current.close();
      } else {
        wsRef.current.close();
      }
      wsRef.current = null;
    }
  };

  const runStressTest = (
    modelName: string,
    parameters: Record<string, number>,
    spot: number,
    r: number = 0.05,
    q: number = 0.0
  ) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const stressPayload = {
        action: "stress",
        model_name: modelName,
        parameters,
        S: spot,
        r,
        q,
      };
      wsRef.current.send(JSON.stringify(stressPayload));
    } else {
      console.warn("WebSocket not connected. Cannot run stress test.");
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  return {
    connect,
    disconnect,
    runStressTest,
    connectionStatus,
    isStreaming,
  };
}
