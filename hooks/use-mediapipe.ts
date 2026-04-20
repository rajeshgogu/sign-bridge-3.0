"use client";

import { useState, useCallback } from "react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";

export function useMediapipe() {
  const [landmarker, setLandmarker] = useState<HandLandmarker | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const initialize = useCallback(async () => {
    if (landmarker || loading) return;
    setLoading(true);
    setError(null);

    try {
      const worker = new Worker(
        new URL("../lib/ml/detection.worker.ts", import.meta.url)
      );
      
      worker.onmessage = (e) => {
        if (e.data.type === "INIT_DONE") {
          setLoading(false);
        }
        if (e.data.type === "ERROR") {
          setError(e.data.payload);
          setLoading(false);
        }
      };

      worker.postMessage({ type: "INIT" });
      setLandmarker(worker as any); // Cast worker as the "instance"
    } catch (err) {
      setError("Failed to initialize detection worker");
      console.error(err);
      setLoading(false);
    }
  }, [landmarker, loading]);

  return { landmarker, loading, error, initialize };
}
