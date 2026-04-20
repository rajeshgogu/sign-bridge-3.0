"use client";

import { useRef, useCallback, useEffect } from "react";
import { usePhraseStore } from "@/stores/phrase-store";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import type { HandLandmark } from "@/types/ml";
import { drawHandLandmarks } from "@/lib/ml/hand-landmark-utils";

export function usePhraseRecognition(
  worker: Worker | null,
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  setCurrentSequence?: (seq: number[][]) => void
) {
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const isBusyRef = useRef<boolean>(false);
  const isMountedRef = useRef<boolean>(true); // Mounting safety

  const { isDetecting, targetPhrase, setPrediction, clearPrediction } =
    usePhraseStore();

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const drawResults = useCallback(
    (landmarks: HandLandmark[][], width: number, height: number) => {
      if (!isMountedRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      ctx.clearRect(0, 0, width, height);

      if (landmarks.length === 0) return;

      landmarks.forEach((l, idx) => {
        drawHandLandmarks(ctx, l, width, height, idx === 0 ? "#00e1ff" : "#ffdd00");
      });
    },
    [canvasRef]
  );

  useEffect(() => {
    if (!worker) return;

    const handleMessage = (e: MessageEvent) => {
      if (!isMountedRef.current) return;
      const { type, payload } = e.data;
      if (type === "RESULT") {
        const { landmarks, classification, sequenceBuffer } = payload;
        
        if (videoRef.current) {
          drawResults(landmarks || [], videoRef.current.videoWidth, videoRef.current.videoHeight);
        }

        if (sequenceBuffer && setCurrentSequence) {
          setCurrentSequence(sequenceBuffer);
        }

        if (classification && classification.confidence >= 0.85) {
          setPrediction(classification.label, classification.confidence);
        } else {
          clearPrediction();
        }
        
        isBusyRef.current = false;
      }
    };

    worker.addEventListener("message", handleMessage);
    return () => worker.removeEventListener("message", handleMessage);
  }, [worker, setPrediction, clearPrediction, drawResults, videoRef, setCurrentSequence]);

  const detect = useCallback(async () => {
    if (!worker || !videoRef.current || !isDetecting || isBusyRef.current || !isMountedRef.current) {
      if (isMountedRef.current) {
        animationRef.current = requestAnimationFrame(detect);
      }
      return;
    }

    const video = videoRef.current;
    
    // EXTREMELY STICKY READINESS CHECK
    if (
      video.readyState < 2 || 
      video.videoWidth === 0 || 
      video.videoHeight === 0 || 
      video.paused || 
      video.ended
    ) {
      animationRef.current = requestAnimationFrame(detect);
      return;
    }

    const now = performance.now();
    if (now - lastTimeRef.current < 33) {
      animationRef.current = requestAnimationFrame(detect);
      return;
    }
    lastTimeRef.current = now;

    try {
      // Capture frame as bitmap. 
      const bitmap = await createImageBitmap(video);
      
      if (isMountedRef.current && !isBusyRef.current) {
        isBusyRef.current = true;
        worker.postMessage({
          type: "DETECT",
          payload: {
            image: bitmap,
            timestamp: now,
            targetPhraseId: targetPhrase,
          }
        }, [bitmap]); 
      } else {
        bitmap.close();
      }
    } catch (err) {
      isBusyRef.current = false;
    }

    if (isMountedRef.current) {
      animationRef.current = requestAnimationFrame(detect);
    }
  }, [worker, videoRef, isDetecting, targetPhrase]);

  useEffect(() => {
    if (isDetecting && worker) {
      animationRef.current = requestAnimationFrame(detect);
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isDetecting, worker, detect]);

  return { detect, worker };
}
