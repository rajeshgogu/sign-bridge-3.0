"use client";

import { useState, useRef, useEffect } from "react";
import { useCameraStore } from "@/stores/camera-store";
import { usePhraseStore } from "@/stores/phrase-store";
import { usePhraseRecognition } from "@/hooks/use-phrase-recognition";
import { Loader2, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { SequenceRecorder } from "@/components/practice/sequence-recorder";

export function PhraseDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);

  const [currentSequence, setCurrentSequence] = useState<number[][]>([]);

  const { stream, isActive, startCamera, stopCamera, toggleFacingMode } =
    useCameraStore();

  const {
    isDetecting,
    currentPrediction,
    confidence,
    startDetection,
    stopDetection,
  } = usePhraseStore();

  // ✅ Create worker
  useEffect(() => {
    workerRef.current = new Worker(
      new URL("@/lib/ml/detection.worker.ts", import.meta.url)
    );

    return () => {
      workerRef.current?.terminate();
    };
  }, []);

  // ✅ Use hook correctly (NO destructuring)
  usePhraseRecognition(
    workerRef.current,
    videoRef,
    canvasRef,
    setCurrentSequence
  );

  // Attach stream to video
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  // Start detection when camera active
  useEffect(() => {
    if (isActive) {
      startDetection();
    }
    return () => stopDetection();
  }, [isActive, startDetection, stopDetection]);

  // Handle lifecycle
  useEffect(() => {
    if (!isActive) startCamera();
    return () => {
      stopCamera();
      stopDetection();
    };
  }, []);

  return (
    <div className="space-y-4">
      <div
        className="relative overflow-hidden rounded-lg bg-black shadow-lg ring-1 ring-white/10"
        style={{ height: "480px" }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover scale-x-[-1]"
        />

        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full scale-x-[-1]"
        />

        {/* Sequence Recorder */}
        <div className="absolute left-4 top-4 z-20 w-72">
          <SequenceRecorder currentSequence={currentSequence} />
        </div>

        {/* Prediction UI */}
        {currentPrediction && (
          <div className="absolute bottom-6 left-6 right-6 rounded-2xl bg-background/80 px-6 py-4 backdrop-blur-xl border border-white/20 shadow-2xl">
            <div className="flex items-center justify-between mb-2">
              <div className="text-2xl font-black capitalize text-primary">
                {currentPrediction.replace("_", " ")}
              </div>
              <div className="text-xs font-black bg-primary/20 text-primary px-3 py-1 rounded-full">
                {Math.round(confidence * 100)}% Accuracy
              </div>
            </div>

            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full transition-all duration-500",
                  confidence > 0.9 ? "bg-green-500" : "bg-primary"
                )}
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="absolute right-4 top-4 flex flex-col gap-2">
          <Button
            variant="secondary"
            size="icon"
            className="rounded-full bg-black/40 text-white"
            onClick={toggleFacingMode}
          >
            <RotateCcw className="size-4" />
          </Button>
        </div>

        {/* Loader */}
        {!isActive && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <Loader2 className="size-10 animate-spin text-primary" />
          </div>
        )}
      </div>

      {/* Status */}
      <div className="flex justify-between text-[11px] font-bold uppercase text-muted-foreground px-2">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "size-2 rounded-full",
              isDetecting ? "bg-green-500" : "bg-red-500"
            )}
          />
          {isDetecting ? "System Online" : "System Offline"}
        </div>

        <div className="flex gap-4">
          <span>60 FPS Tracking</span>
          <span>Dual-Hand Active</span>
        </div>
      </div>
    </div>
  );
}