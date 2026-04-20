"use client";

import { useRef, useEffect } from "react";
import { useCameraStore } from "@/stores/camera-store";
import { usePhraseStore } from "@/stores/phrase-store";
import { useMediapipe } from "@/hooks/use-mediapipe";
import { usePhraseRecognition } from "@/hooks/use-phrase-recognition";
import { Loader2, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function PhraseDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
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
  const { landmarker, loading: mpLoading, initialize: initMP } = useMediapipe();

  // Integrated recognition hook with sequence support
  const { worker } = usePhraseRecognition(landmarker, videoRef, canvasRef, setCurrentSequence);

  // Attach stream to video element
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  // Initialize MediaPipe when camera is active
  useEffect(() => {
    if (isActive && !landmarker && !mpLoading) {
      initMP();
    }
  }, [isActive, landmarker, mpLoading, initMP]);

  // Start detection when ready
  useEffect(() => {
    if (isActive && landmarker) {
      startDetection();
    }
    return () => stopDetection();
  }, [isActive, landmarker, startDetection, stopDetection]);

  // Handle lifetime
  useEffect(() => {
    if (!isActive) startCamera();
    return () => {
      stopCamera();
      stopDetection();
    };
  }, []);

  return (
    <div className="space-y-4">
      <div className="relative overflow-hidden rounded-lg bg-black shadow-lg ring-1 ring-white/10" style={{ height: "480px" }}>
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

        {/* Sequence Recorder UI */}
        <div className="absolute left-4 top-4 z-20 w-72">
          <SequenceRecorder currentSequence={currentSequence} />
        </div>

        {/* Prediction Display Overlay */}
        {currentPrediction && (
          <div className="absolute bottom-6 left-6 right-6 rounded-2xl bg-background/80 px-6 py-4 backdrop-blur-xl border border-white/20 shadow-2xl">
            <div className="flex items-center justify-between mb-2">
              <div className="text-2xl font-black capitalize text-primary tracking-tight">
                {currentPrediction.replace('_', ' ')}
              </div>
              <div className="text-xs font-black bg-primary/20 text-primary px-3 py-1 rounded-full uppercase tracking-tighter">
                {Math.round(confidence * 100)}% Accuracy
              </div>
            </div>
            <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
              <div 
                className={cn(
                  "h-full transition-all duration-500 ease-out",
                  confidence > 0.9 ? "bg-green-500 shadow-[0_0_10px_#22c55e]" : "bg-primary"
                )}
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Controls Overlay */}
        <div className="absolute right-4 top-4 flex flex-col gap-2">
          <Button
            variant="secondary"
            size="icon"
            className="rounded-full bg-black/40 hover:bg-black/60 text-white border-0 backdrop-blur-md"
            onClick={toggleFacingMode}
          >
            <RotateCcw className="size-4" />
          </Button>
        </div>

        {/* Status Indicators */}
        {mpLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm">
            <Loader2 className="size-10 animate-spin text-primary" />
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-widest text-muted-foreground px-2">
        <div className="flex items-center gap-2">
          <div className={cn("size-2 rounded-full", isDetecting ? "bg-green-500 animate-pulse" : "bg-red-500")} />
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
