"use client";

import { useState, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Video, Disc, Save } from "lucide-react";

interface SequenceRecorderProps {
  currentSequence: number[][]; // From worker
}

export function SequenceRecorder({ currentSequence }: SequenceRecorderProps) {
  const [label, setLabel] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordedData, setRecordedData] = useState<any[]>([]);

  const captureFrame = useCallback(() => {
    if (isRecording && currentSequence.length === 30) {
      const sample = {
        label: label || "unlabeled",
        timestamp: Date.now(),
        landmarks: JSON.parse(JSON.stringify(currentSequence)),
      };
      setRecordedData((prev) => [...prev, sample]);
      setIsRecording(false); // Auto-stop after one sequence
    }
  }, [isRecording, currentSequence, label]);

  const downloadData = () => {
    const blob = new Blob([JSON.stringify(recordedData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `gestures_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setRecordedData([]);
  };

  return (
    <Card className="border-primary/20 bg-primary/5">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-bold flex items-center gap-2">
          <Disc className="size-4 text-red-500 fill-red-500" />
          Sequence Data Collector
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Input
            placeholder="Sign Label (e.g. hello)"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            className="h-8"
          />
          <Button
            size="sm"
            variant={isRecording ? "destructive" : "default"}
            onClick={() => setIsRecording(true)}
            disabled={currentSequence.length < 30}
          >
            {isRecording ? "Recording..." : "Capture"}
          </Button>
        </div>
        
        <div className="flex items-center justify-between">
          <p className="text-[10px] text-muted-foreground">
            Captured: <strong>{recordedData.length}</strong> samples
          </p>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-[10px]"
            onClick={downloadData}
            disabled={recordedData.length === 0}
          >
            <Save className="mr-1 size-3" />
            Export Dataset
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
