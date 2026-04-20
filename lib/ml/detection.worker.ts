import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { classifyISLPhrase } from "./isl-phrase-classifier";
import { getUnifiedHandVector } from "./hand-landmark-utils";

let landmarker: HandLandmarker | null = null;

// Sequence State
const SEQUENCE_SIZE = 30; // ~1 second @ 30fps
let sequenceBuffer: number[][] = [];
let lastPredictionLabel = "";
let lastPredictionTime = 0;
const DEBOUNCE_MS = 2000;

async function init() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
  );

  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  
  self.postMessage({ type: "INIT_DONE" });
}

self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  if (type === "INIT") {
    await init();
    return;
  }

  if (type === "DETECT" && landmarker) {
    const { image, timestamp, targetPhraseId } = payload;
    
    const results = landmarker.detectForVideo(image, timestamp);
    
    let classification = null;
    
    // 1. Maintain sequence buffer for LSTM
    const unifiedVector = getUnifiedHandVector(results.landmarks || []);
    sequenceBuffer.push(unifiedVector);
    if (sequenceBuffer.length > SEQUENCE_SIZE) sequenceBuffer.shift();

    // 2. Perform classification
    // For now, we still call the heuristic classifier, but we pass the sequence if needed
    if (results.landmarks && results.landmarks.length > 0) {
      classification = classifyISLPhrase(results.landmarks, targetPhraseId);
      
      // Implement basic debounce logic
      const now = Date.now();
      if (classification && classification.label === lastPredictionLabel && now - lastPredictionTime < DEBOUNCE_MS) {
        classification = null; // Suppress repeat within debounce window
      } else if (classification) {
        lastPredictionLabel = classification.label;
        lastPredictionTime = now;
      }
    } else {
      // Clear buffer if hands lost for too long (e.g. 5 frames)
      // This helps with segmentation
      if (sequenceBuffer.length > 5) {
        const lastFew = sequenceBuffer.slice(-5);
        const isIdle = lastFew.every(v => v.every(val => val === 0));
        if (isIdle) sequenceBuffer = [];
      }
    }

    self.postMessage({
      type: "RESULT",
      payload: {
        landmarks: results.landmarks,
        classification,
        timestamp,
        sequenceBuffer: sequenceBuffer.length === SEQUENCE_SIZE ? sequenceBuffer : null,
        sequenceReady: sequenceBuffer.length === SEQUENCE_SIZE
      },
    });
    
    image.close();
  }
};
