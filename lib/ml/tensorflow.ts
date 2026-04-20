import * as tf from "@tensorflow/tfjs";

let model: tf.LayersModel | null = null;

const ISL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export async function loadGestureModel(): Promise<tf.LayersModel> {
  if (model) return model;

  try {
    model = await tf.loadLayersModel("/models/isl-gesture/model.json");
    return model;
  } catch {
    console.warn(
      "ISL gesture model not found at /models/isl-gesture/model.json. " +
        "Gesture classification will use placeholder logic."
    );
    throw new Error("Model not available");
  }
}

export function getLabels(): string[] {
  return ISL_LABELS;
}

export async function predictSequence(
  loadedModel: tf.LayersModel,
  sequence: number[][] // Matrix of shape [30, 126]
): Promise<{ label: string; confidence: number } | null> {
  if (sequence.length < 30) return null;

  // Reshape to [1, 30, 126] (Batch, Time, Features)
  const tensor = tf.tensor3d([sequence]);
  
  try {
    const prediction = loadedModel.predict(tensor) as tf.Tensor;
    const probabilities = await prediction.data();

    let maxIndex = 0;
    let maxProb = 0;
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    prediction.dispose();
    return {
      label: ISL_LABELS[maxIndex] ?? "Unknown",
      confidence: maxProb,
    };
  } catch (err) {
    console.error("LSTM Inference error:", err);
    return null;
  } finally {
    tensor.dispose();
  }
}

export async function predict(
  loadedModel: tf.LayersModel,
  features: number[]
): Promise<{ label: string; confidence: number }> {
  // Legacy support for single-frame models
  const tensor = tf.tensor2d([features]);
  const prediction = loadedModel.predict(tensor) as tf.Tensor;
  const probabilities = await prediction.data();

  tensor.dispose();
  prediction.dispose();

  let maxIndex = 0;
  let maxProb = 0;
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }

  return {
    label: ISL_LABELS[maxIndex] ?? "Unknown",
    confidence: maxProb,
  };
}

export function disposeModel() {
  if (model) {
    model.dispose();
    model = null;
  }
}
