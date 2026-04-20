import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os

# --- Configuration ---
SEQUENCE_LENGTH = 30  # Number of frames per gesture
FEATURES_PER_HAND = 21 * 3
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # Both hands (126)
LABELS = ["hello", "thank_you", "nice_to_meet_you", "how_are_you", "emergency"]

def create_lstm_model(num_classes):
    """
    Creates a Spatio-Temporal LSTM model for gesture recognition.
    Input Shape: (Batch, Sequence, Features) -> (None, 30, 126)
    """
    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, TOTAL_FEATURES)),
        
        # Spatial-Temporal Feature Extraction
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        
        # Classification Head
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(data_dir):
    """
    Loads JSON sequence files and prepares them for training.
    Expected JSON format: { "label": "hello", "landmarks": [ [126], [126], ... (30 times) ] }
    """
    X, y = [], []
    label_map = {label: i for i, label in enumerate(LABELS)}
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                item = json.load(f)
                if len(item['landmarks']) == SEQUENCE_LENGTH:
                    X.append(item['landmarks'])
                    # One-hot encode label
                    one_hot = np.zeros(len(LABELS))
                    one_hot[label_map[item['label']]] = 1
                    y.append(one_hot)
                    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Initializing LSTM Gesture Recognition Model...")
    model = create_lstm_model(len(LABELS))
    model.summary()
    
    # Example Export to TFJS
    # import tensorflowjs as tfjs
    # tfjs.converters.save_keras_model(model, "public/models/isl-gesture")
    
    print("\nTraining Logic Ready.")
    print(f"Input Shape: (None, {SEQUENCE_LENGTH}, {TOTAL_FEATURES})")
    print(f"Output Classes: {LABELS}")
