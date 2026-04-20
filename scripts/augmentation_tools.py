import numpy as np
import json
import os
import glob

class ISLAugmentor:
    """
    Toolkit for augmenting 21-point hand landmark data (MediaPipe format).
    Each gesture is assumed to be an array of shape (N_frames, 21, 3).
    """

    def rotate_3d(self, landmarks, angle_x=0, angle_y=0, angle_z=0):
        """Rotates landmarks around the wrist (index 0)."""
        rad_x, rad_y, rad_z = np.radians([angle_x, angle_y, angle_z])
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(rad_x), -np.sin(rad_x)], [0, np.sin(rad_x), np.cos(rad_x)]])
        Ry = np.array([[np.cos(rad_y), 0, np.sin(rad_y)], [0, 1, 0], [-np.sin(rad_y), 0, np.cos(rad_y)]])
        Rz = np.array([[np.cos(rad_z), -np.sin(rad_z), 0], [np.sin(rad_z), np.cos(rad_z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        # Center at wrist before rotating
        wrist = landmarks[:, 0, :][:, np.newaxis, :]
        return (landmarks - wrist) @ R.T + wrist

    def add_noise(self, landmarks, sigma=0.005):
        """Adds Gaussian noise to simulate tracking jitter."""
        noise = np.random.normal(0, sigma, landmarks.shape)
        return landmarks + noise

    def scale(self, landmarks, factor=1.0):
        """Scales landmarks relative to the wrist."""
        wrist = landmarks[:, 0, :][:, np.newaxis, :]
        return (landmarks - wrist) * factor + wrist

    def extract_keyframes(self, landmarks, threshold=0.02):
        """
        Extracts frames where significant motion occurs (velocity-based).
        Helps in identifying the 'peak' of a gesture.
        """
        velocities = np.linalg.norm(np.diff(landmarks, axis=0), axis=-1).mean(axis=-1)
        # Identify peaks in velocity or frames exceeding threshold
        key_frames = np.where(velocities > threshold)[0]
        return landmarks[key_frames]

def process_dataset(input_dir, output_dir):
    augmentor = ISLAugmentor()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for file_path in glob.glob(os.path.join(input_dir, "*.json")):
        with open(file_path, 'r') as f:
            data = json.load(f)
            landmarks = np.array(data['landmarks']) # Assuming format { "landmarks": [...] }
            
            # Generate variations
            variations = [
                ("orig", landmarks),
                ("rot_z_10", augmentor.rotate_3d(landmarks, angle_z=10)),
                ("rot_z_neg10", augmentor.rotate_3d(landmarks, angle_z=-10)),
                ("scale_0.9", augmentor.scale(landmarks, 0.9)),
                ("noisy", augmentor.add_noise(landmarks)),
            ]
            
            base_name = os.path.basename(file_path).replace(".json", "")
            for suffix, aug_data in variations:
                out_path = os.path.join(output_dir, f"{base_name}_{suffix}.json")
                with open(out_path, 'w') as out_f:
                    json.dump({"landmarks": aug_data.tolist()}, out_f)

if __name__ == "__main__":
    # Example usage:
    # process_dataset("data/raw", "data/augmented")
    print("ISL Augmentation Tools Ready.")
