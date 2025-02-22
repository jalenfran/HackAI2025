import tensorflow as tf
import numpy as np
import json
import os
import cv2
from tensorflow.keras import layers, models

# Load WLASL JSON
def load_wlasl_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Custom dataset class for WLASL
class WLASLDataset(tf.keras.utils.Sequence):
    def __init__(self, json_path, video_folder, batch_size=8, num_frames=16):
        self.data = load_wlasl_json(json_path)
        self.video_folder = video_folder
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.instances = []
        # Determine if the JSON is a dict with key "root" or a list
        root = self.data["root"] if isinstance(self.data, dict) and "root" in self.data else self.data
        for item in root:
            gloss = item["gloss"]
            for instance in item["instances"]:
                video_id = instance["video_id"]
                self.instances.append((video_id, gloss))
    
    def __len__(self):
        return len(self.instances) // self.batch_size
    
    def __getitem__(self, idx):
        batch_instances = self.instances[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_frames = []
        batch_labels = []
        for video_id, gloss in batch_instances:
            video_path = os.path.join(self.video_folder, f"{video_id}.mp4")
            frames = self.load_video_frames(video_path)
            label = self.gloss_to_label(gloss)
            batch_frames.append(frames)
            batch_labels.append(label)
        return np.array(batch_frames), np.array(batch_labels)
    
    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        # Pad with zeros if not enough frames
        if len(frames) < self.num_frames:
            missing = self.num_frames - len(frames)
            pad_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * missing
            frames.extend(pad_frames)
        return np.array(frames) / 255.0
    
    def gloss_to_label(self, gloss):
        return hash(gloss) % 2000

# Define simple CNN-LSTM model
def build_model(num_classes=2000):
    inputs = layers.Input(shape=(16, 224, 224, 3))
    # Downsample spatial dimensions early with strides and fewer filters
    x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same'))(inputs)  # output: (16,112,112,32)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)  # output: (16,56,56,32)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)  # output: (16,32)
    x = layers.LSTM(256, return_sequences=False)(x)  # reduced LSTM units
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training setup
def train_wlasl(json_path, video_folder, epochs=10, batch_size=8):
    dataset = WLASLDataset(json_path, video_folder, batch_size)
    model = build_model()
    model.fit(dataset, epochs=epochs)
    model.save("wlasl_model.keras")

# Example usage with absolute paths
if __name__ == '__main__':
    train_wlasl("assets/WLASL_v0.3.json", "assets/videos", epochs=5)
