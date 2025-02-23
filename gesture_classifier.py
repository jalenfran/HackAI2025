import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Constants
BATCH_SIZE = 32
IMG_SIZE = (64, 64)
NUM_CLASSES = 10

# Define BASE_DIR and DATA_DIR for your gestures dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "leapGestRecog")

GESTURE_MAP = {
    0: 'palm',        # 01_palm
    1: 'l',           # 02_l
    2: 'fist',        # 03_fist
    3: 'fist_moved',  # 04_fist_moved
    4: 'thumb',       # 05_thumb
    5: 'index',       # 06_index
    6: 'ok',          # 07_ok
    7: 'palm_moved',  # 08_palm_moved
    8: 'c',           # 09_c
    9: 'down'         # 10_down
}

# Load dataset (placeholder)
train_ds = keras.preprocessing.image_dataset_from_directory (
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Define the gesture model without using RandomRotation
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (adjust epochs if needed)
EPOCHS = 10
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Save the model to be loaded in gesture.py
model_save_path = os.path.join(BASE_DIR, "app", "models", "gestures", "gesture_model.keras")
model.save(model_save_path)
print("Model saved as gesture_model.keras")
