import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import json
import os

# Define BASE_DIR and dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_dataset")

# Load dataset with 80% training, 10% validation, 10% test
batch_size = 32
img_size = (128, 128)

train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Save class names using BASE_DIR
class_names_path = os.path.join(BASE_DIR, "asl_class_names.json")
with open(class_names_path, "w") as f:
    json.dump(train_ds.class_names, f)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

test_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="validation",
    seed=456,
    image_size=img_size,
    batch_size=batch_size
)

# Define the CNN model
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),  # Modified input shape
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(36, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
epochs = 30
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate model
model.evaluate(test_ds)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('WLASL Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# Save model using BASE_DIR to build a cross-platform path
model_save_path = os.path.join(BASE_DIR, "app", "models", "asl", "asl_model.keras")
model.save(model_save_path)