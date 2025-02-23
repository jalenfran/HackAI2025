import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

# Fix BASE_DIR to point to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_dataset")

# Load dataset with grayscale images
batch_size = 32
img_size = (128, 128)

train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'  # Changed to grayscale
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'  # Changed to grayscale
)

test_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.1,
    subset="validation",
    seed=456,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'  # Changed to grayscale
)

# Define the CNN model with single channel input - removed rescaling layer
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
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

# Save model to correct path relative to project root
model_save_path = os.path.join(BASE_DIR, "app", "models", "asl", "asl_model.keras")
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")