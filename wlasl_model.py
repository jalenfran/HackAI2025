import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import json

# Define dataset path
data_dir = "asl_dataset"

# Load dataset with 80% training, 10% validation, 10% test
batch_size = 32
img_size = (64, 64)

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Save class names for later use in test.py
with open("asl_class_names.json", "w") as f:
    json.dump(train_ds.class_names, f)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

test_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="validation",
    seed=456,
    image_size=img_size,
    batch_size=batch_size
)

# Define the CNN model
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 3)),
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
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate model
model.evaluate(test_ds)

# Save model
model.save("asl_model.keras")
