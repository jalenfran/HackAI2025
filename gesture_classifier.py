import tensorflow as tf
import os

# Constants
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 10
DATA_DIR = os.path.join(os.getcwd(), 'assets/leapGestRecog')
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

# List PNG image files recursively: subjects/*/gesture_folder/*.png
file_pattern = os.path.join(DATA_DIR, '*', '*', '*.png')
list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)

def process_path(file_path):
    # Read and decode image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    # Extract gesture folder from path (subject/gesture_folder/image.png)
    parts = tf.strings.split(file_path, os.sep)
    gesture_folder = parts[-2]
    # Label: first two characters converted to integer minus 1 (e.g. "01_palm" -> 0)
    label = tf.strings.to_number(tf.strings.substr(gesture_folder, 0, 2), tf.int32) - 1
    return img, label

labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

# Split into training and validation datasets (80%-20%)
dataset_size = sum(1 for _ in list_ds)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_ds = labeled_ds.take(train_size)
val_ds = labeled_ds.skip(train_size)

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build a simple CNN classification model
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMG_SIZE + (3,)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train model
EPOCHS = 2
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Evaluate model on validation data
eval_results = model.evaluate(val_ds)
print("Evaluation results:", eval_results)

# Save the trained model
model.save(os.path.join(os.getcwd(), 'gesture_model.keras'))
print("Model saved as gesture_model.keras")
