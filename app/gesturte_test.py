import os
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
import concurrent.futures
from PIL import Image
import tensorflow as tf

# -------------------------
# Gesture classes.
# -------------------------
classes = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c']

# -------------------------
# Load the Keras model.
# -------------------------
model = tf.keras.models.load_model("models/gestures/gesture_model.keras")
model.summary()  # Optional: print model architecture.

# Directories for saving images (if needed)
os.makedirs("saved_images", exist_ok=True)
os.makedirs("model_inputs", exist_ok=True)

# -------------------------
# Initialize MediaPipe Hands.
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------
# Function to crop hand from frame.
# -------------------------
def crop_to_hand(frame, results):
    h, w, _ = frame.shape
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        x_min = int(min(x_coords) * w)
        x_max = int(max(x_coords) * w)
        y_min = int(min(y_coords) * h)
        y_max = int(max(y_coords) * h)
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        return frame[y_min:y_max, x_min:x_max]
    return None

# -------------------------
# Function for background removal.
# -------------------------
def process_hand_region_rembg(cropped_region):
    # Scale down for speed.
    small = cv2.resize(cropped_region, (256, 256))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_small = Image.fromarray(rgb_small)
    result_pil = remove(pil_small)
    result_np = np.array(result_pil)
    if result_np.shape[2] == 4:
        alpha = result_np[:, :, 3] / 255.0
        composite = np.empty_like(result_np[:, :, :3], dtype=np.uint8)
        for c in range(3):
            composite[:, :, c] = (result_np[:, :, c] * alpha).astype(np.uint8)
    else:
        composite = result_np
    composite_resized = cv2.resize(composite, (cropped_region.shape[1], cropped_region.shape[0]))
    return composite_resized

# -------------------------
# Worker function: processes one frame.
# -------------------------
def process_frame(frame, frame_count):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)
    cropped_frame = crop_to_hand(frame, results)
    if cropped_frame is not None:
        processed_cropped = process_hand_region_rembg(cropped_frame)
        gray_processed = cv2.cvtColor(processed_cropped, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(gray_processed)
        # Resize to 64x64 for the Keras model.
        pil_img_resized = pil_img.resize((64, 64))
        
        # Save processed images for inspection.
        inspection_path = os.path.join("saved_images", f"frame_{frame_count}_processed.png")
        pil_img_resized.save(inspection_path)
        model_input_path = os.path.join("model_inputs", f"frame_{frame_count}.png")
        pil_img_resized.save(model_input_path)
        
        # Prepare input for the Keras model.
        input_img = np.array(pil_img_resized)
        # Ensure image has shape (64, 64, 1) for grayscale.
        if input_img.ndim == 2:
            input_img = input_img[..., np.newaxis]
        input_img = input_img.astype("float32") / 255.0  # Normalize pixel values.
        input_img = np.expand_dims(input_img, axis=0)     # Add batch dimension.
        
        # Debug: print input shape.
        print(f"[Frame {frame_count}] Input shape: {input_img.shape}")
        
        # Get predictions.
        preds = model.predict(input_img)
        print(f"[Frame {frame_count}] Raw predictions: {preds}")
        pred = np.argmax(preds, axis=1)[0]
        print(f"[Frame {frame_count}] Prediction index: {pred} ({classes[pred]})")
        return pred, cropped_frame, results
    else:
        print(f"[Frame {frame_count}] No hand detected.")
    return None, None, None

# -------------------------
# Main capture loop with thread pool.
# -------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    frame_count = 0
    predicted_label = "No hand detected"
    results = None  # Initialize results.
    future = None

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        
        # Submit a new frame for processing if none is pending.
        if future is None or future.done():
            future = executor.submit(process_frame, frame.copy(), frame_count)
            frame_count += 1

        # Update prediction if processing is complete.
        if future is not None and future.done():
            pred, cropped_frame, results = future.result()
            future = None
            if pred is not None:
                predicted_label = classes[pred]
            else:
                predicted_label = "No hand detected"

        cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if results is not None and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown()

if __name__ == "__main__":
    main()
