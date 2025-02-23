# Multi-Modal Sign Language & Facial Expression Recognition

This project implements models for American Sign Language and facial expression recognition. It uses various datasets and integrates real-time detection using a PyQt-based overlay.

## Datasets

- **ASL Model (wlasl_model.py):**  
  Dataset: [ASL Dataset on Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset)  
  Description: Used for training the ASL recognition model.

- **Gesture Classifier (gesture_classifier.py):**  
  Dataset: [LeapGestRecog on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)  
  Description: Used for hand gesture recognition.

- **Facial Expression (facial_expression.py):**  
  Dataset: [Facial Expression Recognition on Kaggle](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)  
  Description: Used for emotion detection in faces.

## Pretrained Models

If you prefer not to train the models from scratch, you can download the pretrained models from [this Google Drive link](https://drive.google.com/drive/folders/pretrained_models_link) and place them in the `app/models` folder as follows:
- Place the ASL model (`asl_model.keras`) in `app/models/asl/`
- Place the gesture model (`gesture_model.keras`) in `app/models/gestures/`
- Place the facial expression model (`emotion.keras`) in `app/models/emotion/`
- Place the necessary face detection files (`res10_300x300_ssd_iter_140000.caffemodel', 'deploy.prototxt`) in `app/models/face/` from [https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN]

## Installation

1. Clone this repository.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train the Models

- Run each of the following scripts to train and save the models:
  - `wlasl_model.py` – trains the ASL model and saves it as `asl_model.keras`.
  - `gesture_classifier.py` – trains the gesture classifier and saves it as `gesture_model.keras`.
  - `facial_expression.py` – trains the facial expression model and saves it as `facial_expression.keras`.

### Run the Application

- After training, launch the real-time detection app:
  ```bash
  python app/main.py
  ```

  -- main.py handles emotions
  -- gesture.py handles gestures
  -- hands_no_gesture handles hands and emotions

## Project Structure

```
/Users/jalenfrancis/HackAI2025-1/
├── app/
│   ├── main.py
│   ├── gesture.py
│   ├── hands_no_gesture.py
├──── models/
│    ├── face/
│    ├── emotion/
│    ├── gestures/
│    └── asl/ 
├── assets/
│   └── (icons and other image assets)
├── data/
│   ├── asl_dataset/
│   ├── leapGestRecog/
│   └── fer2013.csv
├── requirements.txt
├── README.md
├── wlasl_model.py
├── gesture_classifier.py
└── facial_expression.py
```

## Notes

- Place the datasets in their specified folders as indicated in the code.
- Trained models will be saved as `asl_model.keras`, `gesture_model.keras`, and `emotion.keras` in the respective subdirectories of the `models` folder.
- For any issues, feel free to open an issue or a pull request.

## Acknowledgments

- Thanks to the Kaggle community for the datasets.
- Special thanks to the developers of TensorFlow, OpenCV, PyQt5, and other libraries used.
