# Sign2Speak – Real-time AI Interpreter for the Deaf

A real-time sign language interpreter that uses your webcam to recognize **Indian Sign Language (ISL)** hand gestures and converts them to text and speech — and vice versa.

## Features

| Mode | Input | Output |
|------|-------|--------|
| **Sign → Text** | Camera + Hand detection | Real-time subtitles |
| **Sign → Speech** | Camera + Hand detection + TTS | Speaks for the user |
| **Voice → Text** | Microphone (Speech Recognition) | Text on screen |

## Tech Stack

- **Computer Vision**: MediaPipe Hands (21 hand landmarks)
- **ML Classifier**: scikit-learn RandomForest on hand keypoints
- **Text-to-Speech**: pyttsx3
- **Speech-to-Text**: Google Speech Recognition
- **Camera**: OpenCV

## Project Structure

```
sign2speak_yolov8/
├── app.py                      # Main application
├── requirements.txt
├── models/
│   └── classifier.joblib       # Trained model (after training)
├── scripts/
│   ├── capture_images.py       # Step 1: Collect training images
│   ├── extract_keypoints.py    # Step 2: Extract hand landmarks
│   └── train_classifier.py     # Step 3: Train classifier
├── data/
│   └── images/                 # Training images (per label folder)
└── outputs/
    └── keypoints_npz/          # Extracted keypoints (per label folder)
```

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** PyAudio may require additional setup on some systems.
> - **Windows**: `pip install pyaudio` (usually works directly)
> - **Linux**: `sudo apt install portaudio19-dev && pip install pyaudio`
> - **Mac**: `brew install portaudio && pip install pyaudio`

## Training Pipeline

You need to train the model before sign recognition works. Follow these 3 steps:

### Step 1 — Capture Images

Capture ~200 images per sign/letter. Run once for each label:

```bash
cd scripts
python capture_images.py --label A --count 200
python capture_images.py --label B --count 200
python capture_images.py --label Hello --count 200
# ... repeat for all signs you want to recognize
```

- Press **SPACE** to capture a frame (landmarks shown in real-time)
- Press **Q** to stop
- Images are saved to `data/images/<label>/`

### Step 2 — Extract Keypoints

```bash
python extract_keypoints.py
```

This runs MediaPipe Hands on every captured image and saves 42-dimensional keypoint vectors (21 landmarks × 2 coordinates) to `outputs/keypoints_npz/`.

### Step 3 — Train Classifier

```bash
python train_classifier.py
```

Trains a RandomForest classifier and saves it to `models/classifier.joblib`. Shows accuracy report on completion.

## Running the App

```bash
python app.py
```

### Controls

| Key | Action |
|-----|--------|
| **M** | Toggle between Sign and Voice mode |
| **SPACE** | Add space to sentence (Sign mode) / Start listening (Voice mode) |
| **ENTER** | Speak the sentence aloud (Sign mode) / Accept voice text (Voice mode) |
| **C** | Clear sentence |
| **BACKSPACE** | Delete last character |
| **+/-** | Adjust letter confirmation speed |
| **Q** | Quit |

### How Sign Mode Works

1. Show a hand sign to the camera
2. The app detects your hand and extracts landmarks
3. The classifier predicts the sign letter/word
4. Hold the sign steady — a progress bar fills up
5. Once confirmed, the letter is added to the sentence and spoken aloud
6. Press **SPACE** between words, **ENTER** to speak the full sentence

### How Voice Mode Works

1. Press **M** to switch to Voice mode
2. Press **SPACE** to start listening
3. Speak into your microphone
4. The recognized text appears on screen
5. Press **ENTER** to add it to the sentence and speak it

## Dataset

For ISL training data, you can use:
- Self-captured images using `capture_images.py`
- [Indian Sign Language Dataset](https://www.kaggle.com/datasets) on Kaggle
- [RWTH-PHOENIX-Weather Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) for full-sentence SL

## License

MIT