# Sign2Speak – Real-time AI Interpreter for the Deaf

A real-time sign language interpreter that uses your webcam to recognize **Indian Sign Language (ISL)** hand gestures and converts them to text and speech — and vice versa.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

| Mode | Input | Output |
|------|-------|--------|
| 🖐 **Sign → Text** | Camera + Hand detection | Real-time subtitles on screen |
| 🖐 **Sign → Speech** | Camera + Hand detection + TTS | Speaks for the user |
| 🗣 **Voice → Text** | Microphone (Speech Recognition) | Text displayed on screen |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Hand Detection | MediaPipe Hands (21 landmarks) |
| Gesture Classification | scikit-learn RandomForest |
| Text-to-Speech | pyttsx3 |
| Speech-to-Text | Google Speech Recognition |
| Camera / UI | OpenCV |
| Object Detection | YOLOv8 (Ultralytics) |

---

## Project Structure

```
sign2speak_yolov8/
├── app.py                          # Main application (Sign + Voice modes)
├── requirements.txt                # Python dependencies
├── .gitignore
├── models/
│   ├── hand_landmarker.task        # MediaPipe hand model (download required)
│   └── classifier.joblib           # Trained classifier (after training)
├── scripts/
│   ├── capture_images.py           # Step 1: Collect training images
│   ├── extract_keypoints.py        # Step 2: Extract hand landmarks
│   └── train_classifier.py         # Step 3: Train classifier
├── data/
│   └── images/                     # Training images organized by label
│       ├── A/
│       ├── B/
│       └── ...
└── outputs/
    └── keypoints_npz/              # Extracted keypoint vectors
        ├── A/
        ├── B/
        └── ...
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/slyfoxANDY/sign2speak_yolov8.git
cd sign2speak_yolov8
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **PyAudio note** (needed for Voice mode):
> - **Windows**: `pip install pyaudio` usually works directly
> - **Linux**: `sudo apt install portaudio19-dev && pip install pyaudio`
> - **Mac**: `brew install portaudio && pip install pyaudio`

### 3. Download the MediaPipe hand model

Download `hand_landmarker.task` into the `models/` folder:

```bash
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models/hand_landmarker.task"

# Linux / Mac
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

---

## Training Pipeline

You must train the model before sign recognition works. Follow these 3 steps:

### Step 1 — Capture Images

Capture ~200 images per sign/letter. Run once for each label:

```bash
cd scripts
python capture_images.py --label A --count 200
python capture_images.py --label B --count 200
python capture_images.py --label Hello --count 200
# ... repeat for all signs
```

- **SPACE** = capture a frame (hand landmarks shown in real-time for verification)
- **Q** = stop

Or download the [Indian Sign Language Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) from Kaggle and place it in `data/images/` with one subfolder per label.

### Step 2 — Extract Keypoints

```bash
python extract_keypoints.py
```

Runs MediaPipe Hands on every image and saves 42-dimensional keypoint vectors (21 landmarks × 2 coordinates) to `outputs/keypoints_npz/`.

### Step 3 — Train Classifier

```bash
python train_classifier.py
```

Trains a RandomForest classifier and saves it to `models/classifier.joblib`. Prints accuracy metrics on completion.

**Our results:** 97.5% validation accuracy on 42,745 samples across 35 labels (A–Z + 1–9).

---

## Running the App

```bash
python app.py
```

### Controls

| Key | Action |
|-----|--------|
| **M** | Toggle between Sign mode and Voice mode |
| **SPACE** | Add space to sentence (Sign) / Start listening (Voice) |
| **ENTER** | Speak the sentence aloud / Accept voice input |
| **C** | Clear sentence |
| **BACKSPACE** | Delete last character |
| **+** / **-** | Speed up / slow down letter confirmation |
| **Q** | Quit |

### Sign Mode

1. Show a hand sign to the camera
2. The app detects your hand and predicts the sign letter/word
3. A progress bar fills while you hold the sign steady
4. Once confirmed, the letter is added to the sentence and spoken aloud
5. Press **SPACE** between words, **ENTER** to speak the full sentence

### Voice Mode

1. Press **M** to switch to Voice mode
2. Press **SPACE** to start listening
3. Speak into your microphone
4. The recognized text appears on screen
5. Press **ENTER** to add it to the sentence and speak it

---

## Dataset Sources

| Dataset | Description |
|---------|-------------|
| [ISL Dataset (Kaggle)](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) | Indian Sign Language A–Z + 1–9, 1200 images per class |
| Self-captured | Use `capture_images.py` for custom signs |
| [RWTH-PHOENIX](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) | Full-sentence sign language (advanced) |

---

## MVP Checklist

- [x] Camera input with real-time hand detection
- [x] ISL alphabet recognition (A–Z) + digits (1–9)
- [x] Real-time text display of recognized signs
- [x] Text-to-Speech conversion (pyttsx3)
- [x] Voice-to-Text reverse communication (Google Speech Recognition)
- [x] Clean UI with accessibility focus
- [x] Adjustable confirmation speed
- [x] Training pipeline (capture → extract → train)

---

## Future Improvements

- [ ] LSTM/Transformer for recognizing sign sequences (full sentences)
- [ ] Common ISL word recognition (beyond single letters)
- [ ] Web app version using Streamlit or Flask
- [ ] Mobile app using TensorFlow Lite
- [ ] Multi-hand detection support
- [ ] Support for more regional sign languages

---

## License

MIT