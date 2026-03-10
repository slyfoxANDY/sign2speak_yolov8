# scripts/capture_images.py
# -------------------------------------------------------
# Capture labelled images for training.
# Shows MediaPipe hand landmarks in real-time so you can
# verify your hand is detected before pressing SPACE.
#
# Usage:
#   python capture_images.py --label A --count 200
# -------------------------------------------------------
import cv2
import argparse
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/images",
                    help="output folder (relative to project root)")
parser.add_argument("--label", required=True,
                    help="label name — creates a subfolder")
parser.add_argument("--count", type=int, default=200,
                    help="max images to capture")
args = parser.parse_args()

BASE = Path(__file__).resolve().parents[1]
label_dir = BASE / args.out / args.label
label_dir.mkdir(parents=True, exist_ok=True)

# MediaPipe HandLandmarker (Tasks API)
model_path = str(BASE / "models" / "hand_landmarker.task")
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.HandLandmarker.create_from_options(options)
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

cap = cv2.VideoCapture(0)
idx = len(list(label_dir.glob("*.jpg")))
frame_ts = 0

print(f"Capturing label '{args.label}' -> {label_dir}")
print(f"Already have {idx} images.  Target: {args.count}")
print("SPACE = capture  |  Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Detect and draw hand landmarks on display copy
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 33
    result = landmarker.detect_for_video(mp_image, frame_ts)
    hand_detected = False
    if result.hand_landmarks:
        hand_detected = True
        for hand_lms in result.hand_landmarks:
            drawing_utils.draw_landmarks(
                display, hand_lms, HAND_CONNECTIONS)

    # Status overlay
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "Hand OK" if hand_detected else "No hand"
    cv2.putText(display, f"Label: {args.label}  |  Count: {idx}/{args.count}"
                f"  |  {status_text}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow("Capture - Sign2Speak", display)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == 32:  # SPACE — save the raw frame (no landmarks baked in)
        fname = label_dir / f"{args.label}_{idx:04d}.jpg"
        cv2.imwrite(str(fname), frame)
        print(f"  Saved {fname.name}  ({idx + 1}/{args.count})")
        idx += 1
        if idx >= args.count:
            print("Target reached!")
            break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print(f"Done — {idx} images in {label_dir}")