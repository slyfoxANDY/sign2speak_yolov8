# scripts/extract_keypoints.py
# -------------------------------------------------------
# Extracts 21 hand landmarks (x, y) per image using
# MediaPipe Hands.  Images must be organized as:
#   data/images/<label>/*.jpg
# Output: outputs/keypoints_npz/<label>/*.npz
# -------------------------------------------------------
import numpy as np
import cv2
import argparse
import mediapipe as mp
from mediapipe.tasks.python import vision
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--images", default="data/images",
                    help="folder with labelled image subfolders")
parser.add_argument("--out", default="outputs/keypoints_npz")
args = parser.parse_args()

BASE = Path(__file__).resolve().parents[1]
images_root = BASE / args.images
out_root = BASE / args.out
out_root.mkdir(parents=True, exist_ok=True)

# MediaPipe HandLandmarker (Tasks API) — IMAGE mode for static images
model_path = str(BASE / "models" / "hand_landmarker.task")
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE,
)
landmarker = vision.HandLandmarker.create_from_options(options)

total, skipped = 0, 0

for label_dir in sorted(images_root.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    dst = out_root / label
    dst.mkdir(parents=True, exist_ok=True)

    # Skip labels already fully processed
    existing = set(f.stem for f in dst.glob("*.npz"))
    images = sorted(label_dir.glob("*.jpg"))
    remaining = [f for f in images if f.stem not in existing]
    if not remaining:
        print(f"[{label}] Already done ({len(existing)} files), skipping.")
        total += len(existing)
        continue

    print(f"[{label}] Processing {len(remaining)} images ...")
    for img_f in remaining:
        img = cv2.imread(str(img_f))
        if img is None:
            continue
        total += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            kp = []
            for lm in hand:
                kp.extend([lm.x, lm.y])  # already normalized 0-1
            arr = np.array(kp, dtype=np.float32)
        else:
            arr = np.zeros(21 * 2, dtype=np.float32)
            skipped += 1

        outp = dst / (img_f.stem + ".npz")
        np.savez_compressed(outp, keypoints=arr)
        print(f"[{label}] Saved {outp.name}")

landmarker.close()
print(f"\nDone — {total} images processed, {skipped} had no hand detected.")