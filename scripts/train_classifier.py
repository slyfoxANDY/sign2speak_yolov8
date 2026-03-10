# scripts/train_classifier.py
# -------------------------------------------------------
# Trains a RandomForest classifier on MediaPipe hand
# keypoints extracted by extract_keypoints.py.
#
# Input:  outputs/keypoints_npz/<label>/*.npz
# Output: models/classifier.joblib
# -------------------------------------------------------
import numpy as np
import joblib
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--kpdir", default="outputs/keypoints_npz")
parser.add_argument("--out", default="models/classifier.joblib")
args = parser.parse_args()

BASE = Path(__file__).resolve().parents[1]
kp_root = BASE / args.kpdir

# ---- gather labels ----
labels = sorted([d.name for d in kp_root.iterdir() if d.is_dir()])
if len(labels) < 2:
    print(f"[ERROR] Need at least 2 label folders in {kp_root}, found {len(labels)}.")
    print("Run capture_images.py and extract_keypoints.py first.")
    raise SystemExit(1)

print(f"Labels ({len(labels)}): {labels}")

# ---- load data ----
X, y = [], []
for idx, label in enumerate(labels):
    label_dir = kp_root / label
    files = list(label_dir.glob("*.npz"))
    for f in files:
        kp = np.load(f)["keypoints"].flatten()
        X.append(kp)
        y.append(idx)
    print(f"  {label}: {len(files)} samples")

X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f"\nTotal samples: {X.shape[0]}  |  Features: {X.shape[1]}")

# ---- split & train ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ---- evaluate ----
pred = clf.predict(X_val)
print("\n" + classification_report(y_val, pred, target_names=labels))

train_acc = clf.score(X_train, y_train)
val_acc = clf.score(X_val, y_val)
print(f"Train accuracy: {train_acc:.1%}")
print(f"Val   accuracy: {val_acc:.1%}")

# ---- save ----
out_path = BASE / args.out
out_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": clf, "labels": labels}, out_path)
print(f"\nModel saved to {out_path}")