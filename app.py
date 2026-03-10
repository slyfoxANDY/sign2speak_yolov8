"""
Sign2Speak - Real-time AI Interpreter for the Deaf
===================================================
Uses MediaPipe Hands for hand landmark detection, a trained ML classifier
for ISL gesture recognition, pyttsx3 for Text-to-Speech, and
SpeechRecognition for Voice-to-Text.

Controls:
  M       Toggle Sign / Voice mode
  SPACE   Add a space (sign mode) / Start listening (voice mode)
  ENTER   Speak the sentence aloud / Accept voice text (voice mode)
  C       Clear sentence
  BACK    Delete last character
  +/-     Adjust confirmation speed
  Q       Quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, HandLandmarksConnections
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading
import queue
import time
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW = "Sign2Speak - AI Sign Language Interpreter"
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 2
CONFIRM_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.45
COLORS = {
    "accent": (0, 200, 100),
    "warn":   (0, 180, 255),
    "text":   (255, 255, 255),
    "dim":    (160, 160, 160),
    "err":    (80, 80, 255),
    "panel":  (30, 30, 30),
}

# Hand connections for drawing
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS


# ---------------------------------------------------------------------------
# Thread-safe TTS wrapper
# ---------------------------------------------------------------------------
class _TTSWorker:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        voices = engine.getProperty("voices")
        for v in voices:
            if "female" in v.name.lower() or "zira" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        while True:
            text = self._q.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def speak(self, text: str):
        self._q.put(text)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class Sign2Speak:
    def __init__(self):
        # -- MediaPipe Hands (Tasks API) --------------------------------------
        model_path = str(Path(__file__).resolve().parent / "models" / "hand_landmarker.task")
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_ts = 0

        # -- Classifier -------------------------------------------------------
        self.classifier = None
        self.labels: list[str] = []
        self._load_classifier()

        # -- TTS / STT --------------------------------------------------------
        self.tts = _TTSWorker()
        self.recognizer = sr.Recognizer()

        # -- UI state ----------------------------------------------------------
        self.mode = "sign"
        self.current_pred = ""
        self.confidence = 0.0
        self.sentence = ""
        self.stable_count = 0
        self.last_stable = ""
        self.confirmed = False
        self.confirm_frames = CONFIRM_FRAMES

        self.is_listening = False
        self.voice_text = ""
        self.voice_status = ""

        self._status_msg = ""
        self._status_ts = 0.0

    def _load_classifier(self):
        base = Path(__file__).resolve().parent / "models"
        for name in ("classifier.joblib", "classifier.pkl"):
            path = base / name
            if not path.exists():
                continue
            try:
                data = joblib.load(path)
                if isinstance(data, dict):
                    self.classifier = data.get("model")
                    self.labels = data.get("labels", [])
                else:
                    self.classifier = data
                self._flash(f"Model loaded: {name}")
                return
            except Exception as exc:
                print(f"[warn] Could not load {name}: {exc}")
        self._flash("No trained model found - collect data & train first")

    @staticmethod
    def _extract_keypoints(result) -> np.ndarray | None:
        if not result.hand_landmarks:
            return None
        hand = result.hand_landmarks[0]
        kp = []
        for lm in hand:
            kp.extend([lm.x, lm.y])
        return np.array(kp, dtype=np.float32)

    def _predict(self, kp: np.ndarray | None):
        if self.classifier is None or kp is None:
            return "", 0.0
        try:
            x = kp.reshape(1, -1)
            idx = self.classifier.predict(x)[0]
            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            conf = 0.0
            if hasattr(self.classifier, "predict_proba"):
                conf = float(np.max(self.classifier.predict_proba(x)[0]))
            return label, conf
        except Exception:
            return "", 0.0

    def _update_stability(self, pred: str):
        if self.confidence < CONFIDENCE_THRESHOLD:
            self.stable_count = 0
            return
        if pred == self.last_stable and pred != "":
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_stable = pred
            self.confirmed = False
        if self.stable_count >= self.confirm_frames and not self.confirmed:
            self.confirmed = True
            self.sentence += pred
            self.tts.speak(pred)
            self._flash(f"Added: {pred}")

    def _listen_voice(self):
        if self.is_listening:
            return
        self.is_listening = True
        self.voice_status = "Adjusting for ambient noise ..."

        def _worker():
            try:
                with sr.Microphone() as src:
                    self.recognizer.adjust_for_ambient_noise(src, duration=0.5)
                    self.voice_status = "Listening ... speak now!"
                    audio = self.recognizer.listen(src, timeout=5,
                                                   phrase_time_limit=10)
                    self.voice_status = "Processing ..."
                    text = self.recognizer.recognize_google(audio)
                    self.voice_text = text
                    self.voice_status = "Done!"
            except sr.WaitTimeoutError:
                self.voice_text = ""
                self.voice_status = "Timeout - no speech detected"
            except sr.UnknownValueError:
                self.voice_text = ""
                self.voice_status = "Could not understand audio"
            except sr.RequestError as e:
                self.voice_text = ""
                self.voice_status = f"API error: {str(e)[:50]}"
            except Exception as e:
                self.voice_text = ""
                self.voice_status = f"Error: {str(e)[:50]}"
            finally:
                self.is_listening = False

        threading.Thread(target=_worker, daemon=True).start()

    def _flash(self, msg: str):
        self._status_msg = msg
        self._status_ts = time.time()

    # =====================================================================
    #  Drawing helpers
    # =====================================================================
    @staticmethod
    def _panel(frame, x, y, w, h, alpha=0.65):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLORS["panel"], -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _put(frame, text, pos, scale=0.6, color=None, thick=1,
             font=cv2.FONT_HERSHEY_SIMPLEX):
        color = color or COLORS["text"]
        cv2.putText(frame, text, pos, font, scale, color, thick, cv2.LINE_AA)

    def _draw_top_bar(self, frame, w):
        self._panel(frame, 0, 0, w, 52, 0.82)
        self._put(frame, "Sign2Speak", (12, 37), 1.0, COLORS["accent"], 2)
        mode_label = "SIGN LANGUAGE" if self.mode == "sign" else "VOICE INPUT"
        self._put(frame, f"Mode: {mode_label}", (w - 340, 37), 0.65,
                  COLORS["text"], 2)

    def _draw_sign_panel(self, frame, fh):
        self._panel(frame, 10, 62, 210, 165, 0.72)
        self._put(frame, "Detected:", (20, 90), 0.55, COLORS["dim"])
        if self.current_pred:
            clr = COLORS["accent"] if self.confirmed else COLORS["warn"]
            self._put(frame, self.current_pred, (28, 162),
                      2.2, clr, 3, cv2.FONT_HERSHEY_DUPLEX)
            progress = min(self.stable_count / max(self.confirm_frames, 1), 1.0)
            bar_x, bar_y, bar_w = 20, 180, 190
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + 14), COLORS["dim"], 1)
            fill_clr = COLORS["accent"] if self.confirmed else COLORS["warn"]
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + 14),
                          fill_clr, -1)
            if self.confidence > 0:
                self._put(frame, f"Conf: {self.confidence:.0%}",
                          (22, 215), 0.45, COLORS["dim"])
        else:
            self._put(frame, "---", (70, 162), 1.6, COLORS["dim"], 2,
                      cv2.FONT_HERSHEY_DUPLEX)
        if self.classifier is None:
            self._panel(frame, 10, fh // 2 - 25, 460, 50, 0.75)
            self._put(frame, "No model loaded. Collect data & train first!",
                      (20, fh // 2 + 5), 0.6, COLORS["err"], 2)

    def _draw_voice_panel(self, frame, fw, fh):
        pw, ph = 500, 220
        px, py = (fw - pw) // 2, (fh - ph) // 2 - 40
        self._panel(frame, px, py, pw, ph, 0.82)
        self._put(frame, "Voice-to-Text Mode", (px + 20, py + 38),
                  0.85, COLORS["accent"], 2)
        if self.is_listening:
            pulse = int(200 + 55 * np.sin(time.time() * 5))
            cv2.circle(frame, (px + 30, py + 80), 10, (0, 0, pulse), -1)
            self._put(frame, self.voice_status, (px + 50, py + 85),
                      0.55, COLORS["warn"])
        else:
            self._put(frame, "Press SPACE to start listening",
                      (px + 50, py + 85), 0.55, COLORS["dim"])
        if self.voice_text:
            self._put(frame, "Heard:", (px + 20, py + 130), 0.55, COLORS["dim"])
            for i, start in enumerate(range(0, len(self.voice_text), 45)):
                chunk = self.voice_text[start:start + 45]
                self._put(frame, chunk, (px + 20, py + 160 + i * 28),
                          0.65, COLORS["text"], 2)
                if i >= 2:
                    break

    def _draw_bottom_bar(self, frame, fw, fh):
        self._panel(frame, 0, fh - 105, fw, 105, 0.82)
        self._put(frame, "Sentence:", (12, fh - 72), 0.55, COLORS["dim"])
        disp = self.sentence if self.sentence else "(empty)"
        if int(time.time() * 2) % 2 == 0:
            disp += "_"
        max_chars = (fw - 140) // 14
        if len(disp) > max_chars:
            disp = "..." + disp[-(max_chars - 3):]
        self._put(frame, disp, (130, fh - 72), 0.7, COLORS["text"], 2)
        if self.mode == "sign":
            keys = ("SPACE: Add space | ENTER: Speak sentence | "
                    "C: Clear | BKSP: Delete | M: Voice mode | "
                    "+/-: Speed | Q: Quit")
        else:
            keys = ("SPACE: Listen | ENTER: Accept + Speak | "
                    "C: Clear | M: Sign mode | Q: Quit")
        self._put(frame, keys, (12, fh - 12), 0.40, COLORS["dim"])

    def _draw_status(self, frame, fw):
        if self._status_msg and time.time() - self._status_ts < 3.0:
            tw = len(self._status_msg) * 11 + 30
            x = (fw - tw) // 2
            self._panel(frame, x, 56, tw, 32, 0.80)
            self._put(frame, self._status_msg, (x + 15, 78),
                      0.50, COLORS["accent"])

    def _draw_ui(self, frame):
        fh, fw = frame.shape[:2]
        self._draw_top_bar(frame, fw)
        if self.mode == "sign":
            self._draw_sign_panel(frame, fh)
        else:
            self._draw_voice_panel(frame, fw, fh)
        self._draw_bottom_bar(frame, fw, fh)
        self._draw_status(frame, fw)

    # =====================================================================
    #  Main loop
    # =====================================================================
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("[ERROR] Cannot open camera.")
            return

        print("=" * 55)
        print("  Sign2Speak - Real-time AI Sign Language Interpreter")
        print("=" * 55)
        print("  M      - Toggle Sign / Voice mode")
        print("  SPACE  - Add space (sign) / Listen (voice)")
        print("  ENTER  - Speak sentence / Accept voice input")
        print("  C      - Clear sentence")
        print("  BKSP   - Delete last character")
        print("  +/-    - Faster / slower confirmation")
        print("  Q      - Quit")
        print("=" * 55)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if self.mode == "sign":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self._frame_ts += 33
                result = self.hand_landmarker.detect_for_video(mp_image, self._frame_ts)

                if result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        drawing_utils.draw_landmarks(
                            frame, hand_lms, HAND_CONNECTIONS)
                    kp = self._extract_keypoints(result)
                    pred, conf = self._predict(kp)
                    self.current_pred = pred
                    self.confidence = conf
                    self._update_stability(pred)
                else:
                    self.current_pred = ""
                    self.confidence = 0.0
                    self.stable_count = 0

            self._draw_ui(frame)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                self.mode = "voice" if self.mode == "sign" else "sign"
                self._flash(f"Switched to "
                            f"{'Voice' if self.mode == 'voice' else 'Sign'}"
                            " mode")
            elif key == 32:  # SPACE
                if self.mode == "sign":
                    self.sentence += " "
                else:
                    self._listen_voice()
            elif key == 13:  # ENTER
                if self.mode == "voice" and self.voice_text:
                    self.sentence += self.voice_text + " "
                    self.voice_text = ""
                if self.sentence.strip():
                    self.tts.speak(self.sentence.strip())
                    self._flash("Speaking ...")
            elif key == ord("c"):
                self.sentence = ""
                self.voice_text = ""
                self._flash("Cleared")
            elif key == 8:  # BACKSPACE
                self.sentence = self.sentence[:-1]
            elif key == ord("+") or key == ord("="):
                self.confirm_frames = max(5, self.confirm_frames - 5)
                self._flash(f"Confirm speed: {self.confirm_frames} frames")
            elif key == ord("-"):
                self.confirm_frames = min(60, self.confirm_frames + 5)
                self._flash(f"Confirm speed: {self.confirm_frames} frames")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Sign2Speak().run()
