import cv2
import numpy as np
import os
import time
import logging
import requests
from insightface.app import FaceAnalysis
from database import mark_attendance

# ---------------- CONFIG ----------------
GALLERY_DIR = "gallery"
SIMILARITY_THRESHOLD = 0.5
COOLDOWN_SECONDS = 10
FRAME_SKIP = 2
RL_TRIGGER_INTERVAL = 5  # seconds

DASHBOARD_API = "http://127.0.0.1:5000/api/attendance"
RECOMPUTE_API = "http://127.0.0.1:5000/api/recompute"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------- UTILS ----------------
def normalize(v):
    return v / np.linalg.norm(v)

def cosine_sim(a, b):
    return np.dot(a, b)

# ---------------- LOAD GALLERY ----------------
def load_gallery(app):
    gallery = {}
    os.makedirs(GALLERY_DIR, exist_ok=True)

    for file in os.listdir(GALLERY_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            wid = os.path.splitext(file)[0]
            img = cv2.imread(os.path.join(GALLERY_DIR, file))

            if img is None:
                continue

            faces = app.get(img)
            if not faces:
                continue

            gallery[wid] = normalize(faces[0].embedding)
            logging.info(f"Loaded {wid}")

    return gallery

# ---------------- TRACKER ----------------
class WorkerTracker:
    def __init__(self):
        self.last_seen = {}
        self.active_workers = set()

    def update(self, worker_id):
        now = time.time()

        self.last_seen[worker_id] = now
        self.active_workers.add(worker_id)

    def cleanup(self):
        """Remove workers not seen recently"""
        now = time.time()
        removed = []

        for w in list(self.active_workers):
            if now - self.last_seen.get(w, 0) > COOLDOWN_SECONDS:
                self.active_workers.remove(w)
                removed.append(w)

        return removed

    def get_active(self):
        return list(self.active_workers)

# ---------------- DASHBOARD ----------------
def push_to_dashboard(worker_id, confidence):
    try:
        requests.post(DASHBOARD_API, json={
            "worker_id": worker_id,
            "confidence": confidence
        }, timeout=0.3)
    except:
        pass

# ---------------- RL TRIGGER ----------------
last_rl_trigger = 0

def trigger_rl_update():
    global last_rl_trigger
    now = time.time()

    if now - last_rl_trigger < RL_TRIGGER_INTERVAL:
        return

    try:
        requests.get(RECOMPUTE_API, timeout=0.3)
        last_rl_trigger = now
        logging.info("🔁 RL Recomputed")
    except:
        pass

# ---------------- MAIN ----------------
def run_camera_system():
    logging.info("Initializing model...")

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    gallery = load_gallery(app)
    tracker = WorkerTracker()

    cap = cv2.VideoCapture(0)
    frame_count = 0

    logging.info("Camera started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Smart Factory AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        faces = app.get(frame)

        detected_this_frame = set()

        for face in faces:
            bbox = face.bbox.astype(int)
            emb = normalize(face.embedding)

            best_id = "Unknown"
            best_score = 0

            for wid, ref in gallery.items():
                score = cosine_sim(emb, ref)
                if score > best_score:
                    best_score = score
                    best_id = wid

            if best_score > SIMILARITY_THRESHOLD:
                color = (0,255,0)
                label = f"{best_id} ({best_score:.2f})"

                detected_this_frame.add(best_id)

                tracker.update(best_id)

                # Only mark if strong confidence
                if best_score > 0.6:
                    confidence = float(min(0.99, best_score))

                    mark_attendance(best_id, confidence)
                    push_to_dashboard(best_id, confidence)

            else:
                color = (0,0,255)
                label = "Unknown"

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), color, 2)

            cv2.putText(frame, label,
                        (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 🔥 Remove inactive workers
        removed = tracker.cleanup()
        if removed:
            logging.info(f"Workers left: {removed}")

        # 🔥 Trigger RL only if workforce changes
        if detected_this_frame:
            trigger_rl_update()

        cv2.imshow("Smart Factory AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_system()