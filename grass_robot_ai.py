Updated code
#!/usr/bin/env python3
"""
grass_robot_ai.py - YOLO-assisted grass cutter (Option B)

- Lightweight YOLOv5n6 for obstacle detection (lazy-loaded)
- Fast HSV-based green detection for grass patches
- Bounding boxes shown on webpage MJPEG stream
- Camera flipped 180° (upside-down mount)
- Manual motor controls preserved
- Safer defaults to reduce Pi load
"""

# limit math threads before heavy imports
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import threading
import signal
import sys
import cv2
import numpy as np
from flask import Flask, render_template_string, Response, redirect, url_for, jsonify
from gpiozero import Motor
from picamera2 import Picamera2

# Model settings (lazy load)
MODEL_PATH = "yolov5n6.pt"   # put this file in same dir
USE_YOLO = True              # start with intention to use YOLO if model loads
YOLO_SKIP_FRAMES = 5         # run YOLO every N frames
INFER_SIZE = (320, 224)      # (width, height) for inference to reduce load

# Globals for lazy loading
YOLO_MODULE = None
model = None

# Flask app
app = Flask(__name__)

# -----------------------
# Motor setup (L298N)
# -----------------------
grass_motor = Motor(forward=5, backward=6)
right_motor = Motor(forward=11, backward=9)
left_motor = Motor(forward=27, backward=22)

# -----------------------
# Shared state and locks
# -----------------------
ai_running = False
stop_all = False

_shared_frame = None
_shared_frame_lock = threading.Lock()

_processed_frame = None
_processed_frame_lock = threading.Lock()

_shared_detections = None
_shared_detections_lock = threading.Lock()

# -----------------------
# HTML page
# -----------------------
html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Grass Cutter Robot - YOLO Assisted</title>
  <style>
    body { font-family: Arial, background:#222; color:#eee; text-align:center; }
    button { padding:12px 20px; margin:6px; font-size:15px; border:none; border-radius:8px; cursor:pointer; }
    .fwd { background:#4CAF50; color:#fff; }
    .rev { background:#f44336; color:#fff; }
    .stop { background:#555; color:#fff; }
    .ai { background:#2196F3; color:#fff; }
    .emg { background:#ff0000; color:#fff; }
    img { border-radius:8px; box-shadow: 0 0 12px rgba(0,0,0,0.5); max-width: 95%; height:auto; }
    .controls { display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:12px; }
    h1,h2 { margin:8px 0; }
    #status { margin-top:10px; font-size:14px; }
  </style>
</head>
<body>
  <h1>Grass Cutter Robot - YOLO Assisted</h1>

  <h2>Live Camera Feed</h2>
  <img id="video" src="{{ url_for('video_feed') }}" alt="camera">

  <div id="status">AI: <span id="ai_state">OFF</span> — Model: <span id="model_state">{model}</span></div>

  <div class="controls">
    <form action="/grass_forward" method="post"><button class="fwd">Grass ON</button></form>
    <form action="/grass_stop" method="post"><button class="stop">Grass OFF</button></form>

    <form action="/right_forward" method="post"><button class="fwd">Right Fwd</button></form>
    <form action="/right_reverse" method="post"><button class="rev">Right Rev</button></form>
    <form action="/right_stop" method="post"><button class="stop">Right Stop</button></form>

    <form action="/left_forward" method="post"><button class="fwd">Left Fwd</button></form>
    <form action="/left_reverse" method="post"><button class="rev">Left Rev</button></form>
    <form action="/left_stop" method="post"><button class="stop">Left Stop</button></form>
  </div>

  <h2>AI Automation</h2>
  <div class="controls">
    <form action="/start_ai" method="post"><button class="ai">Start AI (YOLO)</button></form>
    <form action="/stop_ai" method="post"><button class="stop">Stop AI</button></form>
  </div>

  <h2>Emergency</h2>
  <div class="controls">
    <form action="/emergency_stop" method="post"><button class="emg">EMERGENCY STOP</button></form>
  </div>

  <script>
    function updateStatus() {
      fetch('/status').then(r=>r.json()).then(j=>{
        document.getElementById('ai_state').innerText = j.ai ? 'ON' : 'OFF';
        document.getElementById('model_state').innerText = j.model || 'none';
      }).catch(()=>{});
    }
    setInterval(()=>{
      const img = document.getElementById('video');
      img.src = '{{ url_for("video_feed") }}' + '?t=' + Date.now();
      updateStatus();
    }, 800);
  </script>
</body>
</html>
""".format(model=MODEL_PATH)

# -----------------------
# Utilities
# -----------------------
def shutdown_all_motors():
    try:
        grass_motor.stop()
        left_motor.stop()
        right_motor.stop()
    except Exception:
        pass

def signal_handler(sig, frame):
    global stop_all, ai_running
    print("[MAIN] Signal received, shutting down...")
    stop_all = True
    ai_running = False
    shutdown_all_motors()
    time.sleep(0.5)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -----------------------
# Camera thread (flip + ensure BGR)
# -----------------------
def camera_thread():
    global _shared_frame, stop_all
    print("[CAM] Starting Picamera2...")
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (640, 480)})
    picam.configure(config)
    picam.start()
    time.sleep(0.8)
    try:
        while not stop_all:
            frame = picam.capture_array()
            if frame is None:
                time.sleep(0.01)
                continue
            # rotate 180°
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            # if 4 channels -> drop alpha (BGRA->BGR)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            # convert RGB->BGR defensively if needed
            try:
                r_mean = float(np.mean(frame[:, :, 0]))
                b_mean = float(np.mean(frame[:, :, 2]))
                if r_mean - b_mean > 12:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                pass
            with _shared_frame_lock:
                _shared_frame = frame
            time.sleep(0.01)
    except Exception as e:
        print("[CAM] Exception:", e)
    finally:
        try:
            picam.stop()
        except Exception:
            pass
        print("[CAM] stopped")

# -----------------------
# Lazy model loader
# -----------------------
def load_model_thread():
    """Load YOLO in background; set global model and YOLO_MODULE."""
    global YOLO_MODULE, model, USE_YOLO
    try:
        from ultralytics import YOLO as _YOLO
        YOLO_MODULE = _YOLO
        model = YOLO_MODULE(MODEL_PATH)
        # try to limit torch threads if available
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        print("[MODEL] Loaded model:", MODEL_PATH)
    except Exception as e:
        print("[MODEL] Failed to load model:", e)
        USE_YOLO = False

# -----------------------
# Detection thread (HSV grass + optional YOLO obstacles)
# -----------------------
def detection_thread():
    global _shared_frame, _processed_frame, _shared_detections, stop_all, ai_running, USE_YOLO, model
    print("[DETECT] thread started")

    hsv_low = np.array([30, 40, 40])
    hsv_high = np.array([90, 255, 255])

    frame_count = 0
    yolo_err_count = 0
    YOLO_ERR_LIMIT = 3

    while not stop_all:
        if not ai_running:
            # publish raw frame for stream
            with _shared_frame_lock:
                f = None if _shared_frame is None else _shared_frame.copy()
            if f is not None:
                with _processed_frame_lock:
                    _processed_frame = f
            time.sleep(0.05)
            continue

        with _shared_frame_lock:
            frame_full = None if _shared_frame is None else _shared_frame.copy()
        if frame_full is None:
            time.sleep(0.02)
            continue

        frame = frame_full.copy()
        detections_info = {"obstacles": [], "grass_bbox": None}

        # HSV-based grass detection on full frame
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_low, hsv_high)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 1200:
                    x, y, w, h = cv2.boundingRect(c)
                    detections_info["grass_bbox"] = (x, y, x + w, y + h, area)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"GRASS {int(area)}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        except Exception as e:
            print("[DETECT] HSV error:", e)

        # YOLO obstacle detection (running on downsampled image periodically)
        if USE_YOLO:
            frame_count += 1
            if frame_count >= YOLO_SKIP_FRAMES:
                frame_count = 0
                # lazy load model if not yet present
                if model is None and USE_YOLO:
                    threading.Thread(target=load_model_thread, daemon=True).start()
                    time.sleep(0.2)  # short give time, actual loading may take longer
                if model is not None:
                    try:
                        small = cv2.resize(frame_full, INFER_SIZE)
                        if small.ndim == 3 and small.shape[2] == 4:
                            small = small[:, :, :3]
                        results = model.predict(small, stream=False, verbose=False)
                        with _shared_detections_lock:
                            _shared_detections = results
                        # annotate boxes (scale back to full frame)
                        try:
                            for r in results:
                                for box in r.boxes:
                                    conf = float(box.conf)
                                    if conf < 0.35:
                                        continue
                                    cls = int(box.cls)
                                    label = model.names[cls] if hasattr(model, "names") else str(cls)
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    sx = frame_full.shape[1] / INFER_SIZE[0]
                                    sy = frame_full.shape[0] / INFER_SIZE[1]
                                    X1 = int(x1 * sx); Y1 = int(y1 * sy); X2 = int(x2 * sx); Y2 = int(y2 * sy)
                                    detections_info["obstacles"].append((label, conf, (X1, Y1, X2, Y2)))
                                    cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
                                    cv2.putText(frame, f"{label} {conf:.2f}", (X1, Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        except Exception:
                            pass
                        yolo_err_count = 0
                    except Exception as ye:
                        yolo_err_count += 1
                        print("[DETECT] YOLO error:", ye, "count:", yolo_err_count)
                        if yolo_err_count >= YOLO_ERR_LIMIT:
                            print("[DETECT] disabling YOLO due to repeated errors")
                            USE_YOLO = False

        # publish processed frame and detections
        with _shared_detections_lock:
            _shared_detections = detections_info
        with _processed_frame_lock:
            _processed_frame = frame

        # motor decision: obstacle => stop, else grass => forward else stop
        try:
            obst = detections_info.get("obstacles")
            grass = detections_info.get("grass_bbox")
            if obst and len(obst) > 0:
                # stop and avoid (simple stop here)
                grass_motor.stop()
                left_motor.stop()
                right_motor.stop()
            else:
                if grass:
                    grass_motor.forward()
                    left_motor.forward()
                    right_motor.forward()
                else:
                    grass_motor.stop()
                    left_motor.stop()
                    right_motor.stop()
        except Exception as me:
            print("[MOTOR] error", me)

        time.sleep(0.02)

    print("[DETECT] thread exiting")

# -----------------------
# Motor thread (kept minimal; detection thread drives motors)
# -----------------------
def motor_thread():
    while not stop_all:
        time.sleep(0.5)

# -----------------------
# MJPEG stream generator
# -----------------------
def gen_frames():
    global _processed_frame, _shared_frame
    while not stop_all:
        with _processed_frame_lock:
            frame = None if _processed_frame is None else _processed_frame.copy()
        if frame is None:
            with _shared_frame_lock:
                frame = None if _shared_frame is None else _shared_frame.copy()
        if frame is None:
            time.sleep(0.02)
            continue
        try:
            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ret:
                time.sleep(0.02)
                continue
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                jpeg.tobytes() +
                b'\r\n'
            )
        except Exception:
            time.sleep(0.02)
            continue

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def index():
    return render_template_string(html)

@app.route("/status")
def status():
    return jsonify({"ai": ai_running, "model": (MODEL_PATH if model is not None else "none")})

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Manual motor controls
@app.route("/grass_forward", methods=["POST"])
def grass_forward():
    grass_motor.forward()
    return redirect(url_for("index"))

@app.route("/grass_stop", methods=["POST"])
def grass_stop():
    grass_motor.stop()
    return redirect(url_for("index"))

@app.route("/right_forward", methods=["POST"])
def right_forward():
    right_motor.forward()
    return redirect(url_for("index"))

@app.route("/right_reverse", methods=["POST"])
def right_reverse():
    right_motor.backward()
    return redirect(url_for("index"))

@app.route("/right_stop", methods=["POST"])
def right_stop():
    right_motor.stop()
    return redirect(url_for("index"))

@app.route("/left_forward", methods=["POST"])
def left_forward():
    left_motor.forward()
    return redirect(url_for("index"))

@app.route("/left_reverse", methods=["POST"])
def left_reverse():
    left_motor.backward()
    return redirect(url_for("index"))

@app.route("/left_stop", methods=["POST"])
def left_stop():
    left_motor.stop()
    return redirect(url_for("index"))

# AI controls
@app.route("/start_ai", methods=["POST"])
def start_ai():
    global ai_running, USE_YOLO
    ai_running = True
    # try to start model loader (background) if YOLO intended
    if USE_YOLO and model is None:
        threading.Thread(target=load_model_thread, daemon=True).start()
    return redirect(url_for("index"))

@app.route("/stop_ai", methods=["POST"])
def stop_ai():
    global ai_running
    ai_running = False
    shutdown_all_motors()
    return redirect(url_for("index"))

@app.route("/emergency_stop", methods=["POST"])
def emergency_stop():
    global stop_all, ai_running
    stop_all = True
    ai_running = False
    shutdown_all_motors()
    return "EMERGENCY STOPPED"

# -----------------------
# Start threads and run Flask
# -----------------------
if __name__ == "__main__":
    try:
        threading.Thread(target=camera_thread, daemon=True).start()
        threading.Thread(target=detection_thread, daemon=True).start()
        threading.Thread(target=motor_thread, daemon=True).start()
        app.run(host="0.0.0.0", port=5000, threaded=True)
    except Exception as e:
        print("[MAIN] Fatal error:", e)
    finally:
        stop_all = True
        ai_running = False
        shutdown_all_motors()
        time.sleep(0.2)
        print("[MAIN] Exiting.")
