"""
OCR Tool & Gauge Reader

Fixes applied vs. the original version:
  1. Tesseract path is resolved portably (PATH, then common install
     locations) instead of a hardcoded path that only existed on one
     machine. If it can't be found, the OCR error is surfaced instead
     of silently swallowed.
  2. Gauge analysis is no longer run unconditionally on every image.
     A real circular "dial bezel" is detected first (cv2.HoughCircles).
     If no circle is found, gauge_found = False immediately -- no
     fabricated reading is produced for documents/bills/photos.
  3. Needle detection now requires the winning ray to clearly dominate
     the background noise, instead of just being the single best of
     360 candidates (which is true for almost any image with texture).
  4. Unit detection uses word-boundary regex instead of matching a
     bare "c" anywhere in the OCR text.
"""

import os
import re
import json
import uuid
import time
import math
import shutil
import platform
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import pytesseract

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ── Tesseract path resolution (portable) ────────────────────────────────────
def resolve_tesseract_cmd():
    """Find tesseract on PATH first; fall back to common Windows install
    locations. Returns None (instead of crashing) if not found, so the
    app still boots and reports a clear error per-request."""
    env_override = os.environ.get("TESSERACT_CMD")
    if env_override and os.path.exists(env_override):
        return env_override

    found = shutil.which("tesseract")
    if found:
        return found

    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None


_tess_cmd = resolve_tesseract_cmd()
if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    print(f"Tesseract found at: {_tess_cmd}")
else:
    print(
        "WARNING: Tesseract binary not found on PATH or common install "
        "locations. Set the TESSERACT_CMD environment variable to its "
        "full path, or install Tesseract and ensure it's on PATH. "
        "OCR requests will return a clear error until this is fixed."
    )

print("OCR & Gauge Tool ready.")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Step 1: Is there even a circular dial in this image? ───────────────────
def detect_dial_bezel(gray):
    """
    Looks for a real circular gauge face using a Hough circle transform.
    Returns (cx, cy, r) for the most prominent circle found, or None if
    nothing convincingly circular is present. This is the gate that
    decides whether gauge analysis should run at all -- documents,
    invoices, and regular photos will correctly return None here.
    """
    h, w = gray.shape
    min_dim = min(h, w)
    blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim,          # only expect one dial per image
        param1=100,
        param2=60,                 # higher = stricter / fewer false positives
        minRadius=int(min_dim * 0.25),
        maxRadius=int(min_dim * 0.5),
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    cx, cy, r = max(circles, key=lambda c: c[2])  # largest circle wins
    return int(cx), int(cy), int(r)


# ── Step 2: Given a real dial, find the needle angle ────────────────────────
def detect_needle_angle(gray, cx, cy, max_radius):
    """
    Casts rays from the dial center and looks for the angle with the
    densest dark pixels (the needle). Unlike the original version, this
    requires the winning angle to clearly dominate the rest of the dial
    face -- a thin needle should light up a large fraction of its ray,
    far above the median "background" angle. This prevents random dark
    clusters (text, shadows, scratches) from being mistaken for a needle.
    """
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)

    h, w = gray.shape
    r_start = int(max_radius * 0.18)
    r_end = int(max_radius * 0.75)
    n_samples = r_end - r_start
    if n_samples <= 0:
        return None

    intensities = np.zeros(360, dtype=int)
    for degree in range(360):
        angle_rad = math.radians(degree)
        count = 0
        for r_step in range(r_start, r_end):
            sx = int(cx + r_step * math.cos(angle_rad))
            sy = int(cy + r_step * math.sin(angle_rad))
            if 0 <= sx < w and 0 <= sy < h and thresh[sy, sx] > 0:
                count += 1
        intensities[degree] = count

    best_angle = int(np.argmax(intensities))
    best_score = intensities[best_angle]
    background = float(np.median(intensities))

    # A real needle should cover a large share of its ray AND clearly
    # beat the typical (background) angle -- not just edge out noise.
    min_required = max(8, n_samples * 0.5)
    if best_score < min_required or best_score < background * 2:
        return None

    return float(best_angle)


# ── Calibrated Gauge Scale Mapping ─────────────────────────────────────────
def calculate_gauge_value(needle_angle, ocr_lines, cx, cy, radius):
    full_text_lower = " ".join(ocr_lines).lower() if ocr_lines else ""

    # Word-boundary matching -- a bare "c" anywhere in the text no longer
    # triggers a false unit match.
    unit = "psi"
    if re.search(r"\bbar\b", full_text_lower):
        unit = "bar"
    elif re.search(r"°c\b|\bdeg(?:rees)?\s*c\b|\bcelsius\b", full_text_lower):
        unit = "°C"

    # NOTE: these calibration anchors are specific to one physical gauge
    # (zero/max needle angles). If you support multiple gauge types,
    # this needs to come from per-device config, not be hardcoded here.
    if unit == "°C":
        max_val = 120.0
        zero_angle = 135.0
        max_angle = 45.0
    else:
        max_val = 100.0
        zero_angle = 135.0
        max_angle = 35.0

    total_arc = (max_angle - zero_angle) % 360
    relative_needle = (needle_angle - zero_angle) % 360
    calculated_value = (relative_needle / total_arc) * max_val
    calculated_value = max(0.0, min(max_val, calculated_value))

    return {
        "gauge_found": True,
        "detected_unit": unit,
        "detected_max_scale": max_val,
        "needle_angle_degrees": round(needle_angle, 1),
        "interpolated_reading": f"{round(calculated_value, 1)} {unit}",
        "center_x": cx,
        "center_y": cy,
        "radius": radius,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image field provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    job_id = str(uuid.uuid4())[:8]
    saved_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{job_id}_{secure_filename(file.filename)}"
    )
    file.save(saved_path)

    try:
        start_time = time.time()

        # ── Text OCR ──────────────────────────────────────────────────────
        ocr_error = None
        if _tess_cmd is None:
            raw_text = ""
            ocr_error = "Tesseract binary not found on the server (see startup warning)."
        else:
            try:
                pil_img = Image.open(saved_path)
                raw_text = pytesseract.image_to_string(pil_img)
            except Exception as e:
                raw_text = ""
                ocr_error = str(e)

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        words = [{"text": w, "confidence": 1.0} for w in raw_text.split()]

        # ── Gauge detection (gated by real circle detection) ───────────────
        img = cv2.imread(saved_path)
        gauge_results = {
            "gauge_found": False,
            "detected_unit": "Unknown",
            "detected_max_scale": 0.0,
            "needle_angle_degrees": 0.0,
            "interpolated_reading": "No circular gauge face detected in image.",
        }

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bezel = detect_dial_bezel(gray)
            if bezel is not None:
                cx, cy, r = bezel
                angle = detect_needle_angle(gray, cx, cy, r)
                if angle is not None:
                    gauge_results = calculate_gauge_value(angle, lines, cx, cy, r)
                else:
                    gauge_results = {
                        "gauge_found": False,
                        "detected_unit": "Unknown",
                        "detected_max_scale": 0.0,
                        "needle_angle_degrees": 0.0,
                        "interpolated_reading": "Dial face found but needle could not be isolated.",
                    }

        elapsed = round(time.time() - start_time, 3)

        response = {
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_file": file.filename,
            "processing_time_seconds": elapsed,
            "gpu_used": False,
            "ocr": {
                "full_text": raw_text if raw_text.strip() else "",
                "word_count": len(words),
                "line_count": len(lines),
                "words": words,
                "gauge_analysis": gauge_results,
            },
        }
        if ocr_error:
            response["ocr"]["ocr_error"] = ocr_error

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(saved_path):
            os.remove(saved_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)