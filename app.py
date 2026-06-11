"""
OCR Tool - Flask API
Extracts text from images and returns structured JSON output.
Supports: PNG, JPEG, WEBP, BMP, TIFF, GIF
"""

import os
import json
import uuid
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import easyocr
import numpy as np

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialise EasyOCR reader (English + common languages)
print("Loading OCR model... (first run may take a minute)")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
print("OCR model ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_ocr(image_path: str) -> dict:
    """Run EasyOCR on an image and return structured results."""
    results = reader.readtext(image_path)

    words = []
    lines = []
    full_text_parts = []

    for bbox, text, confidence in results:
        # bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]

        words.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "bounding_box": {
                "x_min": int(min(x_coords)),
                "y_min": int(min(y_coords)),
                "x_max": int(max(x_coords)),
                "y_max": int(max(y_coords)),
            }
        })
        full_text_parts.append(text)

    # Group words into lines by y proximity (within 20px = same line)
    if words:
        sorted_words = sorted(words, key=lambda w: w["bounding_box"]["y_min"])
        current_line = [sorted_words[0]]
        for word in sorted_words[1:]:
            if abs(word["bounding_box"]["y_min"] - current_line[-1]["bounding_box"]["y_min"]) <= 20:
                current_line.append(word)
            else:
                lines.append(" ".join(w["text"] for w in current_line))
                current_line = [word]
        lines.append(" ".join(w["text"] for w in current_line))

    return {
        "full_text": "\n".join(lines),
        "lines": lines,
        "words": words,
        "word_count": len(words),
        "line_count": len(lines),
    }


def get_image_meta(image_path: str) -> dict:
    """Extract basic image metadata."""
    with Image.open(image_path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format or "unknown",
        }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the web UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    """
    Upload an image file via multipart/form-data (key: 'image').
    Returns a JSON file with the extracted text and metadata.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request. Use key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 415

    # Save upload
    original_name = secure_filename(file.filename)
    job_id = str(uuid.uuid4())[:8]
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_{original_name}")
    file.save(saved_path)

    try:
        start = time.time()
        ocr_data = run_ocr(saved_path)
        image_meta = get_image_meta(saved_path)
        elapsed = round(time.time() - start, 3)

        result = {
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_file": original_name,
            "processing_time_seconds": elapsed,
            "image": image_meta,
            "ocr": ocr_data,
        }

        # Save JSON output
        out_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        result["output_saved_to"] = out_path
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(saved_path):
            os.remove(saved_path)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
