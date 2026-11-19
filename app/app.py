"""
Flask Web Application & Diagnostic CAD System
Hybrid YOLO-NAS + EfficientNet Brain Tumor Detection & Classification
Paper: An Automated Deep Learning Framework for Brain Tumor Detection and Classification Using MRI
Authors: Dr. D.Ganesh, Mr. P.Venkateswarlu Reddy, Neeruganti Vidya Sree,
         Thotla Chakra Ganesh, Pudi Nikhil Surya Teja, Galam Dhanunjaneya
Affiliation: Department of CSE, Mohan Babu University, Tirupati, Andhra Pradesh, India
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "matplotlib")
import io
import base64
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from src.pipeline import HybridBrainTumorPipeline
from src.metrics import plot_confusion_matrix_fig6, plot_comparative_results_fig7
from src.dataset import build_sample_dataset, SyntheticMRIGenerator

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Generate baseline figures if they don't exist
fig6_path = os.path.join("outputs", "figure_6_confusion_matrix.png")
if not os.path.exists(fig6_path):
    plot_confusion_matrix_fig6(fig6_path)

fig7_path = os.path.join("outputs", "figure_7_comparative_graph.png")
if not os.path.exists(fig7_path):
    plot_comparative_results_fig7(fig7_path)

# Initialize pipeline
pipeline = HybridBrainTumorPipeline(device="cpu", efficientnet_variant="b0")

# Preload sample MRIs if none exist
sample_dir = os.path.join("dataset", "samples")
if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
    build_sample_dataset(output_dir=sample_dir, n_per_class=10)


def encode_image_base64(img_bgr):
    """Encodes BGR numpy image to base64 jpeg data URL."""
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    # Find preloaded sample images
    samples = []
    classes = ["Glioma", "Meningioma", "Pituitary", "Non-Tumor"]
    for c in classes:
        cdir = os.path.join(sample_dir, c)
        if os.path.exists(cdir):
            files = [f for f in os.listdir(cdir) if f.endswith((".png", ".jpg"))]
            if files:
                rel_path = f"dataset/samples/{c}/{files[0]}"
                samples.append({"name": f"{c} Sample", "class": c, "path": rel_path})

    return render_template("index.html", samples=samples)


@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        image_data = None
        # Check if file uploaded
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            image_data = cv2.imread(save_path)
        # Check if sample selected
        elif "sample_path" in request.form and request.form["sample_path"]:
            sample_rel = request.form["sample_path"]
            if os.path.exists(sample_rel):
                image_data = cv2.imread(sample_rel)

        if image_data is None:
            return jsonify({"success": False, "error": "No image provided or failed to load image."}), 400

        # Execute Pipeline
        res = pipeline.process_image(image_data)

        # Convert images to base64 for frontend display
        orig_b64 = encode_image_base64(cv2.cvtColor(res["original_image"], cv2.COLOR_RGB2BGR))
        prep_b64 = encode_image_base64(cv2.cvtColor(res["preprocessed_image"], cv2.COLOR_RGB2BGR))
        annot_b64 = encode_image_base64(cv2.cvtColor(res["annotated_image"], cv2.COLOR_RGB2BGR))
        roi_b64 = encode_image_base64(cv2.cvtColor(res["roi_image"], cv2.COLOR_RGB2BGR))

        # Save diagnostic summary
        summary_path = os.path.join("outputs", "latest_diagnosis.png")
        pipeline.generate_diagnostic_summary_figure(res, summary_path)

        return jsonify({
            "success": True,
            "detected": res["detected"],
            "bounding_box": res["bounding_box"],
            "detection_confidence": round(res["detection_confidence"] * 100, 2),
            "predicted_class": res["predicted_class"],
            "class_confidence": round(res["class_confidence"] * 100, 2),
            "all_probabilities": {k: round(v * 100, 2) for k, v in res["all_probabilities"].items()},
            "binary_label": res["binary_label"],
            "binary_confidence": round(res["binary_confidence"] * 100, 2),
            "images": {
                "original": orig_b64,
                "preprocessed": prep_b64,
                "annotated": annot_b64,
                "roi": roi_b64,
            },
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/generate_sample", methods=["POST"])
def generate_sample_api():
    """Generates an on-demand synthetic brain MRI scan."""
    tumor_type = request.json.get("type", "Glioma")
    view = request.json.get("view", "axial")

    img_rgb, bbox, label_idx = SyntheticMRIGenerator.generate_sample(
        tumor_type=tumor_type, size=256, view=view
    )
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    filename = f"generated_{tumor_type.lower()}_{view}.png"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    cv2.imwrite(filepath, bgr)

    return jsonify({
        "success": True,
        "sample_path": filepath,
        "type": tumor_type,
        "view": view,
        "image": encode_image_base64(bgr),
    })


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(os.path.abspath("outputs"), filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
