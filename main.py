"""
Main Project Entry Point: Brain Tumor Detection and Classification Using MRI
Paper: An Automated Deep Learning Framework for Brain Tumor Detection and Classification Using MRI
Authors: Dr. D.Ganesh, Mr. P.Venkateswarlu Reddy, Neeruganti Vidya Sree,
         Thotla Chakra Ganesh, Pudi Nikhil Surya Teja, Galam Dhanunjaneya
Affiliation: Department of CSE, Mohan Babu University, Tirupati
"""

import argparse
import os
import sys

# Ensure local cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Detection and Classification CAD Framework (Hybrid YOLO-NAS + EfficientNet)"
    )
    parser.add_argument(
        "--web", action="store_true", help="Launch the Interactive Web CAD Workstation"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Web server port (default: 5000)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Web server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run full pipeline demonstration and benchmarks"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train detector and classifier networks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.web:
        from app.app import app
        print(f"\n=======================================================")
        print(f"  Starting Brain Tumor CAD Web Workstation...")
        print(f"  Local URL: http://{args.host}:{args.port}")
        print(f"=======================================================\n")
        app.run(host=args.host, port=args.port, debug=False)

    elif args.train:
        from src.dataset import build_sample_dataset, BrainMRIDataset
        from src.trainer import PipelineTrainer
        from src.classifier_efficientnet import BrainTumorEfficientNet
        from src.detector_yolo_nas import YoloNasTumorDetector
        from torch.utils.data import DataLoader

        print("\n=== Initializing Training for Hybrid CAD Models ===")
        meta = build_sample_dataset("dataset/samples", n_per_class=15)
        paths = [m["path"] for m in meta]
        labels = [m["label"] for m in meta]
        bboxes = [m["bbox"] if m["bbox"] else [0.0, 0.0, 0.0, 0.0] for m in meta]

        dataset = BrainMRIDataset(paths, labels, bboxes)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        trainer = PipelineTrainer(save_dir="models_saved", device="cpu")

        # Train Classifier
        classifier = BrainTumorEfficientNet(variant="b0", num_classes=4, pretrained=False)
        trainer.train_classifier(classifier, loader, epochs=args.epochs, model_name="efficientnet_b0")

        # Train Detector
        detector = YoloNasTumorDetector()
        trainer.train_detector(detector, loader, epochs=args.epochs, model_name="yolo_nas")

        print("\n=== All Models Successfully Trained & Saved to models_saved/ ===")

    else:
        # Default action: run the complete end-to-end demo and generate figures
        from demo import run_pipeline_demo
        run_pipeline_demo()


if __name__ == "__main__":
    main()
