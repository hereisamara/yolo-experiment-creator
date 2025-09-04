#!/usr/bin/env python

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def parse_args():
    # Argument parsing function
    parser = argparse.ArgumentParser(description="Generate YOLO experiment template")

    # Experiment name (optional)
    parser.add_argument(
        "name", 
        nargs="?", 
        default=None, 
        help="The name of the experiment. If not provided, a timestamp-based name will be used."
    )
    
    # Dataset path (optional)
    parser.add_argument(
        "--dataset", 
        help="Path to YOLO dataset", 
        default=None
    )

    return parser.parse_args()

def create_experiment(args):
    # Use current date-time and append to name if provided
    name = f"Exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if not args.name else f"Exp_{args.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    exp_dir = Path(name)
    exp_dir.mkdir(exist_ok=True)

    # Files
    train_file = exp_dir / "train.txt"
    val_file = exp_dir / "val.txt"
    yaml_file = exp_dir / "data.yaml"
    train_py = exp_dir / "train.py"
    predict_py = exp_dir / "predict.py"
    log_file = exp_dir / "train_logs.txt"
    status_file = exp_dir / "status.txt"

    # Create train.txt and val.txt
    if args.dataset and Path(args.dataset).exists():
        dataset_path = Path(args.dataset)
        images_dir = dataset_path / "images"
        train_images = list((images_dir / "train").glob("*.jpg"))
        val_images = list((images_dir / "val").glob("*.jpg"))

        with open(train_file, "w") as f:
            f.write("\n".join(str(p.resolve()) for p in train_images))
        with open(val_file, "w") as f:
            f.write("\n".join(str(p.resolve()) for p in val_images))

        # Read class names from dataset.yaml if exists
        names = []
        dataset_yaml = dataset_path / "data.yaml"
        if dataset_yaml.exists():
            with open(dataset_yaml, "r") as f:
                data_cfg = yaml.safe_load(f)
                names = data_cfg.get("names", [])
    else:
        train_file.write_text("")
        val_file.write_text("")
        names = []

    # Create new data.yaml
    with open(yaml_file, "w") as f:
        yaml.dump({
            "train": str(train_file.resolve()),
            "val": str(val_file.resolve()),
            "nc": len(names),
            "names": names
        }, f, default_flow_style=False)

    # Create train.py (hardcoding epochs, batch, imgsz)
    train_script = f"""from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # default model

results = model.train(
    data="{yaml_file.resolve()}",
    epochs=100,          # Hardcoded epochs
    imgsz=640,           # Hardcoded image size
    batch=16,            # Hardcoded batch size
    device=0,
    workers=8,
    project="{exp_dir}/runs",
    name="train",
    exist_ok=True,
)

# Save logs
with open("{log_file}", "w") as log:
    log.write(str(results))
"""
    train_py.write_text(train_script)

    # Create predict.py
    predict_script = f"""from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
    source="test.jpg",  # change to your test image/video
    conf=0.001,
    iou=0.45,
    imgsz=640,           # Hardcoded image size
    save=True,
    project="{exp_dir}/runs",
    name="predict",
    exist_ok=True,
)

with open("{exp_dir}/predict_log.txt", "w") as log:
    log.write(str(results))
"""
    predict_py.write_text(predict_script)

    # Create status.txt file
    with open(status_file, "w") as f:
        f.write(f"Experiment Name: {name}\n")
        f.write(f"Date and Time: {datetime.now()}\n")
        f.write(f"Dataset Path: {args.dataset if args.dataset else 'No dataset provided'}\n")
        f.write(f"Train File: {train_file.resolve()}\n")
        f.write(f"Validation File: {val_file.resolve()}\n")
        f.write(f"Data.yaml: {yaml_file.resolve()}\n")
        f.write(f"Class Names: {names if names else 'No classes defined'}\n")

    print(f"✅ Experiment '{name}' created successfully at {exp_dir.resolve()}")

def main():
    args = parse_args()  # Parse the arguments
    create_experiment(args)  # Pass the parsed args to the function

if __name__ == "__main__":
    main()  # Run the main function