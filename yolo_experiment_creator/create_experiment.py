#!/usr/bin/env python
import os
import argparse
import random
import yaml
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO experiment template")
    parser.add_argument("name", nargs="?", default=None, help="Experiment name")
    parser.add_argument("--dataset", help="Path to YOLO dataset", default=None)
    return parser.parse_args()

def create_experiment(args):
    # Use current date-time and append to name if provided
    name = args.name or f"Exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    exp_dir = Path(name)
    exp_dir.mkdir(exist_ok=True)

    # Files
    train_file = exp_dir / "train.txt"
    val_file = exp_dir / "val.txt"
    test_file = exp_dir / "test.txt"
    yaml_file = exp_dir / "data.yaml"
    train_py = exp_dir / "train.py"
    predict_py = exp_dir / "predict.py"
    log_file = exp_dir / "train_logs.txt"
    status_file = exp_dir / "status.txt"

    # Create train.txt, val.txt, and test.txt based on original dataset
    if args.dataset and Path(args.dataset).exists():
        dataset_path = Path(args.dataset)
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        
        # Get the list of image files (assuming they are .jpg or .png)
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = [labels_dir / (img.stem + ".txt") for img in image_files]

        # Ensure that each image has a corresponding label
        valid_image_label_pairs = [
            (img, lbl) for img, lbl in zip(image_files, label_files) if lbl.exists()
        ]

        # Split into train, val, and test sets (80%, 10%, 10%)
        random.shuffle(valid_image_label_pairs)
        total = len(valid_image_label_pairs)
        train_size = int(total * 0.8)
        val_size = int(total * 0.1)
        
        train_split = valid_image_label_pairs[:train_size]
        val_split = valid_image_label_pairs[train_size:train_size + val_size]
        test_split = valid_image_label_pairs[train_size + val_size:]

        # Write train.txt, val.txt, and test.txt
        write_txt(train_file, [str(img.resolve()) for img, lbl in train_split])
        write_txt(val_file, [str(img.resolve()) for img, lbl in val_split])
        write_txt(test_file, [str(img.resolve()) for img, lbl in test_split])

        # Write corresponding label files
        write_txt(train_file.with_name("train_labels.txt"), [str(lbl.resolve()) for img, lbl in train_split])
        write_txt(val_file.with_name("val_labels.txt"), [str(lbl.resolve()) for img, lbl in val_split])
        write_txt(test_file.with_name("test_labels.txt"), [str(lbl.resolve()) for img, lbl in test_split])

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
        test_file.write_text("")
        names = []

    # Create new data.yaml
    with open(yaml_file, "w") as f:
        yaml.dump({
            "train": str(train_file.resolve()),
            "val": str(val_file.resolve()),
            "test": str(test_file.resolve()),
            "train_labels": str(train_file.with_name("train_labels.txt").resolve()),
            "val_labels": str(val_file.with_name("val_labels.txt").resolve()),
            "test_labels": str(test_file.with_name("test_labels.txt").resolve()),
            "nc": len(names),
            "names": names
        }, f, default_flow_style=False)

    # Create train.py (hardcoded parameters)
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
        f.write(f"Test File: {test_file.resolve()}\n")
        f.write(f"Data.yaml: {yaml_file.resolve()}\n")
        f.write(f"Class Names: {names if names else 'No classes defined'}\n")

    print(f"✅ Experiment '{name}' created successfully at {exp_dir.resolve()}")

def write_txt(file_path, lines):
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def main():
    args = parse_args()  # Parse the arguments
    create_experiment(args)  # Pass the parsed args to the function

if __name__ == "__main__":
    main()  # Run the main function