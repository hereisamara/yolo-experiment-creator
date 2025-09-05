#!/usr/bin/env python
import os
import argparse
import random
import yaml
from pathlib import Path
from datetime import datetime

def get_image_paths(images_dir, labels_dir):
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.JPG"))
    return image_files

def get_label_paths(labels_dir):
    return [str(lbl.resolve()) for lbl in labels_dir.glob("*.txt")]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO experiment template")
    parser.add_argument("name", nargs="?", default=None, help="Experiment name")
    parser.add_argument("--dataset", help="Path to YOLO dataset", default=None)
    return parser.parse_args()

def create_experiment(args):
    # Use current date-time and append to name if provided
    name = f"Exp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if not args.name else f"Exp_{args.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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

    # Check if dataset is already split into train, valid, and test folders
    if args.dataset and Path(args.dataset).exists():
        dataset_path = Path(args.dataset)
        if all((dataset_path / folder).exists() for folder in ['train', 'valid']):

            print("train, valid, test folders found in dataset. Using existing splits.")
            # Dataset is already split into train, valid, and test
            image_paths = {
                'train': dataset_path / 'train' / 'images',
                'valid': dataset_path / 'valid' / 'images',
                'test': dataset_path / 'test' / 'images'
            }
            label_paths = {
                'train': dataset_path / 'train' / 'labels',
                'valid': dataset_path / 'valid' / 'labels',
                'test': dataset_path / 'test' / 'labels'
            }

            print(image_paths, label_paths)

            # Create the txt files by fetching corresponding image and label pairs
            write_txt(train_file, get_image_paths(image_paths['train'], label_paths['train']))
            write_txt(val_file, get_image_paths(image_paths['valid'], label_paths['valid']))
            write_txt(test_file, get_image_paths(image_paths['test'], label_paths['test']))

            # Create corresponding label files
            write_txt(train_file.with_name("train_labels.txt"), get_label_paths(label_paths['train']))
            write_txt(val_file.with_name("val_labels.txt"), get_label_paths(label_paths['valid']))
            write_txt(test_file.with_name("test_labels.txt"), get_label_paths(label_paths['test']))

            # Read class names from dataset.yaml if it exists
            names = []
            dataset_yaml = dataset_path / "data.yaml"
            if dataset_yaml.exists():
                with open(dataset_yaml, "r") as f:
                    data_cfg = yaml.safe_load(f)
                    names = data_cfg.get("names", [])

        else:
            dataset_path = Path(args.dataset)
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"
            
            # Get the list of image files (assuming they are .jpg or .png)
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.JPG"))
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
        print("No dataset path provided or path does not exist. Creating empty experiment structure.")
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

    # ============================
    # Initialize YOLO model
    # ============================
    model = YOLO("yolov8n.pt")  # default base model

    # ============================
    # Experiment / Dataset Setup
    # ============================
    DATA_PATH = "{yaml_file.resolve()}"
    PROJECT_DIR = "{exp_dir}/runs"
    EXPERIMENT_NAME = "train"
    DEVICE = 0
    WORKERS = 8

    # ============================
    # Training Hyperparameters
    # ============================
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640
    LEARNING_RATE = 0.005
    LR_FINAL = 0.001
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    OPTIMIZER = "SGD"
    DROPOUT = 0.1
    WARMUP_EPOCHS = 5
    NBS = 64
    PRETRAINED = True
    FREEZE = None
    DETERMINISTIC = True
    SINGLE_CLS = False

    # ============================
    # Augmentation Parameters
    # ============================
    DEGREES = 0.0
    TRANSLATE = 0.05
    SHEAR = 0.0
    FLIP_LR = 0.5
    FLIP_UD = 0.0
    HSV_H = 0.015
    HSV_S = 0.2
    HSV_V = 0.04
    MOSAIC = 0.0
    COPY_PASTE = 0.0
    MIXUP = 0.0
    CLOSE_MOSAIC = 15
    AUTO_AUGMENT = "randaugment"
    ERASING = 0.4
    SCALE = 0.5
    MASK_RATIO = 4
    DYN = False

    # ============================
    # NMS / Detection Settings
    # ============================
    CONF = None
    IOU = 0.7
    MAX_DET = 300
    AGNOSTIC_NMS = False

    # ============================
    # Logging & Saving
    # ============================
    SAVE = True
    SAVE_TXT = False
    SAVE_CONF = False
    LOG_FILE = "{log_file}"

    # ============================
    # Tracker / Vision Settings
    # ============================
    TRACKER = "botsort.yaml"
    SHOW = False
    PLOTS = True
    VERBOSE = True

    # ============================
    # Start Training
    # ============================
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        lrf=LR_FINAL,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        warmup_epochs=WARMUP_EPOCHS,
        fliplr=FLIP_LR,
        flipud=FLIP_UD,
        degrees=DEGREES,
        translate=TRANSLATE,
        shear=SHEAR,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        mosaic=MOSAIC,
        copy_paste=COPY_PASTE,
        mixup=MIXUP,
        close_mosaic=CLOSE_MOSAIC,
        cache=True,
        pretrained=PRETRAINED,
        auto_augment=AUTO_AUGMENT,
        erasing=ERASING,
        scale=SCALE,
        mask_ratio=MASK_RATIO,
        plots=PLOTS,
        verbose=VERBOSE,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        agnostic_nms=AGNOSTIC_NMS,
        single_cls=SINGLE_CLS,
        tracker=TRACKER,
        save=SAVE
    )

    # Save logs
    with open(LOG_FILE, "w") as log:
        log.write(str(results))

    print(f"✅ Training complete! Logs saved at {{LOG_FILE}}")
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
        if len(lines) == 0:
            f.write("")
        else:
            f.write('\n'.join(str(line) for line in lines) + '\n')

def main():
    args = parse_args()  # Parse the arguments
    create_experiment(args)  # Pass the parsed args to the function

if __name__ == "__main__":
    main()  # Run the main function