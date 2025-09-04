import os
import random
from pathlib import Path
import argparse

# Utility function to read text file and return list of lines
def read_txt(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Utility function to write list of lines to a text file
def write_txt(file_path, lines):
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

# 1. Randomly copy x% of dataset and create new train/test/val splits
def copy_subset_of_dataset(dataset_txt, percent=20, output_dir="new_dataset"):
    dataset_lines = read_txt(dataset_txt)  # Read the dataset (image paths)
    
    subset_size = int(len(dataset_lines) * (percent / 100))  # Determine subset size
    subset = random.sample(dataset_lines, subset_size)  # Randomly sample the subset

    # Make output directory for the subset dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Define new text files for the subset dataset
    new_train_txt = Path(output_dir) / "train.txt"
    new_val_txt = Path(output_dir) / "val.txt"
    new_test_txt = Path(output_dir) / "test.txt"
    
    # Split the subset dataset into train, validation, and test (80%, 10%, 10%)
    random.shuffle(subset)
    train_size = int(len(subset) * 0.8)
    val_size = int(len(subset) * 0.1)

    train_split = subset[:train_size]
    val_split = subset[train_size:train_size + val_size]
    test_split = subset[train_size + val_size:]

    # Write the image paths to the new train, val, and test files
    write_txt(new_train_txt, train_split)
    write_txt(new_val_txt, val_split)
    write_txt(new_test_txt, test_split)

    # Create corresponding label files
    train_labels = [str(Path(line).with_suffix(".txt").resolve()) for line in train_split]
    val_labels = [str(Path(line).with_suffix(".txt").resolve()) for line in val_split]
    test_labels = [str(Path(line).with_suffix(".txt").resolve()) for line in test_split]

    # Write the label file paths to new train_labels.txt, val_labels.txt, and test_labels.txt
    write_txt(Path(output_dir) / "train_labels.txt", train_labels)
    write_txt(Path(output_dir) / "val_labels.txt", val_labels)
    write_txt(Path(output_dir) / "test_labels.txt", test_labels)

    print(f"Subset dataset created at {output_dir} with {percent}% of original dataset.")

# 2. Delete files that match a given pattern (e.g., *frame* or _%6d)
# Function to delete pattern entries from text files without deleting actual data
def delete_files_by_pattern(root_dir, pattern):
    # Define the pattern to look for in the filenames
    root_path = Path(root_dir)
    
    # Find all relevant txt files
    txt_files = list(root_path.glob("**/*.txt"))
    
    # Filter out image and label files specifically (train, val, test, etc.)
    image_txt_files = [txt for txt in txt_files if 'train' in txt.stem or 'val' in txt.stem or 'test' in txt.stem]
    label_txt_files = [txt for txt in txt_files if 'train_labels' in txt.stem or 'val_labels' in txt.stem or 'test_labels' in txt.stem]

    # Process the image and label txt files
    for txt_file in image_txt_files + label_txt_files:
        lines = read_txt(txt_file)
        new_lines = []

        for line in lines:
            # Keep the line if it doesn't match the pattern
            if pattern not in line:
                new_lines.append(line)
            else:
                print(f"Removed {line} from {txt_file} (pattern match: {pattern})")

        # Write the cleaned lines back to the txt file
        write_txt(txt_file, new_lines)
        print(f"Updated {txt_file} by removing entries with pattern '{pattern}'.")

# # 3. Create train/test split based on new list of images and labels
# def create_train_test_split(image_label_list, train_txt, val_txt, split_ratio=0.8):
#     random.shuffle(image_label_list)
#     split_index = int(len(image_label_list) * split_ratio)

#     train_list = image_label_list[:split_index]
#     val_list = image_label_list[split_index:]

#     write_txt(train_txt, train_list)
#     write_txt(val_txt, val_list)

#     print(f"Train/test split created with {split_ratio * 100}% for training.")

# 4. Show dataset info: count images, labels, objects, etc.
# Function to show dataset info: count images, labels, objects, etc.
def show_dataset_info(root_dir):
    # Define the pattern of files to look for
    image_txt_files = [Path(root_dir) / "train.txt", Path(root_dir) / "val.txt", Path(root_dir) / "test.txt"]
    label_txt_files = [Path(root_dir) / "train_labels.txt", Path(root_dir) / "val_labels.txt", Path(root_dir) / "test_labels.txt"]

    # Check if essential files exist
    missing_files = []

    if not Path(root_dir).exists():
        print(f"Error: The directory {root_dir} does not exist.")
        return

    # Check for essential text files
    if not any(image_txt_files):
        missing_files.append("train.txt (required)")

    if not any(label_txt_files):
        missing_files.append("train_labels.txt (required)")

    if missing_files:
        print(f"Missing essential files: {', '.join(missing_files)}")
        print("Please create or rename these files according to the expected format.")
        print("The required files are:")
        print(" - train.txt (required)")
        print(" - val.txt (optional)")
        print(" - test.txt (optional)")
        print(" - train_labels.txt (required)")
        print(" - val_labels.txt (optional)")
        print(" - test_labels.txt (optional)")
        return

    images = set()  # To track unique image filenames (without extensions)
    labels = set()  # To track unique label filenames (with .txt extensions)
    total_objects = 0  # Total count of objects in labels

    # Process the image text files (train.txt, val.txt, test.txt)
    for txt_file in image_txt_files:
        if txt_file.exists():
            lines = read_txt(txt_file)
            for line in lines:
                image_file = Path(line).stem  # Get image file name without extension
                label_file = Path(line).with_suffix(".txt")  # Corresponding label file name

                # Check the corresponding label file path in label_txt_files
                # If the label file exists, add to the set
                if label_file.exists():
                    labels.add(label_file)

                images.add(image_file)  # Add the image file to images set

    # Count objects (bounding boxes) in label files
    for label in labels:
        with open(label, "r") as label_file:
            total_objects += len(label_file.readlines())  # Count lines (objects) in label file

    # Display dataset statistics for each split (train, val, or test)
    for txt_file in image_txt_files:
        if txt_file.exists():
            print(f"Info for {txt_file.stem} split:")
            print(f"  - Contains {len(images)} images.")
            print(f"  - Contains {len(labels)} label files.")
            print(f"  - Total objects (bounding boxes) in the dataset: {total_objects}\n")

# 5. Check if labels are correct: Check random samples
def check_labels(txt_file, sample_size=5):
    lines = read_txt(txt_file)
    sample_lines = random.sample(lines, sample_size)

    for line in sample_lines:
        image_file = Path(line).stem
        label_file = Path(line).with_suffix(".txt")

        if label_file.exists():
            print(f"Checking {label_file} for {image_file}")
            with open(label_file, "r") as label:
                print(label.readlines())

# 6. Clean dataset: Delete entries from text files if labels are empty or missing
def clean_dataset(txt_file):
    lines = read_txt(txt_file)
    cleaned_lines = []

    for line in lines:
        image_file = Path(line).stem
        label_file = Path(line).with_suffix(".txt")

        if label_file.exists():
            # Do not delete any files, just remove from txt file if label is empty
            if os.stat(label_file).st_size == 0:
                print(f"Empty label detected, removing {line} from {txt_file}")
            else:
                cleaned_lines.append(line)
        else:
            print(f"Missing label for image: {image_file}, removing {line} from {txt_file}")

    write_txt(txt_file, cleaned_lines)
    print(f"Cleaned {txt_file}, removed mismatched or empty labels.")

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Manipulation for YOLO")
    
    # Subcommands for different operations
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand for copying a subset
    copy_parser = subparsers.add_parser("copy-subset", help="Copy a subset of the dataset")
    copy_parser.add_argument("dataset_txt", help="Path to the dataset text file (train.txt, etc.)")
    copy_parser.add_argument("--percent", type=int, default=20, help="Percentage of dataset to copy")
    copy_parser.add_argument("--output-dir", default="new_dataset", help="Directory to save the subset")

    # Subcommand for deleting files by pattern (from txt files)
    delete_parser = subparsers.add_parser("delete-pattern", help="Delete entries matching a pattern from text files")
    delete_parser.add_argument("root_dir", help="Path to the root directory of the dataset")
    delete_parser.add_argument("pattern", help="Pattern to match and delete")

    # Subcommand for showing dataset info
    info_parser = subparsers.add_parser("info", help="Show dataset information")
    info_parser.add_argument("root_dir", help="Path to the root directory of the dataset")

    # Subcommand for checking labels
    check_parser = subparsers.add_parser("check-labels", help="Check random labels for correctness")
    check_parser.add_argument("txt_file", help="Path to the text file (train.txt, etc.)")
    check_parser.add_argument("--sample-size", type=int, default=5, help="Number of samples to check")

    # Subcommand for cleaning the dataset
    clean_parser = subparsers.add_parser("clean", help="Clean the dataset (remove empty/mismatched labels)")
    clean_parser.add_argument("txt_file", help="Path to the text file (train.txt, etc.)")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "copy-subset":
        copy_subset_of_dataset(args.dataset_txt, percent=args.percent, output_dir=args.output_dir)
    elif args.command == "delete-pattern":
        delete_files_by_pattern(args.txt_file, args.pattern)
    elif args.command == "info":
        show_dataset_info(args.txt_file)
    elif args.command == "check-labels":
        check_labels(args.txt_file, sample_size=args.sample_size)
    elif args.command == "clean":
        clean_dataset(args.txt_file)

if __name__ == "__main__":
    main()
