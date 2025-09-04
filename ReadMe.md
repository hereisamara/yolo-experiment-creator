# **Simple YOLO Experiment Version Control**

A lightweight tool to manage YOLO model versions and datasets without duplicating data. It manipulates filenames via `filenames.txt`, adding prefixes/suffixes to handle multiple datasets. Parameters are logged automatically in `train_logs.txt`, and experiments are named with timestamps for easy versioning and rollback.

- **No dataset duplication**: Works by referencing filenames, not entire datasets.
- **Version control**: Track YOLO models by adding prefixes/suffixes to filenames.
- **Automatic logging**: Parameters are saved in `train_logs.txt` for each experiment.
- **Date/time-based naming**: Automatically names experiments based on the current timestamp.
- **Simplified dataset handling**: Merge datasets easily by manipulating filenames in `filenames.txt`.

## **How It Works**

This tool creates a **new experiment** for every training session, organizing the training logs and dataset references. It can combine multiple dataset types by adjusting the filenames, ensuring you never need to re-copy datasets.

You can control the following:

- Add prefixes or postfixes to filenames to keep track of different dataset versions.
- The model training parameters are automatically logged into `train_logs.txt` for future reference.

### Example:

1. **Experiment Name**: Each experiment is given a name based on the current **date/time**.
2. **Data.yaml**: References filenames and paths, not the entire dataset, ensuring your dataset files are always clean and easy to manipulate.

---

## **Usage**

### **1. Create an Experiment**

Simply run the script and specify the experiment name (optional) and dataset path:

```bash
create-experiment --dataset /path/to/dataset

```

If no name is provided, a timestamp-based name is generated automatically. The experiment will generate the following files:

- **train.txt**: Contains paths to training images.
- **val.txt**: Contains paths to validation images.
- **data.yaml**: A configuration file that references `train.txt` and `val.txt`.
- **train.py**: The training script for YOLO.
- **predict.py**: The prediction script.
- **train_logs.txt**: Log of the training process with details like epochs, batch size, and more.
- **status.txt**: Metadata about the experiment (name, dataset path, class names).

### **2. Manipulate Filenames**

You can easily combine datasets by referencing files in **filenames.txt**. By adding prefixes or suffixes to the filenames, you can experiment with different dataset combinations without duplicating the data.

For example, combine datasets by adding a suffix to filenames:

- **train.txt**: Contains paths to images from multiple datasets combined with different prefixes.

### **3. Track Experiment Parameters**

Each experiment automatically saves training logs in **train_logs.txt**, where all parameters (epochs, batch size, image size) are logged, ensuring you always have a reference to the parameters used for any model version.

### **4. Experiment with Date and Time**

Experiments are named based on the current **date and time** (e.g., `Exp_bird-detection_2025-09-04_17-21-19`). This allows you to track and look back at older experiments easily.

---

## **Example Workflow**

1. **Create Experiment**:

```bash
create-experiment --dataset /path/to/yolo-dataset

```

This will generate the following structure:

```
Exp_2025-09-04_17-21-19/
│
├── train.txt
├── val.txt
├── data.yaml
├── train.py
├── predict.py
├── train_logs.txt
└── status.txt

```

1. **Training the Model**: After creating the experiment, the model will be trained using the parameters defined in **train.py**.
2. **Saving Logs**: The training logs (including the model parameters) are saved in `train_logs.txt` for future reference.
3. **Track and Reproduce**: Later, if you need to reproduce an experiment, you can simply look up the logs and data paths, making it easy to see exactly how the model was trained.


This tool makes it easier to manage multiple experiments and model versions while avoiding the overhead of copying entire datasets. With automatic logging, filename manipulation, and timestamp-based experiment names, you can focus on training models and experimenting without worrying about losing track of parameters or datasets.

---

## **Installation**

To install the package, run:

```bash
pip install .
```

This will install the `create-experiment` command-line tool, allowing you to easily generate and manage experiments.
