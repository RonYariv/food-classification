import os
import shutil
import random
from tqdm import tqdm
from src import config

# ---------------------------
# CONFIGURATION
# ---------------------------
SOURCE_DIR = config.FOOD101_RAW_DIR  # Original folder with all class subfolders
DEST_DIR = config.FOOD101_SPLIT_DIR   # Destination folder for train/val/test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42  # for reproducibility

# ---------------------------
# SETUP
# ---------------------------
random.seed(SEED)
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1"

splits = ["train", "val", "test"]

# Create destination folders
for split in splits:
    os.makedirs(os.path.join(DEST_DIR, split), exist_ok=True)

# ---------------------------
# PROCESS EACH CLASS
# ---------------------------
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
print(f"Found {len(classes)} classes: {classes[:5]}...")

for cls in tqdm(classes, desc="Processing classes"):
    cls_src = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val  # Remaining goes to test

    split_counts = {"train": n_train, "val": n_val, "test": n_test}
    start = 0

    for split in splits:
        count = split_counts[split]
        split_images = images[start:start+count]
        start += count

        dest_class_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(dest_class_dir, exist_ok=True)

        for img in split_images:
            src_path = os.path.join(cls_src, img)
            dst_path = os.path.join(dest_class_dir, img)
            shutil.copy2(src_path, dst_path)  # copy2 preserves metadata

print("Dataset split completed!")