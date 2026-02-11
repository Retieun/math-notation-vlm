import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# 1. Setup paths
DATA_DIR = "data/math_writing"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

print("Downloading MathWriting dataset (Human subset)...")
# We'll start with a small sample (2000 items) to ensure everything works
ds = load_dataset("deepcopy/MathWriting-human", split="train[:2000]")


def process_and_save():
    manifest = []

    print("Processing images and creating manifest...")
    for i, example in enumerate(tqdm(ds)):
        image = example['image']
        latex_code = example['latex']

        # Save the image locally (the training script needs a path)
        img_filename = f"math_{i}.png"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        image.save(img_path)

        # Create the VLM prompt entry
        # Structure: prefix (instruction) and suffix (target LaTeX)
        entry = {
            "image": img_filename,
            "prefix": "transcribe this mathematical expression into LaTeX:",
            "suffix": latex_code
        }
        manifest.append(entry)

    # Save the JSONL manifest
    manifest_path = os.path.join(DATA_DIR, "train.jsonl")
    with open(manifest_path, 'w') as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    print(f"\nDone! Saved {len(manifest)} entries to {manifest_path}")


if __name__ == "__main__":
    process_and_save()