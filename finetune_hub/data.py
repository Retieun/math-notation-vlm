import torch
from datasets import load_dataset
from finetune_hub.config import ModelConfig
from PIL import Image
import os
class DataProcessor:
    """
    Handles data pipeline operations: loading raw datasets and processing batches 
    for the Vision-Language Model.
    
    This class acts as the bridge between the raw Hugging Face dataset and the 
    VLM's expected input format (tensors).
    """
    def __init__(self, processor, cfg):
        """
        Args:
            processor: The Hugging Face AutoProcessor (handles tokenization & image resizing).
            cfg (ModelConfig): Global configuration object containing dataset IDs and prompt text.
        """
        self.processor = processor
        self.cfg = cfg

    def load_data(self, split="train", limit=None):
        """
        Loads the dataset from the Hugging Face Hub based on the config.

        Args:
            split (str): Which split to load (e.g., "train", "test", "validation").
            limit (int, optional): If set, only loads the first N examples. 
                                   Useful for quick debugging or sanity checks.
        """
        print(f"Loading dataset: {self.cfg.dataset_id}...")
        if self.cfg.dataset_id == "json":
            ds = load_dataset(
                "json",
                data_files=self.cfg.data_files,
                split=split
            )
        else:
            # Fallback for standard Hub datasets
            ds = load_dataset(self.cfg.dataset_id, split=split)
        # --- FIX END ---
        
        # Slicing the dataset for rapid prototyping if a limit is specified
        if limit:
            ds = ds.select(range(limit))
        return ds

    def collate_fn(self, examples):
        """
        The critical batch processing function.
        """
        # 1. Get Prompts
        # We repeat the instruction for every image in the batch
        texts = [self.cfg.prompt_text for _ in examples]

        # 2. Load and Process Images (THE FIX)
        images = []
        for ex in examples:
            img_data = ex["image"]

            # FIX: If it's a file path (string), load it from disk
            if isinstance(img_data, str):
                # Try to find the image in the expected folders
                possible_paths = [
                    img_data,  # Absolute path or relative to current dir
                    os.path.join("data", "math_writing", "images", img_data),  # Your project structure
                    os.path.join("data", "images", img_data)
                ]

                found = False
                for p in possible_paths:
                    if os.path.exists(p):
                        img_data = Image.open(p)
                        found = True
                        break

                if not found:
                    raise FileNotFoundError(f"Could not find image: {img_data}")

            # Now that we are sure it's an image, convert to RGB
            images.append(img_data.convert("RGB"))

        # 3. Get Labels (FIX: Key Mismatch)
        # Your creation script used 'suffix', but your data.py looked for 'latex'.
        # This line checks for 'suffix' first, preventing a KeyError.
        labels = [ex.get("suffix", ex.get("latex", "")) for ex in examples]

        # 4. Process Batch
        inputs = self.processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            # FIX: Variable Name Mismatch
            # Your config uses 'max_seq_len', but data.py used 'max_seq_length'
            max_length=self.cfg.max_seq_len
        )

        return inputs