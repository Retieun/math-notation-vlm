import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the cloned repository folder
# Check if the folder "math-vlm-finetune-pipeline" exists, otherwise try the current dir
repo_path = os.path.join(current_dir, "math-vlm-finetune-pipeline")

if os.path.exists(repo_path):
    print(f"Adding repository to Python path: {repo_path}")
    sys.path.append(repo_path)
else:
    print(f"Warning: Could not find 'math-vlm-finetune-pipeline' in {current_dir}")
    print("Assuming code is flattened in the current directory.")
# --- CRITICAL FIX END ---

# Now these imports will work
from finetune_hub.config import ModelConfig
from finetune_hub.engine import VLMEngine
from finetune_hub.data import DataProcessor
from finetune_hub.trainer import TrainerWrapper


def main():
    print("Initializing Configuration...")
    cfg = ModelConfig()

    # SENIOR TWEAK: Ensure the data path is absolute to avoid confusion
    # This fixes issues if the code runs from a different directory
    if not os.path.isabs(cfg.data_files):
        cfg.data_files = os.path.join(current_dir, cfg.data_files)

    print(f"Loading Model: {cfg.model_id} with 4-bit Quantization...")
    engine = VLMEngine(cfg)
    model = engine.load_model()

    print("Preparing Data Pipeline...")
    processor = engine.processor
    data_proc = DataProcessor(processor, cfg)

    train_dataset = data_proc.load_data()

    print("Starting Training Run...")
    trainer = TrainerWrapper(model, processor, train_dataset, cfg, data_proc.collate_fn)
    trainer.train()

    # Create models directory if it doesn't exist
    save_dir = os.path.join(current_dir, "models", "math_vlm_adapter")
    os.makedirs(save_dir, exist_ok=True)

    print("Training Complete! Saving adapter...")
    trainer.save_model(save_dir)


if __name__ == "__main__":
    main()