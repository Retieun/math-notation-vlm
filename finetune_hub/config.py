from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class ModelConfig:
    # --- Data Settings ---
    dataset_id: str = "json"
    # Ensure this points to your local file
    data_files: str = "data/math_writing/train.jsonl"
    prompt_text: str = "Transcribe this image into LaTeX text."
    # --- Model Settings ---
    model_id: str = "google/paligemma-3b-pt-224"

    # --- Optimization (The Fix for your Error) ---
    # The code expects 'use_4bit', not 'load_in_4bit'
    use_4bit: bool = True

    # These prevent the NEXT error you might see
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False

    # --- LoRA Settings ---
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Standard targets for PaliGemma/Llama architectures
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # --- Training Settings ---
    output_dir: str = "checkpoints"
    per_device_train_batch_size: int = 2
    batch_size: int = 2
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 10
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-4
    max_steps: int = 500
    logging_steps: int = 10
    save_steps: int = 100

    # --- System ---
    max_seq_len: int = 512  # Critical for long LaTeX proofs