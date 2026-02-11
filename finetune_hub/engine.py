import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    AutoProcessor
)
from peft import get_peft_model, LoraConfig, TaskType


class VLMEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def load_model(self):
        print(f"Loading Base Model: {self.cfg.model_id}...")

        # 1. Define Quantization Config (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.cfg.use_4bit,
            bnb_4bit_quant_type=self.cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.cfg.use_nested_quant,
        )

        # 2. Load the Base Model (Frozen)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 3. Load the Processor
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)

        # 4. Apply LoRA Adapters (The Trainable Layers)
        print("Applying LoRA Adapters (The Trainable Layers)...")
        peft_config = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.target_modules,
            task_type="CAUSAL_LM",  # PaliGemma treats VLM tasks as Causal LM
            bias="none"
        )

        # This wraps the frozen 4-bit model with trainable LoRA layers
        model = get_peft_model(model, peft_config)

        # Verify trainable parameters (Sanity Check)
        model.print_trainable_parameters()

        self.model = model
        print("Model loaded successfully.")

        # CRITICAL: Return the model so train.py can use it
        return self.model