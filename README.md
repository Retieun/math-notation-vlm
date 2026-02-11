Math-Notation-VLM: Handwritten Math to $\LaTeX$ Transcription

An end-to-end multimodal AI pipeline built to transcribe handwritten mathematical expressions into clean, accurate LaTeX code. This project demonstrates the fine-tuning of Google's PaliGemma-3B Vision-Language Model (VLM) using Parameter-Efficient Fine-Tuning (PEFT) techniques.

🚀 The Result

The model was successfully trained to identify complex mathematical structures, including Greek letters, accents, and subscripts.

Handwritten Input: **

Model Output: V(\tilde{\phi}_{3})

Training Loss: Successfully converged from ~3.0 to 0.57 over 3 epochs.

🛠️ Engineering & Optimization Challenges

This project involved significant systems engineering to overcome hardware and software compatibility limitations.

1. Cross-Platform Hardware Migration (AMD to NVIDIA)

The initial development phase targeted a local AMD Radeon RX 6800 XT on Windows. Due to the lack of native bitsandbytes support for ROCm on Windows, I migrated the training stack to a Linux-based NVIDIA T4 (Google Colab) environment. This allowed the use of CUDA kernels required for 4-bit quantization.

2. Memory Efficiency (QLoRA)

To fit a 3B parameter model into a 16GB VRAM budget, I implemented 4-bit NormalFloat (NF4) quantization using QLoRA.

Trainable Parameters: ~6.6M (0.22% of total model).

Optimization: Used paged_adamw_8bit and gradient accumulation to maintain stability without OOM (Out of Memory) errors.

3. Data Pipeline Engineering

Designed a robust collate_fn to handle multimodal data ingestion:

Path Resolution: Implemented dynamic pathing to handle serialization differences between local and cloud storage.

Instruction Tuning: Explicitly managed <image> token injection to ensure deterministic adherence to transcription prompts.

4. Git Security & Secret Scrubbing

Identified and resolved credential leakage risks by scrubbing Hugging Face tokens from the Git history. Used history rewriting (git reset and --force pushes) to comply with GitHub Push Protection rules.

📂 Project Structure

finetune_hub/: Core engine containing model loading and training logic.

data/: Dataset manifest and image directory.

train.py: Main entry point for the training loop.

inference.py: Script for merging LoRA adapters and running transcriptions on new images.

🚦 Quick Start

1. Installation

pip install torch transformers peft bitsandbytes accelerate pillow


2. Running Inference

# Use the provided inference.py to test the model
python inference.py --image_path "data/math_writing/images/test_sample.png"


⚖️ Acknowledgments & License

This project is an optimized fork (see description for the original author).

Key Modifications:

Added Windows/Linux path interoperability.

Integrated 4-bit quantization for consumer-grade GPU support.

Developed standalone inference scripts for post-training validation.

