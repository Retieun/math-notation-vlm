# Math-Notation-VLM

Fine-tuning PaliGemma-3B for handwritten mathematical notation to LaTeX transcription, using QLoRA for memory-efficient training on consumer hardware.

## Results

The model learns to transcribe handwritten math into LaTeX, handling Greek letters, accents, and subscripts.

**Example:**

| Handwritten Input | Model Output |
|---|---|
| ![math_0](https://github.com/Retieun/math-notation-vlm/raw/main/data/math_writing/images/math_0.png) | `V(\tilde{\phi}_{3})` |

Training loss converged from ~3.0 to 0.57 over 3 epochs with ~6.6M trainable parameters (0.22% of the full model).

## Approach

This project fine-tunes Google's [PaliGemma-3B](https://ai.google.dev/gemma/docs/paligemma) vision-language model using Parameter-Efficient Fine-Tuning (PEFT) via QLoRA. The goal was to build an end-to-end pipeline from data preparation through training to inference, gaining hands-on experience with multimodal model fine-tuning.

Key technical decisions:

- **4-bit quantization (QLoRA with NF4)** to fit a 3B-parameter model into 16GB VRAM. Used `paged_adamw_8bit` with gradient accumulation to avoid OOM errors.
- **Cross-platform migration**: Initial development targeted an AMD RX 6800 XT on Windows, but lack of `bitsandbytes` support for ROCm on Windows required migrating to an NVIDIA T4 (Google Colab) for CUDA-based quantization kernels.
- **Custom data pipeline**: Built a `collate_fn` for multimodal data ingestion with dynamic path resolution across local and cloud storage, and explicit `<image>` token injection for instruction tuning.

## Project Structure

```
.
├── data/
│   └── math_writing/
│       ├── images/            # Handwriting images
│       └── train.jsonl        # Image-text pairs
├── finetune_hub/              # Core training logic
│   ├── config.py              # Hyperparameters and model settings
│   ├── data.py                # DataLoader and collate logic
│   ├── engine.py              # Model loading and initialization
│   └── trainer.py             # HuggingFace Trainer wrapper
├── create_dataset.py          # Dataset preparation
├── train.py                   # Training entry point
└── inference.py               # Inference script for predictions
```

## Quick Start

### Installation

```bash
pip install torch transformers peft bitsandbytes accelerate pillow
```

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py --image_path "data/math_writing/images/test_sample.png"
```

## Limitations

This was a learning project focused on understanding VLM fine-tuning pipelines end-to-end, not a production system.

- Trained on a small custom dataset of handwritten math expressions
- Accuracy degrades on multi-line equations and deeply nested structures
- Evaluation was qualitative (visual inspection of outputs on held-out examples), not a formal benchmark
- The model works best on single-line expressions similar to the training distribution

## Acknowledgments

This project is built on the [math-vlm-finetune-pipeline](https://github.com/E1ims/math-vlm-finetune-pipeline) by E1ims. Key modifications include Windows/Linux path interoperability, 4-bit quantization integration for consumer GPUs, and standalone inference scripts.
