# Fine_Tune_BLOOM_for_Financial_QA

A Colab-based notebook to fine-tune a BLOOM causal language model for financial Question Answering (QA). This repository contains an interactive Jupyter/Colab notebook that walks through preparing a financial QA dataset, configuring the tokenizer and model, applying parameter-efficient fine-tuning (e.g., LoRA/PEFT), running training, and saving the fine-tuned weights for inference.

## Repository contents
- `Fine_Tune_BLOOM_for_Financial_QA.ipynb` — Main Colab/Jupyter notebook with step-by-step code and instructions.
- `README.md` — This file.

## Project goals
- Fine-tune BLOOM for domain-specific (financial) QA tasks.
- Use parameter-efficient tuning (LoRA/PEFT) to reduce GPU/memory requirements.
- Provide an easy-to-run Colab notebook for replication and experimentation.
- Save the final model in a format usable for inference (HF-style or safetensors).

## Key features
- Notebook tested on Colab (T4 GPU) — includes dependency install and model download steps.
- Dataset preprocessing for QA pairs (question + answer).
- Tokenization and data collation suitable for causal language modeling and QA instruction formats.
- Training loop using Hugging Face Transformers + PEFT / LoRA (or equivalent, depending on notebook code).
- Example inference cell to load and use the fine-tuned model.

## Requirements
- Python 3.8+ (Colab has compatible environment)
- Recommended libraries (installed in the notebook):
  - transformers
  - datasets
  - accelerate
  - peft
  - bitsandbytes (if using 8-bit optimizations)
  - safetensors (optional)
  - torch (compatible with your CUDA / Colab GPU)
  - sentencepiece / tokenizers (as required by tokenizer)

(Install commands are included in the notebook. Example:)
pip install -U "transformers>=4.30" datasets accelerate peft bitsandbytes safetensors

## Usage

### 1) Run the notebook (Colab)
1. Open `Fine_Tune_BLOOM_for_Financial_QA.ipynb` in Google Colab (notebook contains a Colab badge).
2. Run the cells sequentially:
   - Install dependencies
   - Set `MODEL_NAME` (base BLOOM model, e.g., `bigscience/bloom-1b1` or another size suited to your GPU)
   - Mount Google Drive if you want to save checkpoints
   - Prepare and load dataset
   - Apply PEFT/LoRA configuration
   - Run training and save the resulting model files

Tip: Reduce batch size or use gradient accumulation on limited-GPU environments.

### 2) Notebook-level variables to set
- MODEL_NAME: base BLOOM model to fine-tune (example: `bigscience/bloom-1b1`).
- DATA_PATH or dataset variable: path to your QA dataset (json/jsonl/csv) expected format described below.
- OUTPUT_DIR: directory to save model checkpoints and final weights.
- LoRA / PEFT hyperparameters: rank, target modules, dropout — editable in the notebook.

### 3) Example dataset format
The notebook assumes a QA dataset with question-answer examples. Two common formats:
- JSONL where each line is:
  {"question": "What is EPS?", "answer": "EPS stands for Earnings Per Share ..."}
- Or a list JSON:
  [
    {"question": "...", "answer": "..."},
    ...
  ]

During preprocessing the notebook converts QA pairs into instruction / response text for causal LM training. Adjust the prompt template if needed for your evaluation metric.

### 4) Training & hyperparameters (example)
- epochs: 3
- per_device_train_batch_size: 1–4 (depends on model size & GPU)
- learning_rate: 1e-4 – 5e-5 (tune as needed)
- weight_decay: 0.0–0.01
- gradient_accumulation_steps: as needed to emulate larger batch sizes
- LoRA rank (r): 8–32 (tradeoff between performance and params)
- Save checkpoints to `OUTPUT_DIR`

(Exact hyperparameters are set in the notebook — use them as a starting point and tune for your data.)

## Inference example (after training)
Example Python snippet to load the fine-tuned model (adjust if using PEFT/LoRA):

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", load_in_8bit=True)
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)  # or use from_pretrained with adapter

prompt = "Question: What is operating margin?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Adjust device_map and quantization flags depending on your hardware.

## Evaluation
- Use held-out QA pairs to compute automated metrics (exact match, F1) or manual review for factual correctness.
- Always test on out-of-sample financial questions and monitor hallucinations. Consider adding constraints or retrieval augmentation for factual grounding.

## Tips & caveats
- Larger BLOOM variants require more memory — choose a model that matches your GPU resources.
- Financial domain data can be sensitive; ensure data privacy and licensing compliance.
- If hallucinations are problematic, consider retrieval-augmented generation (RAG) or fine-tuning with stronger grounding signals.

## Extending the project
- Add an evaluation script that computes EM/F1 over a test set.
- Add a `train.py` script with argparse so training can be run outside a notebook via `accelerate launch`.
- Integrate retrieval (FAISS / Chroma) for grounded answers.
- Add a small demo app for interactive QA (Flask/Streamlit/Gradio).

## License & citation
- Add an appropriate license for your code and dataset. If using Hugging Face models, follow their license and terms.
- Cite the BLOOM paper and any datasets you use.

## Contact
For questions or help reproducing results, open an issue or contact the repository owner.
