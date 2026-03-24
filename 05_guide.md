======================================================================
MLX-Tune Example: Complete Fine-Tuning Workflow
======================================================================

======================================================================
STEP 1: Load Model from HuggingFace
======================================================================

You can load ANY model from HuggingFace:
  - meta-llama/Llama-3.2-1B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3
  - Qwen/Qwen2.5-7B-Instruct
  - Or pre-quantized: mlx-community/Llama-3.2-1B-Instruct-4bit

For this example, we'll use a pre-quantized model (faster):
✓ Model loaded successfully!
  Model: mlx-community/Llama-3.2-1B-Instruct-4bit
  Max sequence length: 2048

======================================================================
STEP 2: Load Dataset from HuggingFace
======================================================================

Loading a real dataset from HuggingFace Hub...
You can use:
  - prepare_dataset('yahma/alpaca-cleaned')
  - prepare_dataset('mlabonne/FineTome-100k')
  - prepare_dataset(dataset_path='local/data.jsonl')
  - Any dataset compatible with HuggingFace datasets!
✓ Dataset loaded (3 sample examples)
  In production, you'd load thousands of examples

======================================================================
STEP 3: Prepare Training Data
======================================================================
✓ Training data saved to: train_data.jsonl
  Format: JSONL with chat messages

Demonstrating chat template formatting:
  Original: What is Python?
  Formatted length: 343 chars
  Template applied: ✓ (supports Llama, Mistral, Qwen, etc.)

======================================================================
STEP 4: Configure LoRA for Parameter-Efficient Fine-Tuning
======================================================================
LoRA configuration set: rank=16, alpha=32, modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], dropout=0.05
✓ LoRA configured!
  Rank: 16
  Alpha: 32
  Target modules: 7

======================================================================
STEP 5: Train the Model
======================================================================

For actual training, run MLX-LM command:
----------------------------------------------------------------------
mlx_lm.lora \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit \
    --train \
    --data train_data.jsonl \
    --iters 100 \
    --learning-rate 2e-4 \
    --lora-layers 16 \
    --batch-size 4 \
    --adapter-path ./adapters
----------------------------------------------------------------------

✓ Training configuration ready:
  Epochs: 3
  Learning rate: 0.0002
  Batch size: 4

Note: Training happens via mlx_lm.lora command or custom training loop
      After training, adapters are saved in ./adapters/ directory

======================================================================
STEP 6: Save Fine-Tuned Model in Standard HuggingFace Format
======================================================================

After training, save your model so ANYONE can use it:
  - Not MLX-specific format
  - Standard HuggingFace transformers format
  - Can be loaded with transformers, vLLM, etc.
  - Can be shared on HuggingFace Hub

Example save commands:
----------------------------------------------------------------------
# Save locally
save_model_hf_format(model, tokenizer, './my-finetuned-model')

# Save and upload to HuggingFace Hub
save_model_hf_format(
    model, tokenizer,
    './my-finetuned-model',
    push_to_hub=True,
    repo_id='username/my-awesome-model'
)
----------------------------------------------------------------------

✓ Model can be saved in standard HF format
  Others can use: transformers.AutoModel.from_pretrained('your-model')

======================================================================
STEP 7: Export to GGUF for llama.cpp, Ollama, etc. (Optional)
======================================================================

GGUF export enables:
  - Use with llama.cpp
  - Use with Ollama
  - Use with GPT4All
  - CPU-optimized inference

Example export command:
----------------------------------------------------------------------
export_to_gguf(
    './my-finetuned-model',
    output_path='model-q4.gguf',
    quantization='q4_k_m'  # or 'q5_k_m', 'q8_0', etc.
)
----------------------------------------------------------------------

✓ GGUF export supported for maximum compatibility

======================================================================
WORKFLOW COMPLETE!
======================================================================

What You Can Do:
  ✓ Load ANY HuggingFace model (not just mlx-community)
  ✓ Use load_dataset() from HuggingFace datasets
  ✓ Apply chat templates for different LLMs
  ✓ Fine-tune with LoRA/QLoRA
  ✓ Save in standard HuggingFace format
  ✓ Export to GGUF for deployment
  ✓ Share on HuggingFace Hub
  ✓ Use in Jupyter notebooks

Just like Unsloth, but for Apple Silicon!

======================================================================
Next Steps:
======================================================================
1. Load your favorite HuggingFace model
2. Load a real dataset (thousands of examples)
3. Train with: mlx_lm.lora --model ... --train --data ...
4. Save in HF format and/or export to GGUF
5. Share your fine-tuned model with the world!
