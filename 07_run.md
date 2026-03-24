======================================================================
MLX-Tune: Exact Same Pipeline as Unsloth
======================================================================

1. Loading model (SAME API as Unsloth)...
✓ Model loaded!

2. Applying LoRA adapters (SAME API)...
LoRA configuration set: rank=16, alpha=16, modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], dropout=0
✓ LoRA configured!

3. Preparing dataset (SAME as Unsloth)...
✓ Dataset prepared with 3 examples

4. Training with SFTTrainer (SAME API)...
   Note: Uses MLX under the hood (not TRL), but API is compatible!
Trainer initialized:
  Output dir: unsloth_comparison_output
  Adapter path: unsloth_comparison_output/unsloth_comparison_adapters
  Learning rate: 0.0002
  Iterations: 10
  Batch size: 1
  LoRA r=16, alpha=16
  Native training: True
  LR scheduler: cosine
  Grad checkpoint: False

Starting training...
======================================================================
Starting Fine-Tuning
======================================================================

[Using Native MLX Training]

Applying LoRA adapters...
Applying LoRA to 16 layers: {'rank': 16, 'scale': 1.0, 'dropout': 0, 'keys': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']}
✓ LoRA applied successfully to 16 layers
  Trainable LoRA parameters: 224
Preparing training data...
  Detected format: chatml
✓ Prepared 3 training samples
  Saved to: unsloth_comparison_output/train.jsonl
✓ Created validation set (copied from train)

Training configuration:
  Iterations: 10
  Batch size: 1
  Learning rate: 0.0002
  LR scheduler: cosine
  Grad checkpoint: True
  Adapter file: unsloth_comparison_output/unsloth_comparison_adapters/adapters.safetensors

Loaded 3 training samples, 3 validation samples
Starting training loop...
Starting training..., iters: 10
Iter 1: Val loss 5.759, Val took 0.275s
Iter 10: Val loss 0.354, Val took 0.099s
Iter 10: Train loss 1.607, Learning Rate 4.894e-06, It/sec 5.733, Tokens/sec 290.655, Trained Tokens 507, Peak mem 1.063 GB
Saved final weights to unsloth_comparison_output/unsloth_comparison_adapters/adapters.safetensors.
  Adapter config saved to: unsloth_comparison_output/unsloth_comparison_adapters/adapter_config.json

======================================================================
Training Complete!
======================================================================
  Adapters saved to: unsloth_comparison_output/unsloth_comparison_adapters
✓ Training complete!

5. Inference (SAME API as Unsloth)...
Inference mode enabled with KV caching
  Q: What is Python?
  A: Python is a high-level programming language....

6. Save Options (NOW SAME AS UNSLOTH!)...

   a) Save LoRA adapters only:
      model.save_pretrained('lora_model')  # ✅ SAME API!

   b) Save merged model (base + adapters):
      model.save_pretrained_merged('merged_16bit', tokenizer)  # ✅ SAME API!

   c) Save as GGUF for llama.cpp/Ollama:
      model.save_pretrained_gguf('model', tokenizer, quantization_method='q4_k_m')  # ✅ SAME API!

======================================================================
COMPARISON SUMMARY
======================================================================

✅ SAME as Unsloth:
  - FastLanguageModel.from_pretrained()
  - FastLanguageModel.get_peft_model()
  - SFTTrainer(...)
  - trainer.train()
  - FastLanguageModel.for_inference()
  - model.save_pretrained()
  - model.save_pretrained_merged()
  - model.save_pretrained_gguf()
  - load_dataset() from HuggingFace
  - tokenizer.apply_chat_template()

⚠️  DIFFERENT (but compatible):
  - Backend: MLX instead of CUDA/Triton
  - Trainer: MLX-based instead of TRL-based
  - Platform: Apple Silicon instead of NVIDIA

💡 ADVANTAGE:
  - Develop locally on Mac
  - Deploy to CUDA just by changing import!
  - Code is 99% identical

======================================================================
Just like Unsloth, but for Mac! 🚀
======================================================================
