======================================================================
EXACT UNSLOTH PIPELINE - API Compatibility Demo
======================================================================

[Step 1] Loading model with 4-bit quantization...
✓ Model loaded: mlx-community/Llama-3.2-1B-Instruct-4bit

[Step 2] Applying LoRA adapters...
LoRA configuration set: rank=16, alpha=16, modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], dropout=0
✓ LoRA adapters configured

[Step 3] Preparing dataset...
✓ Dataset prepared with 4 examples

[Step 4] Configuring training...

[Step 5] Creating SFTTrainer...
Trainer initialized:
  Output dir: outputs
  Adapter path: outputs/adapters
  Learning rate: 0.0002
  Iterations: 20
  Batch size: 2
  LoRA r=16, alpha=16
  Native training: True
  LR scheduler: linear
  Grad checkpoint: False
✓ Trainer configured
  - Learning rate: 0.0002
  - Batch size: 2
  - Iterations: 20

[Step 6] Training the model...
(This will actually train using MLX under the hood)
======================================================================
Starting Fine-Tuning
======================================================================

[Using Native MLX Training]

Applying LoRA adapters...
Applying LoRA to 16 layers: {'rank': 16, 'scale': 1.0, 'dropout': 0, 'keys': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']}
✓ LoRA applied successfully to 16 layers
  Trainable LoRA parameters: 224
Preparing training data...
  Detected format: text
✓ Prepared 4 training samples
  Saved to: outputs/train.jsonl
✓ Created validation set (copied from train)

Training configuration:
  Iterations: 20
  Batch size: 2
  Learning rate: 0.0002
  LR scheduler: linear
  Grad checkpoint: True
  Adapter file: outputs/adapters/adapters.safetensors

Loaded 4 training samples, 4 validation samples
Starting training loop...
Starting training..., iters: 20
Iter 1: Val loss 5.165, Val took 0.440s
Iter 1: Train loss 5.046, Learning Rate 2.000e-04, It/sec 2.050, Tokens/sec 272.663, Trained Tokens 133, Peak mem 1.338 GB
Iter 2: Train loss 3.189, Learning Rate 1.900e-04, It/sec 3.652, Tokens/sec 467.477, Trained Tokens 261, Peak mem 1.449 GB
Iter 3: Train loss 1.591, Learning Rate 1.800e-04, It/sec 4.420, Tokens/sec 587.798, Trained Tokens 394, Peak mem 1.449 GB
Iter 4: Train loss 1.443, Learning Rate 1.700e-04, It/sec 4.382, Tokens/sec 560.918, Trained Tokens 522, Peak mem 1.449 GB
Iter 5: Train loss 0.656, Learning Rate 1.600e-04, It/sec 4.329, Tokens/sec 575.797, Trained Tokens 655, Peak mem 1.449 GB
Iter 6: Train loss 0.890, Learning Rate 1.500e-04, It/sec 4.284, Tokens/sec 548.338, Trained Tokens 783, Peak mem 1.449 GB
Iter 7: Train loss 0.457, Learning Rate 1.400e-04, It/sec 4.397, Tokens/sec 584.790, Trained Tokens 916, Peak mem 1.449 GB
Iter 8: Train loss 0.583, Learning Rate 1.300e-04, It/sec 4.302, Tokens/sec 550.715, Trained Tokens 1044, Peak mem 1.449 GB
Iter 9: Train loss 0.477, Learning Rate 1.200e-04, It/sec 4.291, Tokens/sec 570.706, Trained Tokens 1177, Peak mem 1.449 GB
Iter 10: Train loss 0.457, Learning Rate 1.100e-04, It/sec 4.264, Tokens/sec 545.782, Trained Tokens 1305, Peak mem 1.449 GB
Iter 11: Train loss 0.428, Learning Rate 1.000e-04, It/sec 4.450, Tokens/sec 569.646, Trained Tokens 1433, Peak mem 1.449 GB
Iter 12: Train loss 0.729, Learning Rate 9.000e-05, It/sec 4.440, Tokens/sec 590.469, Trained Tokens 1566, Peak mem 1.449 GB
Iter 13: Train loss 0.420, Learning Rate 8.000e-05, It/sec 4.406, Tokens/sec 563.923, Trained Tokens 1694, Peak mem 1.449 GB
Iter 14: Train loss 0.437, Learning Rate 7.000e-05, It/sec 4.426, Tokens/sec 588.683, Trained Tokens 1827, Peak mem 1.449 GB
Iter 15: Train loss 0.417, Learning Rate 6.000e-05, It/sec 4.248, Tokens/sec 543.698, Trained Tokens 1955, Peak mem 1.449 GB
Iter 16: Train loss 0.407, Learning Rate 5.000e-05, It/sec 4.351, Tokens/sec 578.700, Trained Tokens 2088, Peak mem 1.449 GB
Iter 17: Train loss 0.421, Learning Rate 4.000e-05, It/sec 4.325, Tokens/sec 553.649, Trained Tokens 2216, Peak mem 1.449 GB
Iter 18: Train loss 0.397, Learning Rate 3.000e-05, It/sec 4.351, Tokens/sec 578.642, Trained Tokens 2349, Peak mem 1.449 GB
Iter 19: Train loss 0.396, Learning Rate 2.000e-05, It/sec 4.415, Tokens/sec 587.131, Trained Tokens 2482, Peak mem 1.449 GB
Iter 20: Val loss 0.409, Val took 0.165s
Iter 20: Train loss 0.425, Learning Rate 1.000e-05, It/sec 4.461, Tokens/sec 571.014, Trained Tokens 2610, Peak mem 1.449 GB
Saved final weights to outputs/adapters/adapters.safetensors.
  Adapter config saved to: outputs/adapters/adapter_config.json

======================================================================
Training Complete!
======================================================================
  Adapters saved to: outputs/adapters

[Step 7] Enabling inference mode...
Inference mode enabled with KV caching

📝 Test Inference:
   Q: What is the capital of France?
   A: The capital of France is Paris.

[Step 8] Save options (Unsloth-compatible):
  model.save_pretrained('lora_model')  # Adapters only
  model.save_pretrained_merged('merged_16bit', tokenizer)
  model.save_pretrained_gguf('model', tokenizer, quantization_method='q4_k_m')

======================================================================
SUCCESS! The EXACT Unsloth pipeline works on Mac!
======================================================================

API COMPATIBILITY SUMMARY:
-------------------------
✅ FastLanguageModel.from_pretrained() - SAME API
✅ FastLanguageModel.get_peft_model() - SAME API
✅ SFTConfig - SAME API as TRL
✅ SFTTrainer - SAME API
✅ trainer.train() - SAME API
✅ FastLanguageModel.for_inference() - SAME API
✅ model.save_pretrained() - SAME API
✅ model.save_pretrained_merged() - SAME API
✅ model.save_pretrained_gguf() - SAME API

MIGRATION GUIDE:
---------------
1. Change import: from unsloth import -> from mlx_tune import
2. Change import: from trl import SFTTrainer, SFTConfig -> already in mlx_tune
3. Use mlx-community models instead of unsloth/ models
4. That's it! Rest of the code is IDENTICAL!

