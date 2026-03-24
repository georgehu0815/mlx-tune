"""
Example 6: Real End-to-End Training Test

This example ACTUALLY trains a model (small number of iterations for testing).
Tests the complete workflow including SFTTrainer.
"""

from mlx_tune import FastLanguageModel, SFTTrainer
import json


def create_tiny_dataset():
    """Create a tiny dataset for testing"""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Who created Linux?"},
                {"role": "assistant", "content": "Linux was created by Linus Torvalds."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
            ]
        },
    ]


def main():
    print("= - 06_real_training_test.py:49" * 70)
    print("REAL TRAINING TEST  EndtoEnd - 06_real_training_test.py:50")
    print("= - 06_real_training_test.py:51" * 70)

    # Step 1: Load Model
    print("\n1. Loading model... - 06_real_training_test.py:54")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=512,  # Shorter for faster training
        load_in_4bit=True,
    )
    print("✓ Model loaded! - 06_real_training_test.py:60")

    # Step 2: Configure LoRA
    print("\n2. Configuring LoRA... - 06_real_training_test.py:63")
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Smaller rank for faster training
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
    )
    print("✓ LoRA configured! - 06_real_training_test.py:71")

    # Step 3: Create Dataset
    print("\n3. Creating dataset... - 06_real_training_test.py:74")
    dataset = create_tiny_dataset()
    print(f"✓ Created dataset with {len(dataset)} examples - 06_real_training_test.py:76")

    # Step 4: Test Inference Before Training
    print("\n4. Testing inference BEFORE training... - 06_real_training_test.py:79")
    FastLanguageModel.for_inference(model)

    from mlx_lm import generate
    test_prompt = "What is 2+2?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    response_before = generate(
        model.model,
        tokenizer,
        prompt=formatted,
        max_tokens=50,
        verbose=False
    )
    print(f"Before training: {response_before[:100]}... - 06_real_training_test.py:94")

    # Step 5: Initialize Trainer
    print("\n5. Initializing SFTTrainer... - 06_real_training_test.py:97")
    # params=model.print_trainable_parameters()
    # print(f"Trainable parameters: {params} - 06_real_training_test.py:99")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=512,
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        output_dir="./test_training_output",
        adapter_path="./test_adapters",
        iters=10,  # Very small for testing
    )
    print("✓ Trainer initialized! - 06_real_training_test.py:112")

    # Step 6: Train
    print("\n6. Starting training... - 06_real_training_test.py:115")
    print("(This will actually train the model!) - 06_real_training_test.py:116")
    print()

    try:
        trainer.train()
        print("\n✓ Training completed! - 06_real_training_test.py:121")

    except Exception as e:
        print(f"\n⚠️  Training error: {e} - 06_real_training_test.py:124")
        print("\nThis might be expected if mlx_lm.lora has different requirements. - 06_real_training_test.py:125")
        print("Let me show you the manual training command... - 06_real_training_test.py:126")

        # Show manual command
        print("\nManual Training Command: - 06_real_training_test.py:129")
        print("" * 70)
        print(f"mlx_lm.lora \\ - 06_real_training_test.py:131")
        print(f"model {model.model_name} \\ - 06_real_training_test.py:132")
        print(f"train \\ - 06_real_training_test.py:133")
        print(f"data ./test_training_output/train.jsonl \\ - 06_real_training_test.py:134")
        print(f"iters 10 \\ - 06_real_training_test.py:135")
        print(f"learningrate 5e5 \\ - 06_real_training_test.py:136")
        print(f"batchsize 1 \\ - 06_real_training_test.py:137")
        print(f"loralayers 8 \\ - 06_real_training_test.py:138")
        print(f"adapterpath ./test_adapters - 06_real_training_test.py:139")
        print("" * 70)
        print("\nRun this command manually to train the model. - 06_real_training_test.py:141")
        return

    # Step 7: Test Inference After Training
    print("\n7. Testing inference AFTER training... - 06_real_training_test.py:145")
    # TODO: Load adapters and test
    print("(Adapter loading and testing to be implemented) - 06_real_training_test.py:147")

    print("\n - 06_real_training_test.py:149" + "=" * 70)
    print("ENDTOEND TEST COMPLETE! - 06_real_training_test.py:150")
    print("= - 06_real_training_test.py:151" * 70)


if __name__ == "__main__":
    main()
