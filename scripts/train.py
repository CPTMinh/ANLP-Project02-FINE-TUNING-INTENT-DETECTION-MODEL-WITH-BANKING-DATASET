from unsloth import FastLanguageModel               # For loading the pre-trained language model
import os               # For file and directory operations
import yaml             # For reading configuration files
import torch            # For PyTorch operations
import pandas as pd     # For data manipulation
from datasets import Dataset                        # For handling datasets
from transformers import TrainingArguments          # For setting up training configurations
from trl import SFTTrainer                          # For supervised fine-tuning

def load_config(config_path = "configs/train.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def format_data(example):
    # LLaMA-3 prompt format for intent classification
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nClassify the intent of this text into a category (0-76):\nText: {example['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nIntent ID: {example['label']}<|eot_id|>"
    return {"text": prompt}

def main():
    print("1. Loading configurations")
    config = load_config()

    print("2. Loading dataset")
    df_train = pd.read_csv("sample_data/train.csv")

    # Convert pandas DataFrame to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(df_train)

    # Format the dataset for Causual LM training (next-token prediction)
    train_dataset = train_dataset.map(format_data, remove_columns=["label"])

    print("3. Loading Unsloth LLaMA-3 model")
    # Initialize the Fast LanguageModel from Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model_name"],
        max_seq_length = config["max_seq_length"],
        dtype = None,                   # Auto-detect (Float16/Bfloat16 depending on Colab GPU)
        load_in_4bit = True,            # Using 4-bit quantization to fit in Colab T4/L4
    )

    # Add LoRA adapters (PEFT) for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora_r"],
        target_modules = config["target_modules"],
        lora_alpha = config["lora_alpha"],
        lora_dropout = config["lora_dropout"],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
    )

    print("4. Setting up Trainer")
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config["max_seq_length"],
        dataset_num_proc = 2,
        packing = False, # Set to True for faster training if texts are very short
        args = TrainingArguments(
            per_device_train_batch_size = config["batch_size"],
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = config["epochs"],
            learning_rate = float(config["learning_rate"]),
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = config["optimizer"],
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = config["output_dir"],

            # Checkpointing ARGS
            save_strategy = "steps",
            save_steps = 50,
            save_total_limit = 2,
            load_best_model_at_end = False,
        ),
    )

    print("5. Starting training")
    trainer_stats = trainer.train(resume_from_checkpoint=True) 

    print("6. Saving the LoRA adapter")
    model.save_pretrained(config["save_model_path"])
    tokenizer.save_pretrained(config["save_model_path"])

    print("Training complete")

if __name__ == "__main__":
    main()