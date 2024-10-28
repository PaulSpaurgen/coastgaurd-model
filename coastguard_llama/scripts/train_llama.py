from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

def prepare_training_data(json_path):
    # Load the dataset
    dataset = load_dataset('json', data_files=json_path)
    return dataset['train']

def train_model():
    # Use TinyLlama instead of Meta's Llama
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to reduce memory usage
    )
    
    # Split dataset into train and evaluation sets
    print("Preparing dataset...")
    dataset = prepare_training_data("../data/training_data.json")
    dataset = dataset.train_test_split(test_size=0.1)  # 10% for evaluation
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    print("Tokenizing dataset...")
    tokenized_train_dataset = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    tokenized_eval_dataset = dataset['test'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['test'].column_names
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="../models/coastguard_tinyllama",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced batch size for lower memory usage
        gradient_accumulation_steps=4,   # Accumulate gradients to simulate larger batch size
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-5,
        logging_steps=10,
        fp16=True,                       # Use mixed precision training
        optim="adamw_torch",
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,  # Add evaluation dataset
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("../models/coastguard_tinyllama")
    print("Training complete!")

if __name__ == "__main__":
    train_model()
