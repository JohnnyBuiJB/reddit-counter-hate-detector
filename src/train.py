import os

import torch
import pandas as pd

import random

from datasets import concatenate_datasets

from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = "roberta-base"
SEED = 123
random.seed(SEED)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train_3_label_stance_base_null_tokens():
    print("------------------- Data Loading -------------------")
    gold_train_tokens = torch.load("gold_3-label_train_tokens.pt")
    silver_train_tokens = torch.load("silver_3-label_train_tokens.pt")
    
    gold_val_tokens = torch.load("gold_3-label_val_tokens.pt")
    silver_val_tokens = torch.load("silver_3-label_val_tokens.pt")
    
    mixed_train_tokens = concatenate_datasets([gold_train_tokens, silver_train_tokens]).shuffle(seed=SEED)
    mixed_val_tokens = concatenate_datasets([gold_val_tokens, silver_val_tokens]).shuffle(seed=SEED)
    
    gold_train_tokens.shuffle(seed=SEED)
    silver_train_tokens.shuffle(seed=SEED)
    
    gold_train_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    gold_val_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    mixed_train_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    mixed_val_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    
        
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=3)
    batch_size = 4
    model_name = MODEL_CKPT + "-finetune-3-class" + "-null-token"
    
    logging_steps = len(mixed_train_tokens) // batch_size
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=2,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_total_limit = 2)

    trainer = Trainer(model=model, args=training_args, 
                    compute_metrics=compute_metrics,
                    train_dataset=mixed_train_tokens,
                    eval_dataset=mixed_val_tokens)
    
    print("------------------- Training with mixed data -------------------")
    trainer.train()
    
    logging_steps = len(gold_train_tokens) // batch_size
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=7,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_total_limit = 2)
    
    trainer = Trainer(model=model, args=training_args, 
                    compute_metrics=compute_metrics,
                    train_dataset=gold_train_tokens,
                    eval_dataset=gold_val_tokens)
    print("------------------- Training with gold data -------------------")
    trainer.train()
    
def train_3_label_stance_base():
    print("------------------- Data Loading -------------------")
    gold_train_tokens = torch.load("gold_3-label_train_tokens.pt")
    silver_train_tokens = torch.load("silver_3-label_train_tokens.pt")
    
    gold_val_tokens = torch.load("gold_3-label_val_tokens.pt")
    silver_val_tokens = torch.load("silver_3-label_val_tokens.pt")
    
    mixed_train_tokens = concatenate_datasets([gold_train_tokens, silver_train_tokens]).shuffle(seed=SEED)
    mixed_val_tokens = concatenate_datasets([gold_val_tokens, silver_val_tokens]).shuffle(seed=SEED)
    
    gold_train_tokens.shuffle(seed=SEED)
    silver_train_tokens.shuffle(seed=SEED)
    
    gold_train_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    gold_val_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    mixed_train_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    mixed_val_tokens.set_format("torch", columns=["input_ids", "attention_mask", 
                                            "label"])
    
        
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=3)
    batch_size = 4
    model_name = MODEL_CKPT + "-finetune-3-class"
    
    logging_steps = len(mixed_train_tokens) // batch_size
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=2,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_total_limit = 2)

    trainer = Trainer(model=model, args=training_args, 
                    compute_metrics=compute_metrics,
                    train_dataset=mixed_train_tokens,
                    eval_dataset=mixed_val_tokens)
    
    print("------------------- Training with mixed data -------------------")
    trainer.train()
    
    logging_steps = len(gold_train_tokens) // batch_size
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=7,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=False, 
                                    log_level="error",
                                    save_total_limit = 2)
    
    trainer = Trainer(model=model, args=training_args, 
                    compute_metrics=compute_metrics,
                    train_dataset=gold_train_tokens,
                    eval_dataset=gold_val_tokens)
    print("------------------- Training with gold data -------------------")
    trainer.train()
    
def main():
    train_3_label_stance_base_null_tokens()

if __name__ == "__main__":
    main()
    