from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

from transformers import RobertaTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt

from shift_correction import analyze_val_data, update_probs

MODEL_CKPT = "self-train"

def compute_metrics_3_class(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average=None)
    acc = balanced_accuracy_score(labels, preds)
    return {"accuracy": acc, "f1-0": f1[0], "f1-1": f1[1], "f1-2" : f1[2]}

def compute_metrics_2_class(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="binary", pos_label=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix(y_preds, y_true, title):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title(title)
    plt.savefig(title)

def evaluate_2_class_model(data_fn):
    print("------------------- LOADING DATA -------------------")
    test_tokens = torch.load(data_fn, map_location="cpu")
    
    print("------------------- LOADING MODEL -------------------")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=2)
    model.eval()

    batch_size = 8
    logging_steps = len(test_tokens) // batch_size
    
    model_name = MODEL_CKPT
    
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=8,
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
                    compute_metrics=compute_metrics_2_class,
                    eval_dataset=test_tokens)
    
    print("------------------- EVALUATING -------------------")
    t_preds_list = trainer.predict(test_tokens) # predict
    
    t_logits = torch.Tensor(t_preds_list.predictions)
    t_probs = torch.nn.functional.softmax(t_logits, dim=1)
    t_preds = np.argmax(t_probs, 1)
    t_true = t_preds_list.label_ids
    
    torch.save(t_preds_list, "reddit_preds.pt")
    torch.save(t_true, "reddit_trues.pt")
    
    print(t_preds_list)
    print('Test CK:', cohen_kappa_score(t_true, t_preds))
    print('Test ROC:', roc_auc_score(t_true, t_probs[:, 1]))
    
def evaluate_3_class_model(data_fn):
    print("------------------- LOADING DATA -------------------")
    test_tokens = torch.load(data_fn, map_location="cpu")
    
    print("------------------- LOADING MODEL -------------------")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, 
                                                            num_labels=2)
    model.eval()

    batch_size = 8
    logging_steps = len(test_tokens) // batch_size
    model_name = MODEL_CKPT
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=8,
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
                    compute_metrics=compute_metrics_3_class,
                    eval_dataset=test_tokens)
    
    print("------------------- EVALUATING -------------------")
    t_preds_list = trainer.predict(test_tokens) # predict
    
    t_logits = torch.Tensor(t_preds_list.predictions)
    t_probs = torch.nn.functional.softmax(t_logits, dim=1)
    t_preds = np.argmax(t_probs, 1)
    t_true = t_preds_list.label_ids
    
    f1 = f1_score(t_true, t_preds)
    print('Test F1 = [%.02f, %.02f, %.02f]' % (f1[0], f1[1], f1[2]))
    print('Test CK:', cohen_kappa_score(t_true, t_preds))
    print('Test ROC:', roc_auc_score(t_true, t_probs, average='macro', multi_class='ovo'))

def main():
    evaluate_2_class_model("reddit_2-label_test_tokens.pt")
    

if __name__ == "__main__":
    main()
    