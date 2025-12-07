import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

DATA_PATH = "data/cleaned/direct_corpus_labelled.csv" 
OUTPUT_DIR = "models/finetuned_categories"

MAX_LEN = 256
BATCH_SIZE = 64 

device = torch.device('cuda:0')

def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=1) 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

df = pd.read_csv(DATA_PATH)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
input_ids = []
attention_masks = []

for sent in df['text']:
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df['is_negative'].values, dtype=torch.long)

dataset = TensorDataset(input_ids, attention_masks, labels)

final_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
final_model.to(device)
final_model.eval()

eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SequentialSampler(dataset))
final_preds = []
final_labels = []

for step, batch in enumerate(eval_dataloader):
    if step % 50 == 0 and step != 0:
        print(f"  Batch {step} / {len(eval_dataloader)}")
        
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
        result = final_model(b_input_ids, attention_mask=b_input_mask)

    final_preds.append(result.logits.detach().cpu().numpy())
    final_labels.append(b_labels.to('cpu').numpy())

final_preds = np.concatenate(final_preds, axis=0)
final_labels = np.concatenate(final_labels, axis=0)
final_metrics = compute_metrics(final_preds, final_labels)

print("\n")
print("FINAL MODEL RESULTS (Full Data Fit)")
print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
print(f"F1 Score:  {final_metrics['f1']:.4f}")
print(f"Precision: {final_metrics['precision']:.4f}")
print(f"Recall:    {final_metrics['recall']:.4f}")
print("\n")