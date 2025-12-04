import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import os
import numpy as np

MODEL_DIR = "models/finetuned_categories"
INPUT_CSV = "data/processed/X_2024_combined.csv"
OUTPUT_CSV = "data/cleaned/the_dataset.csv"
BATCH_SIZE = 32
MAX_LEN = 128
NUM_LABELS = 4

device = torch.device('cuda:0')
    
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=NUM_LABELS)
model.to(device)
model.eval()

df = pd.read_csv(INPUT_CSV)
texts = df["content"].astype(str).tolist()

input_ids = []
attention_masks = []

for sent in texts:
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

dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=BATCH_SIZE)

predictions = []

for batch in dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    
    with torch.no_grad():
        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = result.logits
    pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
    predictions.extend(pred_ids)

df['predicted_category'] = [p + 1 for p in predictions]

df.to_csv(OUTPUT_CSV, index=False)