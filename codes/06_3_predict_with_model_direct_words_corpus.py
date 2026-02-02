import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import os
import numpy as np

MODEL_DIR = "models/finetuned_categories"
INPUT_CSV = "data/cleaned/direct_corpus.csv"
OUTPUT_CSV = "data/cleaned/direct_corpus_predicted.csv"
BATCH_SIZE = 64
MAX_LEN = 256
NUM_LABELS = 2

device = torch.device('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=NUM_LABELS)
model.to(device)
model.eval()

df = pd.read_csv(INPUT_CSV)
texts = df["text"].astype(str).tolist() 

encoded_dict = tokenizer.batch_encode_plus(
    texts,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

input_ids = encoded_dict['input_ids']
attention_masks = encoded_dict['attention_mask']

dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=BATCH_SIZE, num_workers=2)

predictions = []

for batch in dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    
    with torch.no_grad():
        result = model(b_input_ids, attention_mask=b_input_mask)
    
    logits = result.logits
    pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
    predictions.extend(pred_ids)

df['is_negative'] = predictions

df.to_csv(OUTPUT_CSV, index=False)