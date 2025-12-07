import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, r2_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import time
import datetime
import random
import os

DATA_PATH = "data/cleaned/direct_corpus_labelled.csv"
OUTPUT_DIR = "models/finetuned_categories"
MODEL_NAME = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
K_FOLDS = 4
SEED = 114514
NUM_LABELS = 2

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

set_seed(SEED)
device = torch.device('cuda:0')

df = pd.read_csv(DATA_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input_ids = []
attention_masks = []

encoded_dict = tokenizer.batch_encode_plus(
    df['text'].tolist(),
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = encoded_dict['input_ids']
attention_masks = encoded_dict['attention_mask']

labels = torch.tensor(df['is_negative'].values, dtype=torch.long)
dataset = TensorDataset(input_ids, attention_masks, labels)

kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    train_subsampler = SubsetRandomSampler(train_idx)
    val_subsampler = SubsetRandomSampler(val_idx)
    
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    val_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch_i in range(0, EPOCHS):
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = result.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    del model
    torch.cuda.empty_cache()

full_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(dataset))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True 
)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(full_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch_i in range(0, EPOCHS):
    model.train()
    for step, batch in enumerate(full_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        model.zero_grad()
        result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = result.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)