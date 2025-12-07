import pandas as pd
import glob
import os

INPUT_PATH_1 = "data/cleaned/depression_corpus.csv" 
INPUT_PATH_2 = "data/cleaned/suicide_corpus.csv"
OUTPUT_PATH = "data/cleaned/depression_suicide_corpus.csv"

df_1 = pd.read_csv(INPUT_PATH_1)
df_2 = pd.read_csv(INPUT_PATH_2)
df = pd.concat([df_1, df_2], ignore_index=True)
df = df.reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

