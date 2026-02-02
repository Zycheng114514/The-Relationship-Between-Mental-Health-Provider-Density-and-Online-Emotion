import pandas as pd
import glob
import os

INPUT_DIR = "data/processed/by_city/negative"
OUTPUT_DIR = "data/cleaned/direct_corpus.csv"
WORD_LIST_FILE = "data/word_list_direct_words.txt"

depress_words = []
with open(WORD_LIST_FILE, 'r', encoding='utf-8') as f:
    content = f.read()
    for word in content.split(','):
        depress_words.append(word.strip().lower())

submissions = glob.glob(os.path.join(INPUT_DIR, "*_submissions.csv"))
comments = glob.glob(os.path.join(INPUT_DIR, "*_comments.csv"))

df_full = pd.DataFrame(columns=["id", "text", "source_type", "city", "date"])

for file in submissions + comments:
    df = pd.read_csv(file, low_memory=False)
    filename = os.path.basename(file)
    if file.endswith("_submissions.csv"):
        df['text'] = df['title'].fillna('') + " " + df['selftext'].fillna('')
        df['source_type'] = 'submission'
        city_name = filename.replace("_submissions.csv", "")
    elif file.endswith("_comments.csv"):
        df['text'] = df['body'].fillna('')
        df['source_type'] = 'comment'
        city_name = filename.replace("_comments.csv", "")
    df['city'] = city_name
    df['dt_obj'] = pd.to_datetime(df['created_utc'])
    df['date'] = df['dt_obj'].dt.date
    df_full = pd.concat([df_full, df[['id', 'text', 'source_type', 'city', 'date']]], ignore_index=True)

pattern = '|'.join(depress_words)
mask = df_full['text'].str.contains(pattern, case=False, na=False)
filtered_df = df_full[mask].copy()
filtered_df.to_csv(OUTPUT_DIR, index=False)
