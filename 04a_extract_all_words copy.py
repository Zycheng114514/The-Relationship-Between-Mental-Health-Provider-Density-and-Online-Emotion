import pandas as pd

# --- Configuration ---
SOURCE_FILE_WITH_DATE = "data/cleaned/all_words_corpus.csv"      # The one you just ran
LABELLED_FILE = "data/cleaned/all_words_corpus_predicted.csv"     # The one with is_negative
OUTPUT_FILE = "data/cleaned/all_words_corpus_labelled_with_date.csv"

print("Loading data...")

# 1. Load the file that has the dates (we only need ID and Date)
# usecols is faster and uses less memory
df_date = pd.read_csv(SOURCE_FILE_WITH_DATE, usecols=['id', 'date'], dtype={'id': str})

# 2. Load the labelled file
df_labelled = pd.read_csv(LABELLED_FILE, dtype={'id': str})

print("Merging dates...")

# 3. Merge them based on 'id'
# how='left' keeps all your labelled rows intact
df_final = pd.merge(df_labelled, df_date, on='id', how='left')

# 4. Save the result
print(f"Saving to {OUTPUT_FILE}...")
df_final.to_csv(OUTPUT_FILE, index=False)

print("Done.")