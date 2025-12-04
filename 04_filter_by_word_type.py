"""
Filter CSVs in processed/bycity to keep rows that:
- at least 1 direct indication word appears
- at least 2 indirect indication words appear.
"""

import os
import pandas as pd
import re

INPUT_DIR = "data/processed/by_city"
OUTPUT_DIR = "data/processed/by_city/negative"
DIRECT_WORDS_FILE = "data/word_list_direct_words.txt"
INDIRECT_WORDS_FILE = "data/word_list_indirect_words.txt"

CITY_LIST = [
    "Chicago",
    "LosAngeles",
    "NYC",
    "Portland",
    "Atlanta",
    "Austin",
    "Boston",
    "Houston",
    "Philadelphia",
    "SanFrancisco",
    "Seattle",
]

def load_word_set(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        words = set()
        for word in content.split(','):
            words.add(word.strip().lower())
    return words

def check_row_criteria(matched_keywords_str, text_str, direct_words, indirect_words):
    
    keywords = [kw.strip().lower() for kw in matched_keywords_str.split(',')]
    direct_count = sum(1 for kw in keywords if kw in direct_words)
    if direct_count >= 1:
        return True
    
    sorted_keywords = sorted(indirect_words, key=len, reverse=True)
    pattern = r'\b(' + '|'.join(sorted_keywords) + r')\b'
    keyword_pattern = re.compile(pattern, re.IGNORECASE)

    matches = keyword_pattern.findall(text_str.lower())
    if len(matches) >= 2:
        return True

    return False

def filter_csv(input_path, output_path, direct_words, indirect_words,type):
    df = pd.read_csv(input_path, dtype=str, low_memory=False)
    
    if type == "submissions":
        df["full_text"]=df['title'].fillna('') + ' ' + df['selftext'].fillna('')
        mask = df.apply(
            lambda x: check_row_criteria(x['matched_keywords'],x["full_text"],direct_words,indirect_words),
            axis=1
        )
    else:
        mask = df.apply(
            lambda x: check_row_criteria(x['matched_keywords'],x["body"],direct_words,indirect_words),
            axis=1
        )

    filtered_df = df[mask]
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

direct_words = load_word_set(DIRECT_WORDS_FILE)
indirect_words = load_word_set(INDIRECT_WORDS_FILE)

for city in CITY_LIST:
    submissions_input = os.path.join(INPUT_DIR, f"{city}_submissions.csv")
    submissions_output = os.path.join(OUTPUT_DIR, f"{city}_submissions.csv")
    filter_csv(submissions_input, submissions_output, direct_words, indirect_words, "submissions")
    
    comments_input = os.path.join(INPUT_DIR, f"{city}_comments.csv")
    comments_output = os.path.join(OUTPUT_DIR, f"{city}_comments.csv")
    filter_csv(comments_input, comments_output, direct_words, indirect_words, "comments")
