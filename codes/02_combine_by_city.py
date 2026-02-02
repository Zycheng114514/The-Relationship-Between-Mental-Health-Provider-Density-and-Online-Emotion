"""
This script combines the individual subreddit CSV outputs 
into one submissions file and one comments file per city and put the result under processed/by_city/.
"""

import os
import pandas as pd

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed/by_city"

CITY_SUBREDDIT_MAP = {
    "Chicago": ["AskChicago", "chicago"],
    "LosAngeles": ["AskLosAngeles", "LosAngeles"],
    "NYC": ["AskNYC", "nyc"],
    "Portland": ["askportland", "Portland"],
    "Atlanta": ["Atlanta"],
    "Austin": ["Austin", "UTAustin"],
    "Boston": ["BostonBruins", "bostonceltics", "boston"],
    "Houston": ["houston"],
    "Philadelphia": ["philadelphia"],
    "SanFrancisco": ["sanfrancisco"],
    "Seattle": ["Seattle", "SeattleWA"],
}

def find_csv_files(input_dir, subreddit_prefix, file_type):
    expected_filename = f"{subreddit_prefix}_{file_type}_keyword_matches.csv"
    filepath = os.path.join(input_dir, expected_filename)
    return filepath

def combine_csv_files(file_list, output_path):

    dfs = []

    for filepath in file_list:
        df = pd.read_csv(filepath, dtype=str, low_memory=False)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    
    if 'id' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')

    if 'created_utc' in combined_df.columns:
        combined_df = combined_df.sort_values('created_utc', ascending=True)
    
    combined_df.to_csv(output_path, index=False, encoding='utf-8')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for city, subreddits in CITY_SUBREDDIT_MAP.items():
    submission_files = []
    for subreddit in subreddits:
        submission_files.extend(find_csv_files(INPUT_DIR, subreddit, "submissions"))
        submissions_output = os.path.join(OUTPUT_DIR, f"{city}_submissions.csv")
        combine_csv_files(submission_files, submissions_output)
    
    comment_files = []
    for subreddit in subreddits:
        comment_files.extend(find_csv_files(INPUT_DIR, subreddit, "comments"))
        comments_output = os.path.join(OUTPUT_DIR, f"{city}_comments.csv")
        combine_csv_files(comment_files, comments_output)