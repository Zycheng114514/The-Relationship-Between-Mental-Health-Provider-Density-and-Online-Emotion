"""
Count observations for each city and year in negative folder, then, combine the result with total observation counts and independent variables.
"""

import os
import pandas as pd

NEGATIVE_DIR = "data/processed/by_city/negative"
TOTAL_COUNTS_FILE = "data/cleaned/city_total_observation_counts.csv"
INDEPENDENT_VARS_FILE = "data/cleaned/independent_variables_final.csv"
OUTPUT_DIR = "data/cleaned"
OUTPUT_FILE = "the_data.csv"

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

CITY_NAME_MAP = {
    "Chicago": "Chicago city",
    "LosAngeles": "Los Angeles city",
    "NYC": "New York city",
    "Portland": "Portland city",
    "Atlanta": "Atlanta city",
    "Austin": "Austin city",
    "Boston": "Boston city",
    "Houston": "Houston city",
    "Philadelphia": "Philadelphia city",
    "SanFrancisco": "San Francisco city",
    "Seattle": "Seattle city",
}

def count_by_year(filepath):
    df = pd.read_csv(filepath, dtype=str, low_memory=False)
    df['year'] = pd.to_datetime(df['created_utc']).dt.year
    return df.groupby('year').size().to_dict()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

negative_counts = []
for city in CITY_LIST:
    submissions_path = os.path.join(NEGATIVE_DIR, f"{city}_submissions.csv")
    comments_path = os.path.join(NEGATIVE_DIR, f"{city}_comments.csv")
    
    submissions_by_year = count_by_year(submissions_path)
    comments_by_year = count_by_year(comments_path)
    
    all_years = set(submissions_by_year.keys()) | set(comments_by_year.keys())
    
    for year in all_years:
        sub_count = submissions_by_year.get(year, 0)
        com_count = comments_by_year.get(year, 0)
        negative_counts.append({
            'city': city,
            'year': int(year),
            'negative_submissions': sub_count,
            'negative_comments': com_count,
            'negative_total': sub_count + com_count
        })

negative_df = pd.DataFrame(negative_counts)

total_df = pd.read_csv(TOTAL_COUNTS_FILE)

combined_df = pd.merge(negative_df, total_df, on=['city', 'year'], how='left')

combined_df['negative_rate'] = combined_df['negative_total'] / combined_df['total_observations']
combined_df['negative_submission_rate'] = combined_df['negative_submissions'] / combined_df['total_submissions']
combined_df['negative_comment_rate'] = combined_df['negative_comments'] / combined_df['total_comments']

combined_df['city_full'] = combined_df['city'].map(CITY_NAME_MAP)

independent_df = pd.read_csv(INDEPENDENT_VARS_FILE)
independent_df = independent_df.rename(columns={'city': 'city_full'})

final_df = pd.merge(
    combined_df,
    independent_df,
    on=['city_full', 'year'],
    how='left'
)

final_df = final_df.sort_values(['city', 'year'])

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
final_df.to_csv(output_path, index=False, encoding='utf-8')