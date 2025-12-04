"""
Script to count ALL observations(posts/comments) in raw ZST files, grouped by city and year.
"""

import zstandard
import os
import json
import multiprocessing as mp
import pandas as pd
from datetime import datetime

ZST_FILES_LIST = "data/zst_files_list.txt"
ZST_INPUT_DIR = "data/raw/reddit_zst"
OUTPUT_DIR = "data/cleaned"
OUTPUT_FILE = "city_total_observation_counts.csv"

NUM_WORKERS = None  # None means use all available cores

CITY_SUBREDDIT_MAP = {
    "Chicago": ["AskChicago", "chicago"],
    "LosAngeles": ["AskLosAngeles", "LosAngeles"],
    "NYC": ["AskNYC", "nyc"],
    "Houston": ["houston"],
    "Philadelphia": ["philadelphia"],
    "SanFrancisco": ["sanfrancisco"],
    "Seattle": ["Seattle", "SeattleWA"],
    "Portland": ["askportland", "Portland"],
    "Atlanta": ["Atlanta"],
    "Austin": ["Austin", "UTAustin"],
    "Boston": ["BostonBruins", "bostonceltics", "boston"],
}

SUBREDDIT_TO_CITY = {}
for city, subreddits in CITY_SUBREDDIT_MAP.items():
    for subreddit in subreddits:
        SUBREDDIT_TO_CITY[subreddit.lower()] = city

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)
    
def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip(), file_handle.tell()
            buffer = lines[-1]
        reader.close()

def count_zst_file(args):
    input_file, subreddit, file_type = args
    year_counts = {}
    
    for line, file_bytes_processed_useless_here in read_lines_zst(input_file):
        obj = json.loads(line)
        
        created_utc = obj.get('created_utc')
        if created_utc:
            year = datetime.fromtimestamp(int(created_utc)).year
            if year not in year_counts:
                year_counts[year] = 0
            year_counts[year] += 1
    
    return {
        'subreddit': subreddit,
        'file_type': file_type,
        'year_counts': year_counts,
    }

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

zst_files = []
with open(ZST_FILES_LIST, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and line.endswith('.zst'):
            zst_files.append(line)

tasks = []
for zst_file in zst_files:
    input_path = os.path.join(ZST_INPUT_DIR, zst_file)
    filename_no_ext = zst_file.replace('.zst', '')
    parts = filename_no_ext.rsplit('_', 1)
    subreddit = parts[0]
    file_type = parts[1]
    
    tasks.append((input_path, subreddit, file_type))

num_workers = NUM_WORKERS if NUM_WORKERS else mp.cpu_count()
num_workers = min(num_workers, len(tasks))

if num_workers > 1 and len(tasks) > 1:
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(count_zst_file, tasks)
else:
    results = [count_zst_file(task) for task in tasks]

city_year_counts = {}

for result in results:
    subreddit = result['subreddit']
    file_type = result['file_type']
    year_counts = result['year_counts']
    city = SUBREDDIT_TO_CITY.get(subreddit.lower())
    
    if city is None:
        continue
    
    for year, count in year_counts.items():
        key = (city, year)
        if key not in city_year_counts:
            city_year_counts[key] = {'submissions': 0, 'comments': 0}
        
        if file_type == 'submissions':
            city_year_counts[key]['submissions'] += count
        elif file_type == 'comments':
            city_year_counts[key]['comments'] += count

rows = []
for (city, year) in sorted(city_year_counts.keys()):
    rows.append({
        'city': city,
        'year': year,
        'total_submissions': city_year_counts[(city, year)]['submissions'],
        'total_comments': city_year_counts[(city, year)]['comments'],
        'total_observations': city_year_counts[(city, year)]['submissions'] + city_year_counts[(city, year)]['comments']
    })

df = pd.DataFrame(rows)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
df.to_csv(output_path, index=False, encoding='utf-8')
