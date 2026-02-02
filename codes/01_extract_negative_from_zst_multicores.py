"""
extract submissions and comments from zst files that contain any of a list of keywords 
and put the result of each zst file separately into processed CSV files.
"""

import zstandard
import os
import json
import csv
import re
import multiprocessing as mp

ZST_FILES_LIST="data/zst_files_list.txt"
WORD_LIST_FILE="data/word_list.txt"
ZST_INPUT_DIR="data/raw/reddit_zst"
OUTPUT_DIR="data/processed"

NUM_WORKERS = None # NONE means use all available cores

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

def get_searchable_text(obj, is_submission):
    texts = []

    if is_submission:
        if 'title' in obj and obj['title']:
            texts.append(obj['title'])
        if 'selftext' in obj and obj['selftext']:
            texts.append(obj['selftext'])
    else:
        if 'body' in obj and obj['body']:
            texts.append(obj['body'])
    
    return ' '.join(texts)


def process_zst_file(args):
    input_file, output_file, keyword_pattern, is_submission = args
        
    if is_submission:
        field_list = [
            'id', 'author', 'created_utc', 'subreddit', 'score',
            'num_comments', 'matched_keywords', 'title', 'selftext', 'permalink',
        ]
    else:
        field_list = [
            'id', 'author', 'created_utc', 'subreddit', 'score',
            'matched_keywords', 'body', 'link_id', 'parent_id', 'permalink',
        ]
    
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_list, extrasaction='ignore')
        writer.writeheader()

        for line, file_bytes_processed_useless_here in read_lines_zst(input_file):

            obj = json.loads(line)

            searchable_text = get_searchable_text(obj, is_submission)    
            if not searchable_text:
                continue

            matches = keyword_pattern.findall(searchable_text.lower())
            if matches:
                unique_matches = sorted(set(matches))
                obj['matched_keywords'] = ','.join(unique_matches)
                obj['created_utc'] = datetime.fromtimestamp(int(obj['created_utc'])).strftime('%Y-%m-%d %H:%M:%S')
                
                clean_obj = {}
                for field in field_list:
                    value = obj.get(field, '')
                    if value is None:
                        clean_obj[field] = ''
                    elif isinstance(value, (dict, list)):
                        clean_obj[field] = json.dumps(value)
                    else:
                        clean_obj[field] = str(value)
                
                writer.writerow(clean_obj)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

keywords = []
with open(WORD_LIST_FILE, 'r', encoding='utf-8') as f:
    content = f.read()
    for word in content.split(','):
        keywords.append(word.strip().lower())
sorted_keywords = sorted(keywords, key=len, reverse=True)
pattern = r'\b(' + '|'.join(sorted_keywords) + r')\b'
keyword_pattern = re.compile(pattern, re.IGNORECASE)

zst_files = []
with open(ZST_FILES_LIST, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        zst_files.append(line)

tasks = []
for zst_file in zst_files:
    input_path = os.path.join(ZST_INPUT_DIR, zst_file)
    
    is_submission = "submission" in zst_file.lower()
    output_filename = zst_file.replace('.zst', '_keyword_matches.csv')
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    tasks.append((input_path, output_path, keyword_pattern, is_submission))

num_workers = NUM_WORKERS if NUM_WORKERS else mp.cpu_count()
num_workers = min(num_workers, len(tasks))

if num_workers > 1 and len(tasks) > 1:
    with mp.Pool(processes=num_workers) as pool:
        pool.map(process_zst_file, tasks)
else:
    for task in tasks:
        process_zst_file(task)