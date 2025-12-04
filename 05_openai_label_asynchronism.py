import os
import re
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH_1 = "data/cleaned/depression_corpus.csv" 
INPUT_PATH_2 = "data/cleaned/suicide_corpus.csv"
OUTPUT_PATH = "data/cleaned/depression_suicide_corpus_labelled.csv"

RESULT_COLUMN = "is_negative"

SAVE_INTERVAL = 500
MAX_CONCURRENT = 15
TIMEOUT_SECONDS = 60

client = AsyncOpenAI(
    base_url="http://api.yesapikey.com/v1"
)

async def chat_with_retry(semaphore, prompt: str, system_prompt: str, retries=3) -> str:
    async with semaphore:
        for attempt in range(retries):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="gpt-5-mini-2025-08-07",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                    ),
                    timeout=TIMEOUT_SECONDS
                )
                return response.choices[0].message.content

            except asyncio.TimeoutError:
                if attempt < retries - 1:
                    print(f"Retrying {attempt+1}")
                    await asyncio.sleep(2) 
                else:
                    print(f"Timeout error after {retries} attempts.")
            except Exception as e:
                print(e)
    return None

async def process_batch(df, batch_indices, semaphore, system_prompt):
    tasks = []
    valid_indices = []

    for i in batch_indices:
        current_val = str(df.at[i, RESULT_COLUMN]).strip()
        if current_val and current_val.lower() != "nan":
            continue
            
        text_content = str(df.at[i, "text"])
        prompt = f'Text: "{text_content}"'
        
        task = chat_with_retry(semaphore, prompt, system_prompt)
        tasks.append(task)
        valid_indices.append(i)

    if not tasks:
        return False

    results = await asyncio.gather(*tasks)

    updates = 0
    for idx, response in zip(valid_indices, results):
        if response:
            match = re.search(r'\b([01])\b', response)
            
            if match:
                result = match.group(1)
                df.at[idx, RESULT_COLUMN] = result
                print(f"Row {idx} Success: {result}")
                updates += 1
            else:
                fallback_match = re.search(r'([01])', response)
                if fallback_match:
                     result = fallback_match.group(1)
                     df.at[idx, RESULT_COLUMN] = result
                     print(f"Row {idx} Success (fallback): {result}")
                     updates += 1
                else:
                    print(f"Row {idx} Parse Error: {response}")
        else:
            print(f"Row {idx} Failed")
    
    return updates > 0

async def main():
    df_1 = pd.read_csv(INPUT_PATH_1)
    df_2 = pd.read_csv(INPUT_PATH_2)
    df = pd.concat([df_1, df_2], ignore_index=True)
    df = df.sample(n=4000)
    df = df.reset_index(drop=True)

    if os.path.exists(OUTPUT_PATH):
        df_existing = pd.read_csv(OUTPUT_PATH)
        if RESULT_COLUMN not in df.columns:
            df[RESULT_COLUMN] = ""
        if RESULT_COLUMN in df_existing.columns:
             df.update(df_existing[[RESULT_COLUMN]])
    
    if RESULT_COLUMN not in df.columns:
        df[RESULT_COLUMN] = ""

    system_prompt = """
    You are a sentiment classifier. Analyze the text provided.
    Does the text express negative feelings (e.g., depression, hopelessness, sadness, anxiety, suicide)?
    
    Return '1' if YES.
    Return '0' if NO.
    
    Output ONLY the single number (0 or 1). Do not output words.
    """

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    total_rows = len(df)
    for i in range(0, total_rows, SAVE_INTERVAL):
        batch_end = min(i + SAVE_INTERVAL, total_rows)
        batch_indices = range(i, batch_end)
        
        needs_save = await process_batch(df, batch_indices, semaphore, system_prompt)
        
        if needs_save:
            df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(main())