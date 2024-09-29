#### Translation Code

import pandas as pd
import ast
import os
from google.cloud import translate_v2 as translate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_reviews(df_chunk):
    translate_client = translate.Client()
    
    df_chunk.reset_index(drop=True, inplace=True)
    
    all_reviews = []
    review_positions = []  

    for idx, row in df_chunk.iterrows():
        reviews_str = row['reviews_text']
        if pd.isnull(reviews_str) or reviews_str.strip() == '[]':
            continue
        if not reviews_str.strip().startswith('[') or not reviews_str.strip().endswith(']'):
            logger.warning(f"Malformed reviews in row {idx}")
            continue
        try:
            reviews_list = ast.literal_eval(reviews_str)
            if not isinstance(reviews_list, list):
                logger.warning(f"Expected a list in row {idx}, got {type(reviews_list)}")
                continue
            for i, review in enumerate(reviews_list):
                all_reviews.append(review)
                review_positions.append((idx, i))
        except (ValueError, SyntaxError) as e:
            logger.error(f"Error parsing reviews for row {idx}: {e}")
            continue

    unique_reviews = list(set(all_reviews))
    review_to_translation = {}

    batch_size = 100  
    for i in range(0, len(unique_reviews), batch_size):
        batch = unique_reviews[i:i+batch_size]
        try:
            results = translate_client.translate(batch, target_language='en')
            for original_text, result in zip(batch, results):
                translated_text = result['translatedText']
                review_to_translation[original_text] = translated_text
        except Exception as e:
            logger.error(f"Error translating batch starting at index {i}: {e}")
            for text in batch:
                review_to_translation[text] = text

    for idx, row in df_chunk.iterrows():
        reviews_str = row['reviews_text']
        if pd.isnull(reviews_str) or reviews_str.strip() == '[]':
            continue
        if not reviews_str.strip().startswith('[') or not reviews_str.strip().endswith(']'):
            continue
        try:
            reviews_list = ast.literal_eval(reviews_str)
            if not isinstance(reviews_list, list):
                continue
            updated_reviews = []
            for review in reviews_list:
                translated_review = review_to_translation.get(review, review)
                updated_reviews.append(translated_review)
            df_chunk.at[idx, 'reviews_text'] = str(updated_reviews)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Error updating reviews for row {idx}: {e}")
            continue

    return df_chunk

def translate_reviews_threading(csv_file, output_file):
    chunksize = 10000 
    df_iter = pd.read_csv(csv_file, chunksize=chunksize)

    processed_chunks = []
    chunk_num = 0

    max_workers = 2 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk_num = {}
        futures = []
        for df_chunk in df_iter:
            future = executor.submit(process_reviews, df_chunk)
            future_to_chunk_num[future] = chunk_num
            chunk_num += 1
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
            try:
                processed_chunk = future.result()
                chunk_order = future_to_chunk_num[future]
                processed_chunks.append((chunk_order, processed_chunk))
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

    processed_chunks.sort(key=lambda x: x[0])
    dataframes = [chunk for _, chunk in processed_chunks]
    df_processed = pd.concat(dataframes, ignore_index=True)
    df_processed.to_csv(output_file, index=False)
 
if __name__ == "__main__":
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'API_KEY'

    translate_reviews_threading('dubai_expanded_filtered.csv', 'dubai_translated.csv')