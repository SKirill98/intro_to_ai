import pandas as pd
import numpy as np
from utils import (load_and_preprocess, calculate_similarity_method1_bow, calculate_similarity_tfidf,
                   calculate_similarity_hashing, get_top_pairs)
import time

if __name__ == "__main__":
    SAMPLE_SIZE = 5000
    FILE_PATH = 'data/train.csv'

    processed_df = load_and_preprocess(FILE_PATH, sample_n=SAMPLE_SIZE)
    processed_df.head()

    if processed_df is not None:
        start_time = time.time()

        similarity_method = calculate_similarity_hashing

        count_matrix, similarity_matrix, top_pairs = get_top_pairs(similarity_method
                                                                   , processed_df
                                                                   , top_n=5
                                                                   )

        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")