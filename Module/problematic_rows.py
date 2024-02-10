import pandas as pd

edge_weights_file = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_category_edge_weight.csv'

problematic_rows_count = 0

try:
    chunksize = 10000
    for chunk in pd.read_csv(edge_weights_file, chunksize=chunksize, error_bad_lines=False):
        try:
            problematic_rows = chunk[chunk.isnull().any(axis=1)]
            if not problematic_rows.empty:
                problematic_rows_count += len(problematic_rows)
        except pd.errors.ParserError as e:
            print(f"ParserError in chunk: {e}")

except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

print(f"Total count of problematic rows: {problematic_rows_count}")
