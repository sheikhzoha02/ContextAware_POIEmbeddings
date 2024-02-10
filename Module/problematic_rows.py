import pandas as pd

# Specify the file causing the issue
edge_weights_file = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_category_edge_weight.csv'

# Initialize a counter for problematic rows
problematic_rows_count = 0

try:
    # Use chunksize to read the file in chunks
    chunksize = 10000  # Adjust this based on your file size
    for chunk in pd.read_csv(edge_weights_file, chunksize=chunksize, error_bad_lines=False):
        # Iterate through each chunk and count the problematic rows
        try:
            # Check for any NaN values in the chunk
            problematic_rows = chunk[chunk.isnull().any(axis=1)]
            if not problematic_rows.empty:
                problematic_rows_count += len(problematic_rows)
        except pd.errors.ParserError as e:
            print(f"ParserError in chunk: {e}")
            # Handle the error or log the problematic chunk

except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
    # Handle the error or log the problematic line number

print(f"Total count of problematic rows: {problematic_rows_count}")
