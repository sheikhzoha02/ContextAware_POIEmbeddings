import pandas as pd
import h3
import geohash2
import sys

def generate_one_hot_encoding(input_string):
    valid_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    one_hot_encoding = []
    for char in valid_characters:
        if char in input_string:
            one_hot_encoding.append(1)
        else:
            one_hot_encoding.append(0)
    return one_hot_encoding

def get_geohash(latitude, longitude, precision=12):
    return geohash2.encode(latitude, longitude, precision=precision)

csv_file_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings_1.csv'
df = pd.read_csv(csv_file_path)

geohash_precision = 12
geohash_embeddings_list = []

for index, row in df.iterrows():
    lat, lon = row['Lat'], row['Lon']
    geohash_representation = get_geohash(lat, lon, geohash_precision)
    one_hot_encoding = generate_one_hot_encoding(geohash_representation.upper())
    geohash_embeddings_list.append(one_hot_encoding)

df['geohash_embedding'] = geohash_embeddings_list

output_csv_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings_3.csv'
df.to_csv(output_csv_path, index=False)
