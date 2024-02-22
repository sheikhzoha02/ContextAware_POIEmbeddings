import pandas as pd
import h3
import geohash2
import chars2vec

def get_geohash(latitude, longitude, precision=12):
    return geohash2.encode(latitude, longitude, precision=precision)

def char2vec_conversion(words):
    c2v_model = chars2vec.load_model('eng_50')
    word_embeddings = c2v_model.vectorize_words(words)
    return word_embeddings

csv_file_path = '/content/POI_ZoneID_NYC.csv'
df = pd.read_csv(csv_file_path)

geohash_precision = 12
geohash_embeddings_list = []

for index, row in df.iterrows():
    lat, lon = row['Lat'], row['Lon']
    geohash_representation = get_geohash(lat, lon, geohash_precision)
    geohash_embeddings_list.append(geohash_representation)

geohash_embeddings = char2vec_conversion(geohash_embeddings_list)

df['geohash_embedding'] = geohash_embeddings.tolist()

output_csv_path = 'output_with_embeddings.csv'
df.to_csv(output_csv_path, index=False)