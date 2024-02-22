import pandas as pd
import fasttext
import numpy as np
from tqdm import tqdm
import torch

df = pd.read_csv('/home/iailab41/sheikhz0/BaseWord2Vec/data/poi_nyc_clean.csv')
data = pd.read_csv('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings.csv')
model = fasttext.load_model('/home/iailab41/sheikhz0/BaseWord2Vec/models/fasttext_model_32dim.bin')

vectors = []
vectors_1stlevel = []
for index, row in tqdm(df.iterrows()):
    vectors.append(model.get_word_vector(row['fclass']).tolist())
    vectors_1stlevel.append(model.get_word_vector(row['1st_level']).tolist())

data['fclass_embedding'] = vectors
data['1stlevel_embedding'] = vectors_1stlevel

data.to_csv('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings.csv', index=False)