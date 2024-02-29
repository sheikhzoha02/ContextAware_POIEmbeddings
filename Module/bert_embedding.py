from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import pandas as pd

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_word_vector_bert(word):
    input_ids = tokenizer(word, return_tensors='pt')['input_ids']
    with torch.no_grad():
        output = model(input_ids)['last_hidden_state']

    word_vector = output[:, 0, :]
    linear_layer = nn.Linear(word_vector.shape[-1], 32)
    reduced_vector = linear_layer(torch.tensor(word_vector).float())
    return reduced_vector.flatten()


df = pd.read_csv('/home/iailab41/sheikhz0/BaseWord2Vec/data/poi_nyc_clean.csv')
data = pd.read_csv('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings_3.csv')

unique_records_fclass = df['fclass'].unique()
unique_records_1st_level = df['1st_level'].unique()


word_embeddings_fclass = {}
for word in unique_records_fclass:
    embeddings = get_word_vector_bert(word.replace("_", " ")).tolist()
    word_embeddings_fclass[word] = embeddings


word_embeddings_1st_level = {}
for word in unique_records_1st_level:
    embeddings = get_word_vector_bert(word.replace("_", " ")).tolist()
    word_embeddings_1st_level[word] = embeddings

vectors = []
vectors_1stlevel = []
for index, row in tqdm(df.iterrows()):
    vectors.append(word_embeddings_fclass[row['fclass']])
    vectors_1stlevel.append(word_embeddings_1st_level[row['1st_level']])

data['fclass_embedding'] = vectors
data['1stlevel_embedding'] = vectors_1stlevel

data.to_csv('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings_4.csv', index=False)