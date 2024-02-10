from tqdm import tqdm
import networkx as nx
import pandas as pd
from haversine import haversine
import itertools
import multiprocessing
import sys

file_path = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NRW.csv'
pois_data = pd.read_csv(file_path)
data = pois_data.to_numpy()

manager = multiprocessing.Manager()
existing_ids = manager.list()

existing_csv_path = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nrw_distance_edge_weights.csv'
existing_data = pd.read_csv(existing_csv_path, low_memory=False).to_numpy()

for data_exist in existing_data:
  id = str(int(data_exist[0]))+','+str(int(data_exist[1]))
  existing_ids.append(id)

def process_chunk(chunk):
    with open('/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nrw_distance_edge_weights.csv', 'a') as f:
        #f.write('source,target,weight\n')
        for record_a in chunk:
            for record_b in data:
                if record_a[0] != record_b[0]:
                    id = str(int(record_a[6]))+','+str(int(record_b[6]))
                    id_2 = str(int(record_b[6]))+','+str(int(record_a[6]))
                    if id not in existing_ids  and id_2 not in existing_ids:
                        coords_a = (float(record_a[2]), float(record_a[1]))
                        coords_b = (float(record_b[2]), float(record_b[1]))                        
                        distance = haversine(coords_a, coords_b)
                        distance = distance * 1000
                        if distance <= 1000:
                            f.write(str(int(record_a[6]))+','+str(int(record_b[6]))+','+str(distance)+'\n')
                            existing_ids.append(id)


def split_into_chunks(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

if __name__ == "__main__":
    chunk_size = 19000
    chunks = list(split_into_chunks(data, chunk_size))

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Processing Chunks"))