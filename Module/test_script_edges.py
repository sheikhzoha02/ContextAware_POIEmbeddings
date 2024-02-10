from tqdm import tqdm
import networkx as nx
import pandas as pd
from haversine import haversine
import itertools
import sys

file_path = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NRW.csv'
pois_data = pd.read_csv(file_path)
data = pois_data.to_numpy()

existing_csv_path = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nrw_distance_edge_weights.csv'
existing_data = pd.read_csv(existing_csv_path, low_memory=False).to_numpy()
existing_ids = set()
for data_exist in existing_data:
  id = str(int(data_exist[0]))+','+str(int(data_exist[1]))
  existing_ids.add(id)

with open('/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nrw_distance_edge_weights.csv', 'a') as f:
    #f.write('source,target,weight\n')
    for record_a in tqdm(data):
        value_not_exists = record_a[6] not in existing_data[:, 0]
        if value_not_exists:
            for record_b in data:
                if record_a[0] != record_b[0]:
                    id = str(int(record_a[6]))+','+str(int(record_b[6]))
                    id_2 = str(int(record_b[6]))+','+str(int(record_a[6]))
                    if id not in existing_ids  and id_2 not in existing_ids:
                        coords_a = (float(record_a[2]), float(record_a[1]))
                        coords_b = (float(record_b[2]), float(record_b[1]))                        
                        distance = haversine(coords_a, coords_b)
                        distance = distance * 1000
                        existing_ids.add(id)
                        if distance <= 1000:
                            f.write(str(int(record_a[6]))+','+str(int(record_b[6]))+','+str(distance)+'\n')
