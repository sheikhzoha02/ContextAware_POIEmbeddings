from tqdm import tqdm
import osmnx as ox
import networkx as nx
import pandas as pd
from haversine import haversine

file_path = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NYC.csv'
pois_data = pd.read_csv(file_path)
data = pois_data.to_numpy()

existing_ids = set()
with open('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/test.csv', 'w') as f:
    f.write('source,target,weight\n')
    for record_a in tqdm(data):
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
                        existing_ids.add(id)
            elif id_2 in existing_ids or id in existing_ids:
                if id_2 in existing_ids and id in existing_ids:
                    continue
                file_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/test.csv'
                edge_data = pd.read_csv(file_path)
                match_condition = (
                    (edge_data['source'] == int(record_a[6])) & (edge_data['target'] == int(record_b[6])) |
                    (edge_data['source'] == int(record_b[6])) & (edge_data['target'] == int(record_a[6]))
                )
                
                matched_rows = edge_data[match_condition]
                
                if not matched_rows.empty:
                    weight = matched_rows['weight'].values[0]
                    f.write(str(int(record_a[6]))+','+str(int(record_b[6]))+','+str(weight)+'\n')
                    existing_ids.add(id)
