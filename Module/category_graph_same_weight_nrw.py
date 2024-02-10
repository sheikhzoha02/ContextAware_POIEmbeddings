import csv
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import sys
import pandas as pd
from tqdm import tqdm
import osmnx as ox
import sys
import multiprocessing

pairs_as_integers = []

parent_child_dict = defaultdict(list)
data_all = pd.read_csv('/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NRW.csv')

with open('/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NRW.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader) 
    for row in csvreader:
        parent_id = row[3]
        child_id = row[4]
        parent_child_dict[parent_id].append(child_id)

new_parent_child_dict = {}
for parent, child_ids in parent_child_dict.items():
    unique_child_ids = list(set(child_ids))
    new_parent_child_dict[parent] = unique_child_ids

unique_pairs_same_parent = set()
#different_parent_pairs = set()

for child_ids in new_parent_child_dict.values():
    if len(child_ids) >= 2:
        pairs = list(combinations(child_ids, 2))
        unique_pairs_same_parent.update(set(pairs))


#all_child_ids = [child_id for child_ids in parent_child_dict.values() for child_id in child_ids]
#all_pairs = list(combinations(all_child_ids, 2))
#different_parent_pairs = set(all_pairs) - unique_pairs_same_parent

same_parent_pairs_as_integers = [tuple(map(int, pair)) for pair in unique_pairs_same_parent]
print(same_parent_pairs_as_integers)
sys.exit()
#different_parent_pairs_as_integers = [tuple(map(int, pair)) for pair in different_parent_pairs]
existing_ids = set()

def process_chunk(chunk):
    with open('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NRW/nrw_category_edge_weight_new.csv', 'a') as f:
        for value in chunk:
#            f.write('source,target,weight,distance\n')
            condition = data_all['SecondLeve'] == value[0]
            results_0 = data_all[condition]
            condition = data_all['SecondLeve'] == value[1]
            results_1 = data_all[condition]
            for index, record_a in results_0.iterrows():
                for index, record_b in results_1.iterrows():
                    if record_a['index'] != record_b['index']:    
                        id = str(int(record_a['index']))+','+str(int(record_b['index']))
                        if id not in existing_ids:
                            distance = ox.distance.great_circle_vec(float(record_a['Lat']), float(record_a['Lon']), float(record_b['Lat']), float(record_b['Lon']))
                            if distance <= 1000:
                                    f.write(str(int(record_a['index']))+','+str(int(record_b['index']))+','+str(1.0)+','+str(distance)+'\n')
                                    existing_ids.add(id)

def split_into_chunks(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

if __name__ == "__main__":
    chunk_size = 10
    chunks = list(split_into_chunks(same_parent_pairs_as_integers, chunk_size))

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Processing Chunks"))


#print('Different Parent Ids')
#with open('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_category_edge_weight_new.csv', 'a') as f:
#    for children_id in tqdm(different_parent_pairs_as_integers):
#        condition = data_all['SecondLeve'] == children_id[0]
#        results_0 = data_all[condition]
#        condition = data_all['SecondLeve'] == children_id[1]
#        results_1 = data_all[condition]
#        for index, record_a in results_0.iterrows():
#            for index, record_b in results_1.iterrows():
#                if record_a['index'] != record_b['index']:    
#                    id = str(int(record_a['index']))+','+str(int(record_b['index']))
#                    if id not in existing_ids:
#                          f.write(str(int(record_a['index']))+','+str(int(record_b['index']))+','+str(0.5)+'\n')
#                          existing_ids.add(id)
