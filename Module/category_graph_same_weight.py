import csv
from itertools import combinations
from collections import defaultdict
import sys
import pandas as pd
from tqdm import tqdm
import osmnx as ox
import sys
from haversine import haversine


pairs_as_integers = []

parent_child_dict = defaultdict(list)
data_all = pd.read_csv('/Users/zohasheikh/Downloads/GeoVectors/data/POI_ZoneID_NRW_type.csv')

with open('/Users/zohasheikh/Downloads/GeoVectors/data/POI_ZoneID_NRW_type.csv', 'r') as csvfile:
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

for child_ids in new_parent_child_dict.values():
    if len(child_ids) >= 2:
        pairs = list(combinations(child_ids, 2))
        unique_pairs_same_parent.update(set(pairs))


same_parent_pairs_as_integers = [tuple(map(int, pair)) for pair in unique_pairs_same_parent]


existing_ids = set()
print('Same Parent Ids')
with open('/Users/zohasheikh/Downloads/GeoVectors/data/nrw_category_edge_weights.csv', 'a') as f:
    f.write('source,target,weight,distance\n')
    for children_id in tqdm(same_parent_pairs_as_integers):
        condition = data_all['SecondLeve'] == children_id[0]
        results_0 = data_all[condition].to_numpy()
        condition = data_all['SecondLeve'] == children_id[1]
        results_1 = data_all[condition].to_numpy()

        for record_a in results_0:
            for record_b in results_1:
                if record_a[6] != record_b[6]:    
                    id = str(int(record_a[6]))+','+str(int(record_b[6]))
                    if id not in existing_ids:
                        coords_a = (float(record_a[2]), float(record_a[1]))
                        coords_b = (float(record_b[2]), float(record_b[1]))                        
                        distance = haversine(coords_a, coords_b)
                        distance = distance * 1000
                        if distance <= 1000:
                                f.write(str(int(record_a[6]))+','+str(int(record_b[6]))+','+str(1.0)+','+str(distance)+'\n')
                                existing_ids.add(id)