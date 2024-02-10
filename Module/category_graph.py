import networkx as nx
import matplotlib.pyplot as plt
import csv
import osmnx as ox
from tqdm import tqdm
import sys
import pandas as pd

categories = {}

def filter_records(csv_file, parent, child):
    found_records = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if custom_condition(row, parent, child):
                found_records.append(row)
    return found_records

def custom_condition(record, first_level, second_level):
    return record['FirstLevel'] == first_level and record['SecondLeve'] == second_level


def filter_records_a(csv_file, parent, child, other_child):
    found_records = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if custom_condition_a(row, parent, child, other_child):
                found_records.append(row)
    return found_records

def custom_condition_a(record, first_level, second_level, other_child):
    return (record['SecondLeve'] == second_level or record['SecondLeve'] == other_child) and record['FirstLevel'] == first_level

file_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NRW/POI_ZoneID_NRW.csv'
with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        parent = row['FirstLevel']
        child = row['SecondLeve']
        if parent not in categories:
            categories[parent] = {child}
        else:
            categories[parent].add(child)

existing_ids = set()
existing_csv_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NRW/nrw_category_edge_weight.csv'
existing_data = pd.read_csv(existing_csv_path)
existing_ids = set(existing_data['source'].astype(str) + ',' + existing_data['target'].astype(str))

with open('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NRW/nrw_category_edge_weight.csv', 'a') as f:
    for parent, children in tqdm(categories.items()):
        for child in children:
            matching_records = filter_records(file_path, parent, child)
            for record_a in matching_records:
              for record_b in matching_records:
                if record_a['index'] != record_b['index']:
                    id = str(int(record_a['index']))+','+str(int(record_b['index']))
                    if id not in existing_ids:
                      distance = ox.distance.great_circle_vec(float(record_a['Lat']), float(record_a['Lon']), float(record_b['Lat']), float(record_b['Lon']))
                      if distance <= 500:
                          f.write(str(int(record_a['index']))+','+str(int(record_b['index']))+','+str(distance)+'\n')
                          existing_ids.add(id)

            other_children = [c for c in children if c != child]
            for other_child in other_children:
              matching_records = filter_records_a(file_path, parent, child, other_child)
              for record_a in matching_records:
                for record_b in matching_records:
                  if record_a['index'] != record_b['index']:
                      id = str(int(record_a['index']))+','+str(int(record_b['index']))
                      if id not in existing_ids:
                        distance = ox.distance.great_circle_vec(float(record_a['Lat']), float(record_a['Lon']), float(record_b['Lat']), float(record_b['Lon']))
                        if distance <= 500:
                            f.write(str(int(record_a['index']))+','+str(int(record_b['index']))+','+str(distance)+'\n')
                            existing_ids.add(id)

print('success')