import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import sys
import pandas as pd
from sklearn.model_selection import KFold
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
import csv
import argparse

#python Downstream_tasks/population_density_estimation.py --city köln

#set the parser
parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, help='Name of the city')
args = parser.parse_args()

if args.city == 'köln':
    dataset = pd.read_csv('Data/ground_truth/köln/population_density_köln.csv')
    poi_csv = pd.read_csv('Data/NRW/Köln/poi_köln_zoneid.csv')
    poi_emb = torch.load(f"{args.city}_embeddings.tensor")
elif args.city == 'düsseldorf':
    dataset = pd.read_csv('Data/ground_truth/dusseldorf/population_density_dusseldorf.csv')
    poi_csv = pd.read_csv('Data/NRW/Düsseldorf/poi_dusseldorf_zoneid.csv')
    poi_emb = torch.load(f"{args.city}_embeddings.tensor")
elif args.city == 'nyc':
    dataset = pd.read_csv('Data/ground_truth/nyc/population_density_nyc.csv')
    poi_csv = pd.read_csv('Data/NYC/poi_zoneid_nyc.csv')
    poi_emb = torch.load(f"{args.city}_embeddings.tensor")


zone_ids = []
zone_list = []
zone_population = []
poi_zone_grouped = poi_csv.groupby(['ZoneID'])

for zone_id, pois in poi_zone_grouped:
    zone_emb_list = []
    attention_weights = []
    
    for index, poi in pois.iterrows():
        poi_index = int(poi['index'])
        zone_emb_list.append(list(poi_emb[poi_index]))
    
    zone_tensor = torch.tensor(zone_emb_list)
    attention_scores = torch.matmul(zone_tensor, zone_tensor.T)
    attention_weights = F.softmax(attention_scores, dim=1)
    weighted_zone_emb_list = zone_tensor * attention_weights.unsqueeze(-1)
    zone_embedding = torch.sum(weighted_zone_emb_list, dim=0)
    zone_embedding = torch.sum(zone_embedding, dim=0)
    zone_ids.append(zone_id)
    if zone_id in dataset['ZoneID'].values:
        population = dataset.loc[dataset['ZoneID'] == zone_id, 'population'].values[0]
    else:
        population = 0.0
        
    zone_population.append(population)
    zone_list.append(zone_embedding)


zone_temp_list = torch.stack(zone_list)
zone_final_list = list(zip(zone_temp_list.tolist(), list(zone_population)))


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

rmse_list = []
r2_list = []
mae_list = []
zone_ids_list = []

for fold, (train_index, test_index) in enumerate(kf.split(zone_final_list)):
    print("Fold:", fold+1)
    
    x_train, x_test = np.array([zone_final_list[i][0] for i in train_index]), np.array([zone_final_list[i][0] for i in test_index])
    y_train, y_test = np.array([zone_final_list[i][1] for i in train_index]), np.array([zone_final_list[i][1] for i in test_index])
    test_zone_ids = [zone_ids[i] for i in test_index]

    rf = RandomForestRegressor(random_state=fold)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    #save the actual and predicted populations in csv for deeper analysis
    with open(f'{args.city}_population.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for zone_id, actual, predicted in zip(test_zone_ids, y_test, y_pred):
            writer.writerow([zone_id, actual, predicted])

    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    rmse_list.append(rmse)
    print("RMSE:", rmse)
    
    r2 = metrics.r2_score(y_test, y_pred)
    r2_list.append(r2)
    print("R^2 Score:", r2)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mae_list.append(mae)
    print("MAE:", mae)


avg_rmse = np.mean(rmse_list)
avg_r2 = np.mean(r2_list)
avg_mae = np.mean(mae_list)

print("\nAverage RMSE:", avg_rmse)
print("Average R^2 Score:", avg_r2)
print("Average MAE:", avg_mae)