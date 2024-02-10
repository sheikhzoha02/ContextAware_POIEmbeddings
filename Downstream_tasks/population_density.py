import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


#dataset = pd.read_csv('/home/iailab41/sheikhz0/HGI/Data/ground_truth/nrw/population_density_nrw.csv')
dataset = pd.read_csv('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/population_density_nyc.csv')

poi_emb = torch.load('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/poi_embedding_nyc.tensor')
#NYC GeoVectors
#poi_emb = torch.load("/home/iailab41/sheikhz0/GeoVectors/embeddings_merged/poi_embedding_merged_nyc_32.tensor")
#NYC Base2vec
#poi_emb = torch.load("/home/iailab41/sheikhz0/BaseWord2Vec/poi_word_embedding_nyc.tensor")
#NRW Base2vec
#poi_emb = torch.load('/home/iailab41/sheikhz0/BaseWord2Vec/poi_word_embedding_nrw.tensor')
#NRW Geovectors
#poi_emb = torch.load('/home/iailab41/sheikhz0/GeoVectors/embeddings_merged/poi_embedding_merged_nrw_32.tensor')


#poi_csv = pd.read_csv("/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NRW_updated.csv", encoding='utf-8')
poi_csv = pd.read_csv("/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/POI_ZoneID_NYC.csv", encoding='utf-8')
zone_list = []
zone_ids = []
zone_population = []
poi_zone_grouped = poi_csv.groupby(['ZoneID'])
for zone_id, pois in poi_zone_grouped:
    zone_emb_list = []
    for index, poi in pois.iterrows():
        zone_emb_list.append(list(poi_emb[int(poi['index'])]))
    zone_ids.append(zone_id)
    if zone_id in dataset['ZoneID'].values:
        population = dataset.loc[dataset['ZoneID'] == zone_id, 'population'].values[0]
    else:
        population = 0.0    
    zone_population.append(population)    
    zone_tensor = torch.tensor(zone_emb_list)
    zone_embedding = torch.mean(zone_tensor, dim=0)
    zone_list.append(zone_embedding)

zone_temp_list = torch.stack(zone_list)
zone_final_list = list(zip(zone_temp_list.tolist(), list(zone_population)))

rmse_list = []
r2_list = []
mae_list = []

for iter in range(10):
    print("iter: ", iter)
    np.random.shuffle(zone_final_list)
    x_list = [zone[0] for zone in zone_final_list]
    y_list = [zone[1] for zone in zone_final_list]
    x_train = np.array(x_list[:int(len(x_list) * 0.8)])
    y_train = np.array(y_list[:int(len(y_list) * 0.8)])
    rf = RandomForestRegressor(random_state=iter)
    rf.fit(x_train, y_train)
    x_test = np.array(x_list[int(len(x_list) * 0.8):])
    y_test = np.array(y_list[int(len(y_list) * 0.8):])
    y_pred = rf.predict(x_test)
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    rmse_list.append(rmse)
    print("rmse:", rmse)
    r2 = metrics.r2_score(y_test, y_pred)
    r2_list.append(r2)
    print("r2:", r2)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mae_list.append(mae)
    print("mae:", mae)
average_MAE = np.mean(mae_list)
std_MAE = np.std(mae_list)
average_RMSE = np.mean(rmse_list)
std_RMSE = np.std(rmse_list)
average_R2 = np.mean(r2_list)
std_R2 = np.std(r2_list)
print('=============Result Table=============')
print('MAE\t\tstd\t\tRMSE\t\tstd\t\tR2\t\tstd')
print(f'{average_MAE}\t{std_MAE}\t{average_RMSE}\t{std_RMSE}\t{average_R2}\t{std_R2}')

