"""
@description: this is part of the Function Distribution Learning proposed in the paper
"Estimating urban functional distributions with semantics preserved POI embedding" (https://www.semanticscholar.org/paper/Estimating-urban-functional-distributions-with-POI-Huang-Cui/14e56189b19a385d8cd73eae145098d6ddaf9070)
The code is modified from the original implementation (https://github.com/RightBank/Semantics-preserved-POI-embedding)
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import Set2Set
import pandas as pd
from scipy.spatial.distance import canberra, chebyshev
import sys
import numpy as np
import random
import csv
import argparse

#python Downstream_tasks/functional_distribution_learning.py  --city köln

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FunctionZoneDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(FunctionZoneDataset, self).__init__()
        self.data, self.slices = self.collate(data_list)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.set2set = Set2Set(embedding_size, processing_steps=5)
        self.lin0 = torch.nn.Linear(embedding_size*2, embedding_size)
        self.lin1 = torch.nn.Linear(embedding_size, 10)

    def forward(self, data):
        out = self.set2set(data.x, data.batch).float()
        out = torch.tanh(self.lin0(out)).float()
        out = F.softmax(self.lin1(out), -1).float()
        return out.view(-1).float()


#set the parser
parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, help='Name of the city')
args = parser.parse_args()


if args.city == 'köln':
    ground_truth = torch.load('Data/ground_truth/köln/ground_truth_köln_fd.tensor')
    poi_csv = pd.read_csv('Data/NRW/Köln/poi_köln_zoneid.csv')
    emb = torch.load(f"{args.city}_embeddings.tensor")
elif args.city == 'düsseldorf':
    ground_truth = torch.load('Data/ground_truth/dusseldorf/ground_truth_dusseldorf_fd.tensor')
    poi_csv = pd.read_csv('Data/NRW/Düsseldorf/poi_dusseldorf_zoneid.csv')
    emb = torch.load(f"{args.city}_embeddings.tensor")
elif args.city == 'nyc':
    ground_truth = torch.load('Data/ground_truth/nyc/ground_truth_nyc_fd.tensor')
    poi_csv = pd.read_csv('Data/NYC/poi_zoneid_nyc.csv')
    emb = torch.load(f"{args.city}_embeddings.tensor")

embedding_size = emb.shape[1]
emb = torch.tensor(emb)

zone_list = []
poi_zone_grouped = poi_csv.groupby(['ZoneID'])

for zone_id, pois in poi_zone_grouped:
    second_class_list = []
    second_class_emb_list = []
    for index, poi in pois.iterrows():
        second_class_list.append(poi['index'])
        second_class_emb_list.append(list(emb[int(poi['index'])]))
    zone = Data(x=torch.tensor(second_class_emb_list), y=ground_truth[zone_id], zone_id=zone_id)
    zone_list.append(zone)

function_zone_dataset = FunctionZoneDataset(zone_list)

def train():
    model.train()
    loss_all = 0
    p_a = 0
    actual_distributions = []
    estimated_distributions = []
    zone_ids = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_estimated = model(data)
        loss = F.kl_div(torch.log(y_estimated.reshape((-1, 10))), data.y.reshape((-1, 10)).float(), 
                        reduction='batchmean').float()
        p_a += (y_estimated - data.y).abs().sum()
        actual_distributions.append(data.y.cpu().detach().numpy().reshape((-1, 10)).tolist())
        estimated_distributions.append(y_estimated.cpu().detach().numpy().reshape((-1, 10)).tolist())
        zone_ids.append(data.zone_id.cpu().detach().numpy().tolist())
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset), p_a / len(train_loader.dataset), actual_distributions, estimated_distributions, zone_ids


def test(loader):
    model.eval()
    error = 0
    canberra_distance = 0
    chebyshev_distance = 0
    kl_dist = 0
    cos_dist = 0
    actual_distributions = []
    estimated_distributions = []
    zone_ids = []
    for data in loader:
        data = data.to(device)
        y_estimated = model(data)
        error += ((y_estimated - data.y).abs().sum())
        canberra_distance += canberra(y_estimated.cpu().detach().numpy(), data.y.cpu().detach().numpy())
        kl_dist += F.kl_div(torch.log(y_estimated.reshape((-1, 10))), data.y.reshape((-1, 10)).float(),
                            reduction='batchmean').float()
        chebyshev_distance += chebyshev(y_estimated.cpu().detach().numpy(), data.y.cpu().detach().numpy())
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dist += cos(y_estimated, data.y)

        actual_distributions.append(data.y.cpu().detach().numpy().reshape((-1, 10)).tolist())
        estimated_distributions.append(y_estimated.cpu().detach().numpy().reshape((-1, 10)).tolist())
        zone_ids.append(data.zone_id.cpu().detach().numpy().tolist())

    return error/len(loader.dataset), canberra_distance/len(loader.dataset), kl_dist/len(loader.dataset), \
           chebyshev_distance/len(loader.dataset), cos_dist/len(loader.dataset), actual_distributions, estimated_distributions, zone_ids


function_zone_dataset = function_zone_dataset.shuffle()
training_batch_size = 64
test_dataset = function_zone_dataset[:(len(function_zone_dataset) // 10 * 2)]
train_dataset = function_zone_dataset[(len(function_zone_dataset) // 10 * 2):]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

kl_div, l1, cos = [], [], []
all_actual_distributions = []
all_estimated_distributions = []
all_zone_ids = []

for epoch in range(100):
    loss, p_a, actual_distributions_train, estimated_distributions_train, zone_ids_train = train()
    test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist, actual_distributions_test, estimated_distributions_test, zone_ids_test = test(test_loader)
    kl_div.append(kl_dist.item())
    l1.append(test_error.item())
    cos.append(cos_dist.item())

    if (epoch % 5 == 0) or (epoch == 1):
        print('Epoch: {:03d}, p_a: {:7f}, Loss: {:.7f}, '
              'Test MAE: {:.7f}, canberra_dist:{:.7f}, kl_dist:{:.7f}, chebyshev_distance:{:.7f}, cos_distance:{:.7f}'.
              format(epoch, p_a, loss, test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist))

    if epoch == 99:
        all_actual_distributions.extend(actual_distributions_train + actual_distributions_test)
        all_estimated_distributions.extend(estimated_distributions_train + estimated_distributions_test)
        all_zone_ids.extend(zone_ids_train + zone_ids_test)

#save estimated and actual distributions in csv
with open(f"{args.city}_distrbution_train.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Zone ID", "Actual Distribution", "Estimated Distribution"])
    for batch_index, (batch_zone_ids, batch_actual_distributions, batch_estimated_distributions) in enumerate(zip(zone_ids_train, actual_distributions_train, estimated_distributions_train)):
        for zone_id, actual, estimated in zip(batch_zone_ids, batch_actual_distributions, batch_estimated_distributions):
            actual_str = ", ".join(map(str, actual))
            estimated_str = ", ".join(map(str, estimated))
            writer.writerow([zone_id, actual_str, estimated_str])

with open(f"{args.city}_distrbution_test.csv", "a", newline='') as file:
    writer = csv.writer(file)
    for batch_index, (batch_zone_ids, batch_actual_distributions, batch_estimated_distributions) in enumerate(zip(zone_ids_test, actual_distributions_test, estimated_distributions_test)):
        for zone_id, actual, estimated in zip(batch_zone_ids, batch_actual_distributions, batch_estimated_distributions):
            actual_str = ", ".join(map(str, actual))
            estimated_str = ", ".join(map(str, estimated))
            writer.writerow([zone_id, actual_str, estimated_str])


print('Result for ')
print('=============Result Table=============')
print('L1\tstd\tKL-Div\tstd\tCosine\tstd')
print('{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}'.format(np.mean(l1), np.std(l1), np.mean(kl_div), np.std(kl_div),
                                                                np.mean(cos), np.std(cos)))