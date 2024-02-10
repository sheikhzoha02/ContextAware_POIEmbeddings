import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import Set2Set
import pandas as pd
from scipy.spatial.distance import canberra, chebyshev
import sys


class FunctionZoneDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(FunctionZoneDataset, self).__init__()
        self.data, self.slices = self.collate(data_list)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb_file_name = "/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/poi_embedding_nyc.tensor"
#NYC GeoVectors
#emb_file_name = "/home/iailab41/sheikhz0/GeoVectors/embeddings_merged/poi_embedding_merged_nyc_32.tensor"
#NYC Base2vec
#emb_file_name = "/home/iailab41/sheikhz0/BaseWord2Vec/poi_word_embedding_nyc.tensor"
#NRW Base2vec
#emb_file_name = "/home/iailab41/sheikhz0/BaseWord2Vec/poi_word_embedding_nrw.tensor"
#NRW Geovectors
#emb_file_name = '/home/iailab41/sheikhz0/GeoVectors/embeddings_merged/poi_embedding_merged_nrw_32.tensor'

embedding_size = 64
emb = torch.load(emb_file_name)
#poi_csv = pd.read_csv("/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NRW_updated.csv", encoding='utf-8')
poi_csv = pd.read_csv("/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NYC.csv", encoding='utf-8')
zone_list = []
poi_zone_grouped = poi_csv.groupby(['ZoneID'])

#ground_truth = torch.load("/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/ground_truth_nrw.tensor")
ground_truth = torch.load("/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/ground_truth_nyc.tensor")
for zone_id, pois in poi_zone_grouped:
    second_class_list = []
    second_class_emb_list = []
    for index, poi in pois.iterrows():
        second_class_list.append(poi['index'])
        second_class_emb_list.append(list(emb[int(poi['index'])]))
    zone = Data(x=torch.tensor(second_class_emb_list), y=ground_truth[zone_id], zone_id=zone_id)
    zone_list.append(zone)

function_zone_dataset = FunctionZoneDataset(zone_list)
print(function_zone_dataset)


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


def train():
    model.train()
    loss_all = 0
    p_a = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_estimated = model(data)
        loss = F.kl_div(torch.log(y_estimated.reshape((-1, 10))), data.y.reshape((-1, 10)).float(), 
                        reduction='batchmean').float()
        p_a += (y_estimated - data.y).abs().sum()
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset), p_a / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0
    canberra_distance = 0
    chebyshev_distance = 0
    kl_dist = 0
    cos_dist = 0
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
    return error/len(loader.dataset), canberra_distance/len(loader.dataset), kl_dist/len(loader.dataset), \
           chebyshev_distance/len(loader.dataset), cos_dist/len(loader.dataset)


function_zone_dataset = function_zone_dataset.shuffle()
training_batch_size = 64
test_dataset = function_zone_dataset[:(len(function_zone_dataset) // 10 * 2)]

train_dataset = function_zone_dataset[(len(function_zone_dataset) // 10 * 2):]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(100):
    loss, p_a = train()
    test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist = test(test_loader)

    if (epoch % 5 == 0) or (epoch == 1):
        print('Epoch: {:03d}, p_a: {:7f}, Loss: {:.7f}, '
              'Test MAE: {:.7f}, canberra_dist:{:.7f}, kl_dist:{:.7f}, chebyshev_distance:{:.7f}, cos_distance:{:.7f}'.
              format(epoch, p_a, loss, test_error, canberra_dist, kl_dist, chebyshev_distance, cos_dist))


