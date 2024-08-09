import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import to_dense_adj
from sklearn.metrics.pairwise import cosine_similarity
import sys
import ast
from torch import Tensor
from typing import Optional
from Module.InnerProductDecoderClass import InnerProductDecoder
from torch_geometric.utils import negative_sampling
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pytorch_warmup as warmup
import torch
import math
import argparse


#python poi_embedding.py --city köln

#set the logger
writer = SummaryWriter()

#set the parser
parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, help='Name of the city')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.city == 'köln':
    edge_weights_file_distance = 'Data/NRW/Köln/köln_distance_edge_weights_clean.csv'
    edge_weights_file_category = 'Data/NRW/Köln/köln_similarity_edge_weights_clean.csv'
    poi_file = 'Data/NRW/Köln/geohash_embedding_köln_frequency.csv'
    tags_tensor = 'Data/NRW/Köln/poi_tags_embeddings_köln_300.tensor'
elif args.city == 'düsseldorf':
    edge_weights_file_distance = 'Data/NRW/Düsseldorf/dusseldorf_distance_edge_weights_clean.csv'
    edge_weights_file_category = 'Data/NRW/Düsseldorf/dusseldorf_similarity_edge_weights_clean.csv'
    poi_file = 'Data/NRW/Düsseldorf/geohash_embedding_dusseldorf_frequency.csv'
    tags_tensor = 'Data/NRW/Düsseldorf/poi_tags_embeddings_dusseldorf_300.tensor'
elif args.city == 'nyc':
    edge_weights_file_distance = 'Data/NYC/nyc_distance_edge_weights_clean.csv'
    edge_weights_file_category = 'Data/NYC/nyc_similarity_edge_weights_clean.csv'
    poi_file = 'Data/NYC/geohash_embedding_nyc_frequency.csv'
    tags_tensor = 'Data/NYC/poi_tags_embeddings_nyc_300.tensor'

def create_weight_adj_matrix(edge_file):
    df = pd.read_csv(edge_file)
    sources = df['source'].values
    targets = df['target'].values
    edge_weights_array = df['weight'].values
    edge_weights_array = edge_weights_array.astype(np.float32)
    weights = edge_weights_array 
    num_nodes = max(np.max(sources), np.max(targets)) + 1
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(len(weights)):
        adjacency_matrix[sources[i]][targets[i]] = weights[i]
    return torch.tensor(adjacency_matrix)

def extract_edge_index(edge_file):
    data_edge = pd.read_csv(edge_file)
    source_array = data_edge['source'].values
    target_array = data_edge['target'].values

    source_target_pairs = list(zip(source_array, target_array))
    edges = source_target_pairs
    num_edges = len(edges)
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    for idx, (src, tgt) in enumerate(edges):
        edge_index[0, idx] = src
        edge_index[1, idx] = tgt
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def extract_normalize_weights(edge_file):
    data_edge = pd.read_csv(edge_file)
    edge_weights_array = data_edge['weight'].values
    edge_weights_array = edge_weights_array.astype(np.float32)
    extracted_edge_weights = normalize_edge_weights(edge_weights_array)
    return extracted_edge_weights

def normalize_edge_weights(edge_weights_array):
    data_weights = np.array(edge_weights_array)
    min_val = np.min(data_weights)
    max_val = np.max(data_weights)
    normalized_data = (data_weights - min_val) / (max_val - min_val)
    edge_weights_normalized = 1 - normalized_data
    edge_weight = torch.tensor(edge_weights_normalized, dtype=torch.float)
    return edge_weight


def generate_data_model(edge_index,edge_weight,poi_file):
    loaded_tags_embeddings = torch.load(tags_tensor)
    node_features = torch.tensor(loaded_tags_embeddings, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
    return data


def get_indexes_chunked_tensors (anchor_data, positive_data, negative_data, index, batch_size, edge_index, edge_weight, max_size, start_index):
    if start_index != 0:
        start_idx = start_index
    else:    
        start_idx = index * batch_size
    end_idx = min((index + 1) * batch_size, max_size)

    anchor_length = len(anchor_data)
    positive_length = len(positive_data)
    negative_length = len(negative_data)

    result = [] 
    source_start = start_idx
    target_start = start_idx
    source_end = source_start + anchor_length - 1
    target_end = source_start + anchor_length - 1

    last_index = target_end

    filtered_edges_mask = ((edge_index[0] >= source_start) &
                            (edge_index[0] <= source_end) &
                            (edge_index[1] >= target_start) &
                            (edge_index[1] <= target_end))

    filtered_edge_index = edge_index[:, filtered_edges_mask]
    filtered_edge_weights = edge_weight[filtered_edges_mask]
    filtered_edge_index = filtered_edge_index - source_start

    result.append(Data(x=anchor_data, edge_index=filtered_edge_index, edge_attr=filtered_edge_weights))
    
    source_start = last_index + 1
    target_start = last_index + 1
    source_end = source_start + positive_length - 1
    target_end = source_start + positive_length - 1

    last_index =  source_start + positive_length - 1

    filtered_edges_mask = ((edge_index[0] >= source_start) &
                            (edge_index[0] <= source_end) &
                            (edge_index[1] >= target_start) &
                            (edge_index[1] <= target_end))

    filtered_edge_index = edge_index[:, filtered_edges_mask]
    filtered_edge_weights = edge_weight[filtered_edges_mask]
    filtered_edge_index = filtered_edge_index - source_start

    result.append(Data(x=positive_data, edge_index=filtered_edge_index, edge_attr=filtered_edge_weights))

    source_start = last_index + 1
    target_start = last_index + 1
    source_end = source_start + negative_length - 1
    target_end = source_start + negative_length - 1

    filtered_edges_mask = ((edge_index[0] >= source_start) &
                            (edge_index[0] <= source_end) &
                            (edge_index[1] >= target_start) &
                            (edge_index[1] <= target_end))

    filtered_edge_index = edge_index[:, filtered_edges_mask]
    filtered_edge_weights = edge_weight[filtered_edges_mask]
    filtered_edge_index = filtered_edge_index - source_start
    result.append(Data(x=negative_data, edge_index=filtered_edge_index, edge_attr=filtered_edge_weights))

    return result

def get_indexes_graph (data_graph, index, batch_size, edge_index, edge_weight, max_size, start_index):
    if start_index != 0:
        start_idx = start_index
    else:    
        start_idx = index * batch_size
    end_idx = min((index + 1) * batch_size, max_size)

    graph_data_length = len(data_graph)
    source_start = start_idx
    target_start = start_idx
    source_end = source_start + graph_data_length - 1
    target_end = source_start + graph_data_length - 1

    last_index = target_end

    filtered_edges_mask = ((edge_index[0] >= source_start) &
                            (edge_index[0] <= source_end) &
                            (edge_index[1] >= target_start) &
                            (edge_index[1] <= target_end))

    filtered_edge_index = edge_index[:, filtered_edges_mask]
    filtered_edge_weights = edge_weight[filtered_edges_mask]
    filtered_edge_index = filtered_edge_index - source_start
    return filtered_edge_index, filtered_edge_weights



class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TripletGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(TripletGNN, self).__init__()
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.gcn1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index, edge_weight=edge_weight)
        return x


class TripletNetwork(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(TripletNetwork, self).__init__()
        self.gnn = TripletGNN(num_features, hidden_dim, output_dim)

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.gnn(anchor.x, anchor.edge_index, edge_weight=anchor.edge_attr)
        positive_embedding = self.gnn(positive.x, positive.edge_index, edge_weight=positive.edge_attr)
        negative_embedding = self.gnn(negative.x, negative.edge_index, edge_weight=negative.edge_attr)
        return anchor_embedding, positive_embedding, negative_embedding

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        if len(negative) != len(anchor):
            pad_size = len(anchor) - len(negative)
            padding = torch.zeros((pad_size,) + negative.shape[1:], dtype=negative.dtype)
            negative = torch.cat((negative, padding.to(device)), dim=0)
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(loss)


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(p=dropout)


    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.dropout2(x)
        return x


def get_triplets(embeddings):
    num_samples = embeddings.shape[0]
    indices = torch.randint(0, num_samples, (3,))
    anchor, positive, negative = embeddings[indices[0]], embeddings[indices[1]], embeddings[indices[2]]
    return anchor, positive, negative


def merge_embeddings(gae_embedding, other_embedding):
    return torch.cat((gae_embedding, other_embedding), dim=1)


def recon_loss(z: Tensor, pos_edge_index: Tensor,
                neg_edge_index: Optional[Tensor] = None) -> Tensor:
    decoder = InnerProductDecoder()
    pos_pred = decoder(z, pos_edge_index)
    pos_target = torch.ones_like(pos_pred)
    pos_loss = F.mse_loss(pos_pred, pos_target)

    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

    neg_pred = decoder(z, neg_edge_index)
    neg_target = torch.zeros_like(neg_pred)
    neg_loss = F.mse_loss(neg_pred, neg_target)

    return pos_loss + neg_loss


#parameters
input_dim = 300
hidden_dim = 150
output_dim = 300
input_dim_t = 100
hidden_dim_t = 50
output_dim_t = 100
input_dim_f = 400
hidden_dim_f = 200
output_dim_f = 400
num_heads = 4
fused_embedding_size = 400
margin=1.0
dropout_rate = 0.2
max_norm = 0.9
batch_size = 64


# Early stopping parameters
patience = 5
min_delta = 0.001
lowest_loss = math.inf
num_epochs = 1000

#graph reconstruction
model_graph_reconstruction = GraphAutoencoder(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
edge_index = extract_edge_index(edge_weights_file_category)
data_edge = pd.read_csv(edge_weights_file_category)
edge_weights_array = data_edge['weight'].values
edge_weights_array = edge_weights_array.astype(np.float32)
edge_weight = torch.tensor(edge_weights_array).to(torch.float32)
data_graph_reconstruction = generate_data_model(edge_index,edge_weight,poi_file)

#contrastive learning
#contrastive learning
data_poi = pd.read_csv(poi_file)
data_list = [ast.literal_eval(s) for s in data_poi['geohash_embedding'].values]
numpy_data = np.array(data_list, dtype=np.float32)
node_features = torch.tensor(numpy_data)
desired_dimensions = 100
scaler = MinMaxScaler()
normalized_data = torch.empty_like(node_features)
for i in range(node_features.size(1)):
    scaler.fit(node_features[:, i].numpy().reshape(-1, 1))
    normalized_data[:, i] = torch.tensor(scaler.transform(node_features[:, i].numpy().reshape(-1, 1)).flatten())

num_features = normalized_data.shape[1]
if num_features < desired_dimensions:
    padding = np.zeros((normalized_data.shape[0], desired_dimensions - num_features))
    normalized_data = np.concatenate((normalized_data, padding), axis=1)
    normalized_data = torch.tensor(normalized_data, dtype=torch.float)

#target representations
loaded_tags_embeddings = torch.load(tags_tensor)
tags_features = torch.tensor(loaded_tags_embeddings, dtype=torch.float)
target_representations = merge_embeddings(normalized_data, tags_features)
edge_index_distance = extract_edge_index(edge_weights_file_distance)
edge_weight_distance = extract_normalize_weights(edge_weights_file_distance)

#contrastive learning model
model_contrastive_learning = TripletNetwork(input_dim_t, hidden_dim_t, output_dim_t).to(device)
triplet_loss_fn = TripletLoss(margin)

#dataloader
train_dataset = TensorDataset(normalized_data, data_graph_reconstruction.x, target_representations)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

#fcnn initialization
model_fcnn = FCNN(input_dim_f, hidden_dim_f, output_dim_f).to(device)

#optimizer
optimizer = optim.Adam(
    params=list(model_graph_reconstruction.parameters()) + 
           list(model_contrastive_learning.parameters()) + 
           list(model_fcnn.parameters()),
    lr=0.001
)

criterion = nn.MSELoss()
model_contrastive_learning.train()
model_graph_reconstruction.train()
model_fcnn.train()

for epoch in range(num_epochs):
    #contrastive loss
    start_index = 0
    train_losses = []
    loss_recon_array = []
    loss_fcc_array = []
    loss_triplet_array = []
    for index, (node_feature, data_graph, target_representation) in enumerate(train_loader):
        optimizer.zero_grad()
        chunked_tensors = torch.chunk(node_feature, 3, dim=0)

        #contrastive learning model
        x_anchor_train, x_positive_train, x_negative_train = chunked_tensors
        results = get_indexes_chunked_tensors (x_anchor_train, x_positive_train, x_negative_train, index, batch_size, edge_index_distance, edge_weight_distance, len(node_features), start_index)
        anchor_embedding, positive_embedding, negative_embedding = model_contrastive_learning(results[0].to(device), results[1].to(device), results[2].to(device))
        c_embedding = torch.cat([anchor_embedding, positive_embedding, negative_embedding])
        loss_triplet = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        loss_triplet_array.append(loss_triplet.item())

        #graph reconstruction model
        edge_index, edge_weights = get_indexes_graph (data_graph, index, batch_size, data_graph_reconstruction.edge_index, data_graph_reconstruction.edge_weight, len(data_graph_reconstruction.x), start_index)
        g_embedding = model_graph_reconstruction(data_graph.to(device), edge_index.to(device), edge_weights.to(device))
        loss_recon = recon_loss(g_embedding, edge_index)
        loss_recon_array.append(loss_recon.item())

        merged_embedding = merge_embeddings(g_embedding, c_embedding)

        #fcnn model
        final_representations = model_fcnn(merged_embedding)
        loss_fcc = criterion(final_representations, target_representation.to(device))
        loss_fcc_array.append(loss_fcc.item())

        total_loss = loss_recon + loss_triplet + loss_fcc
        train_losses.append(total_loss.item())

        total_loss.backward()
        optimizer.step()

    avg_train_loss = np.round(np.mean(train_losses), 5)
    avg_triplet_loss = np.round(np.mean(loss_triplet_array), 5)
    avg_recon_loss = np.round(np.mean(loss_recon_array), 5)
    avg_fcc_loss = np.round(np.mean(loss_fcc_array), 5)
    
    #log losses in a file
#    with open("/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/losses/losses_koeln_batches.csv", "a") as file:
#        file.write(f"{epoch},{avg_train_loss},{avg_triplet_loss},{avg_recon_loss},{avg_fcc_loss}\n")

    if avg_train_loss < lowest_loss - min_delta:
        lowest_loss = avg_train_loss
        patience = 10
    else:
        patience -= 1
        if patience == 0:
            print('early stopping')
            break

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss}")

train_embeddings = []

with torch.no_grad():
    for index, (node_feature, data_graph, target_representation) in enumerate(train_loader):
        chunked_tensors = torch.chunk(node_feature, 3, dim=0)
        x_anchor_train, x_positive_train, x_negative_train = chunked_tensors
        results = get_indexes_chunked_tensors (x_anchor_train, x_positive_train, x_negative_train, index, batch_size, edge_index_distance, edge_weight_distance, len(node_features), start_index)
        anchor_embedding, positive_embedding, negative_embedding = model_contrastive_learning(results[0].to(device), results[1].to(device), results[2].to(device))
        c_embedding = torch.cat([anchor_embedding, positive_embedding, negative_embedding])
        
        edge_index, edge_weights = get_indexes_graph (data_graph, index, batch_size, data_graph_reconstruction.edge_index, data_graph_reconstruction.edge_weight, len(data_graph_reconstruction.x), start_index)
        g_embedding = model_graph_reconstruction(data_graph.to(device), edge_index.to(device), edge_weights.to(device))
        
        merged_embedding = merge_embeddings(g_embedding, c_embedding)
        final_representations = model_fcnn(merged_embedding)
        train_embeddings.append(final_representations)

#save pre-trained embeddings
train_embeddings = torch.cat(train_embeddings, dim=0)
print("Train Embeddings Shape:", train_embeddings.shape)
torch.save(train_embeddings.clone().cpu().detach(), f"{args.city}_embeddings.tensor")
