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
from Module.set_transformer import PMA
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pytorch_warmup as warmup



writer = SummaryWriter()
edge_weights_file_distance = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_distance_edge_weights.csv'
edge_weights_file_category = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_category_edge_weights_similarity.csv'
poi_file = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings_5_word2vec.csv'
EPS = 1e-15

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(input_dim, num_heads)

    def forward(self, input):
        input = input.unsqueeze(0)  
        output, _ = self.attention(input, input, input)
        output = output.squeeze(0) 
        return output 


def create_weight_adj_matrix(edge_file):
    df = pd.read_csv(edge_file)
    sources = df['source'].values
    targets = df['target'].values
    edge_weights_array = df['weight'].values
    edge_weights_array = edge_weights_array.astype(np.float32)
    weights = edge_weights_array 
    #normalize_edge_weights(edge_weights_array)
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
    data_poi = pd.read_csv(poi_file)
    loaded_tags_embeddings = torch.load("/home/iailab41/sheikhz0/GeoVectors/embeddings_tags/poi_tags_embeddings_nyc_300.tensor")
    node_features = torch.tensor(loaded_tags_embeddings, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
    return data

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
num_heads = 4
fused_embedding_size = 364
margin=1.0
dropout_rate = 0.2
max_norm = 0.9

#graph reconstruction
model_graph_reconstruction = GraphAutoencoder(input_dim, hidden_dim, output_dim, dropout_rate)
edge_index = extract_edge_index(edge_weights_file_category)
data_edge = pd.read_csv(edge_weights_file_category)
edge_weights_array = data_edge['weight'].values
edge_weights_array = edge_weights_array.astype(np.float32)
edge_weight = torch.tensor(edge_weights_array).to(torch.float32)
data_graph_reconstruction = generate_data_model(edge_index,edge_weight,poi_file)

#contrastive learning
data_poi = pd.read_csv(poi_file)
data_list = [ast.literal_eval(s) for s in data_poi['geohash_embedding'].values]
numpy_data = np.array(data_list, dtype=np.float32)
node_features = torch.tensor(numpy_data)
desired_dimensions = 64
num_features = node_features.shape[1]
if num_features < desired_dimensions:
    padding = np.zeros((node_features.shape[0], desired_dimensions - num_features))
    node_features = np.concatenate((node_features, padding), axis=1)
    node_features = torch.tensor(node_features, dtype=torch.float)


edge_index = extract_edge_index(edge_weights_file_distance)
edge_weight = extract_normalize_weights(edge_weights_file_distance)

model_contrastive_learning = TripletNetwork(64, 32, 64)
triplet_loss_fn = TripletLoss(margin)

num_chunks = 3
chunked_tensors = torch.chunk(node_features, num_chunks, dim=0)
x_anchor, x_positive, x_negative = chunked_tensors
source_node_start = 0
target_node_start = 0
source_node_limit = 13895
target_node_limit = 13895
filtered_edges_mask = ((edge_index[0] >= source_node_start) & (edge_index[0] <= source_node_limit) &
                       (edge_index[1] >= target_node_start) & (edge_index[1] <= target_node_limit))

filtered_edge_index_anchor = edge_index[:, filtered_edges_mask]
filtered_edge_weights_anchor = edge_weight[filtered_edges_mask]

anchor_data = Data(x=x_anchor, edge_index=filtered_edge_index_anchor, edge_attr=filtered_edge_weights_anchor)

source_node_start = 13896
target_node_start = 13896
source_node_limit = 27791
target_node_limit = 27791
filtered_edges_mask = ((edge_index[0] >= source_node_start) & (edge_index[0] <= source_node_limit) &
                       (edge_index[1] >= target_node_start) & (edge_index[1] <= target_node_limit))

filtered_edge_index_positive = edge_index[:, filtered_edges_mask]
filtered_edge_weights_positive = edge_weight[filtered_edges_mask]
filtered_edge_index_positive = filtered_edge_index_positive - 13896
positive_data = Data(x=x_positive, edge_index=filtered_edge_index_positive, edge_attr=filtered_edge_weights_positive)

source_node_start = 27792
target_node_start = 27792
source_node_limit = 41688
target_node_limit = 41688
filtered_edges_mask = ((edge_index[0] >= source_node_start) & (edge_index[0] <= source_node_limit) &
                       (edge_index[1] >= target_node_start) & (edge_index[1] <= target_node_limit))

filtered_edge_index_negative = edge_index[:, filtered_edges_mask]
filtered_edge_weights_negative = edge_weight[filtered_edges_mask]
filtered_edge_index_negative = filtered_edge_index_negative - 27792

negative_data = Data(x=x_negative, edge_index=filtered_edge_index_negative, edge_attr=filtered_edge_weights_negative)
optimizer = optim.Adam(list(model_graph_reconstruction.parameters()) + list(model_contrastive_learning.parameters()), lr=0.006)

num_epochs = 2000
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    optimizer.zero_grad()

    #contrastive loss
    anchor_embedding, positive_embedding, negative_embedding = model_contrastive_learning(anchor_data, positive_data, negative_data)
    c_embedding = torch.cat([anchor_embedding, positive_embedding, negative_embedding])
    loss_triplet = triplet_loss_fn(anchor_embedding, positive_embedding, negative_embedding)
    
    #graph reconstruction
    g_embedding = model_graph_reconstruction(data_graph_reconstruction.x, data_graph_reconstruction.edge_index, data_graph_reconstruction.edge_weight)
    merged_embedding = merge_embeddings(g_embedding, c_embedding)
    self_attention = SelfAttention(fused_embedding_size, num_heads)
    output = self_attention(merged_embedding)
    linear_layer = nn.Linear(merged_embedding.shape[1], output_dim)
    final_node_embeddings = linear_layer(output)
    loss = recon_loss(final_node_embeddings, data_graph_reconstruction.edge_index)
    total_loss = loss + loss_triplet

    print('triplet loss')
    print(loss_triplet.item())
    print('recon loss')
    print(loss.item())

    writer.add_scalar('Loss/Contrastive', loss_triplet, epoch)
    writer.add_scalar('Loss/Reconstruction', loss.item(), epoch)
    writer.add_scalar('Loss/CombinedLoss', total_loss, epoch)

    total_loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item()}")


optimized_embeddings = model_graph_reconstruction(data_graph_reconstruction.x, data_graph_reconstruction.edge_index, data_graph_reconstruction.edge_weight)
torch.save(optimized_embeddings, "poi_embedding_tags_update_1.tensor")
