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

#edge_weights_file_distance = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nyc_distance_edge_weights.csv'
#edge_weights_file_category = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nyc_category_edge_weights.csv'
#poi_file = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NYC.csv'


edge_weights_file_distance = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nrw_distance_edge_weights.csv'
edge_weights_file_category = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/new_data/nrw_category_edge_weights.csv'
poi_file = '/home/iailab41/sheikhz0/Semantics-preserved-POI-embedding/Data/POI_ZoneID_NRW_updated.csv'

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


def generate_data_model(edge_index,edge_weight,poi_file,graph_type):
    data_poi = pd.read_csv(poi_file)
    if graph_type == 'category':
        features = ['index', 'FirstLevel','SecondLeve']
    elif graph_type == 'distance':
        features = ['index','Lon','Lat']

    #features normalization
    node_features = torch.tensor(data_poi[features].values).to(torch.float32)
    mean_vals = node_features.mean(dim=0)
    std_vals = node_features.std(dim=0)
    normalized_features_category_graph = (node_features - mean_vals) / std_vals

    node_features_np = normalized_features_category_graph
    num_features = node_features_np.shape[1]
    desired_dimensions = 64
    if num_features < desired_dimensions:
        padding = np.zeros((node_features_np.shape[0], desired_dimensions - num_features))
        node_features_np = np.concatenate((node_features_np, padding), axis=1)

    node_features = torch.tensor(node_features_np, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
    return data

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        return x


class TripletContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, temperature=0.07, margin=0.5):
        super(TripletContrastiveModel, self).__init__()
        self.gat_layer = GATLayer(input_dim, hidden_dim)
        self.projector = nn.Linear(hidden_dim, output_dim)
        self.temperature = temperature
        self.margin = margin

    def forward(self, data):
        x = self.gat_layer(data.x, data.edge_index, data.edge_weight)
        x = F.relu(x)
        x = self.projector(x)
        return F.normalize(x, dim=1)

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def graph_reconstruction_loss(reconstructed_x, x, adj_matrix, edge_weights):
    adj_matrix = adj_matrix.to(torch.float32)
    cos_sim = cosine_similarity(reconstructed_x.detach().numpy())
    threshold = 0.7
#    reconstructed_adj_matrix = reconstructed_x.matmul(reconstructed_x.t())
    reconstructed_adj_matrix = torch.tensor(cos_sim > threshold, dtype=torch.float32)
    reconstructed_adj_matrix = reconstructed_adj_matrix * edge_weights
    adjacency_loss = F.mse_loss(reconstructed_adj_matrix, adj_matrix)
    feature_loss = F.mse_loss(reconstructed_x, x)
    return adjacency_loss + feature_loss


def contrastive_loss(embeddings, temperature=0.07):
    batch_size = embeddings.size(0)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    sim_matrix = sim_matrix - torch.eye(batch_size).to(embeddings.device) * 1e9
    
    positive_pairs = torch.diag(sim_matrix, diagonal=1)
    negative_pairs = torch.logsumexp(sim_matrix, dim=1)
    combined_pairs = torch.cat((positive_pairs, negative_pairs), dim=0)
    loss = F.cross_entropy(combined_pairs[2:],
                           torch.zeros(batch_size).long().to(embeddings.device).type(torch.FloatTensor) )
    return loss


def get_triplets(embeddings):
    num_samples = embeddings.shape[0]
    indices = torch.randint(0, num_samples, (3,))
    anchor, positive, negative = embeddings[indices[0]], embeddings[indices[1]], embeddings[indices[2]]
    return anchor, positive, negative

def triplet_loss(anchor, positive, negative, margin=0.5):
    distance_positive = F.pairwise_distance(anchor, positive)
    distance_negative = F.pairwise_distance(anchor, negative)
    loss = F.relu(distance_positive - distance_negative + margin)
    return torch.mean(loss)


def merge_embeddings(gae_embedding, other_embedding):
    return torch.cat((gae_embedding, other_embedding), dim=1)

#parameters
input_dim = 64
hidden_dim = 64
output_dim = 64
#num_heads = 2
fused_embedding_size = 64

#adjacency matrix of edge weights
#adj_edge_weights = create_weight_adj_matrix(edge_weights_file_category)

#graph reconstruction
model_graph_reconstruction = GraphAutoencoder(input_dim, hidden_dim, output_dim)
edge_index = extract_edge_index(edge_weights_file_category)
#edge_weight = extract_normalize_weights(edge_weights_file_category)
data_edge = pd.read_csv(edge_weights_file_category)
edge_weights_array = data_edge['weight'].values
edge_weights_array = edge_weights_array.astype(np.float32)
edge_weight = torch.tensor(edge_weights_array).to(torch.float32)
graph_type = 'category'
data_graph_reconstruction = generate_data_model(edge_index,edge_weight,poi_file,graph_type)

#contrastive learning
model_contrastive_learning = TripletContrastiveModel(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(list(model_graph_reconstruction.parameters()) + list(model_contrastive_learning.parameters()), lr=0.001)
edge_index = extract_edge_index(edge_weights_file_distance)
edge_weight = extract_normalize_weights(edge_weights_file_distance)
graph_type = 'distance'
data_contrastive_learning = generate_data_model(edge_index,edge_weight,poi_file,graph_type)
num_epochs = 1000
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    #losses = []

    optimizer.zero_grad()

    #contrastive loss
    c_embeddings = model_contrastive_learning(data_contrastive_learning)
    anchor, positive, negative = get_triplets(c_embeddings)
    loss_triplet = triplet_loss(anchor, positive, negative)
    poi_embeddings = torch.stack([anchor, positive, negative])
    loss_contrastive = contrastive_loss(poi_embeddings)
    combined_loss = loss_triplet + loss_contrastive
    
    #graph reconstruction
    g_embedding = model_graph_reconstruction(data_graph_reconstruction.x, data_graph_reconstruction.edge_index, data_graph_reconstruction.edge_weight)
    merged_embedding = merge_embeddings(g_embedding, c_embeddings)
    linear_layer = nn.Linear(g_embedding.shape[1] + c_embeddings.shape[1], fused_embedding_size)
    fused_embedding = linear_layer(merged_embedding)
    adjacency_matrix = to_dense_adj(data_graph_reconstruction.edge_index).squeeze()
    cos_sim = cosine_similarity(fused_embedding.detach().numpy())
    threshold = 0.7
    reconstructed_adj_matrix = torch.tensor(cos_sim > threshold, dtype=torch.float32)
    #reconstructed_adjacency = torch.matmul(merged_embedding, merged_embedding.t())
    loss = criterion(reconstructed_adj_matrix, adjacency_matrix)
    total_loss = loss + combined_loss

    #total loss backward
    total_loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss}")

#change the fusion strategy
#change the min max normalization of distances
#change the adjacency matrix to list (scalabilty)
#use more features name,rating, opening hours
optimized_embeddings = model_graph_reconstruction(data_graph_reconstruction.x, data_graph_reconstruction.edge_index, data_graph_reconstruction.edge_weight)
torch.save(optimized_embeddings, "poi_embedding_nyc.tensor")