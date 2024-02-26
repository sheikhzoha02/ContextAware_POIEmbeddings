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


edge_weights_file_distance = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_distance_edge_weights.csv'
edge_weights_file_category = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_category_edge_weights.csv'
poi_file = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/output_with_embeddings_1.csv'


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


def get_geohash(latitude, longitude, precision=12):
    return geohash2.encode(latitude, longitude, precision=precision)

def char2vec_conversion(words):
    c2v_model = chars2vec.load_model('eng_50')
    word_embeddings = c2v_model.vectorize_words(words)
    return word_embeddings

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
        data_list_fclass = [ast.literal_eval(s) for s in data_poi['fclass_embedding'].values]
        numpy_data_fclass = np.array(data_list_fclass, dtype=np.float32)
        data_list_1stlevel = [ast.literal_eval(s) for s in data_poi['1stlevel_embedding'].values]
        numpy_data_1stlevel = np.array(data_list_1stlevel, dtype=np.float32)
        numpy_data = np.concatenate((numpy_data_fclass, numpy_data_1stlevel), axis=1)
        node_features = torch.tensor(numpy_data, dtype=torch.float)
    elif graph_type == 'distance':
        data_list = [ast.literal_eval(s) for s in data_poi['geohash_embedding'].values]
        numpy_data = np.array(data_list, dtype=np.float32)
        node_features = torch.tensor(numpy_data)

    desired_dimensions = 64
    num_features = node_features.shape[1]
    if num_features < desired_dimensions:
        padding = np.zeros((node_features.shape[0], desired_dimensions - num_features))
        node_features = np.concatenate((node_features, padding), axis=1)
        node_features = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
    return data

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2, dropout=0.2):
        super(GATLayer, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index, edge_weight):
        mask = x != 0.0
        x_masked = x * mask.float()        
        x_out = self.conv(x_masked, edge_index, edge_weight)
        return x_out


class TripletContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, temperature=0.07, margin=0.5, heads=2):
        super(TripletContrastiveModel, self).__init__()
        self.gat_layer = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.projector = nn.Linear(heads * hidden_dim, output_dim)
        self.temperature = temperature
        self.margin = margin

    def forward(self, data):
        mask = data.x != 0.0000
        x_masked = data.x * mask.float()
        x = self.gat_layer(x_masked, data.edge_index, data.edge_weight)
        x = F.relu(x)
        x = self.projector(x.view(-1, self.gat_layer.out_channels * self.gat_layer.heads))
        return F.normalize(x, dim=1)

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2, heads=2):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)  # For the output layer, use heads=1


    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index, edge_weight)
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
input_dim = 64
hidden_dim = 64
output_dim = 64
num_heads = 2
fused_embedding_size = 128
temperature=0.07
margin=0.5
dropout_rate = 0.2

#adjacency matrix of edge weights
#adj_edge_weights = create_weight_adj_matrix(edge_weights_file_category)

#graph reconstruction
model_graph_reconstruction = GraphAutoencoder(input_dim, hidden_dim, output_dim, dropout_rate)
edge_index = extract_edge_index(edge_weights_file_category)
data_edge = pd.read_csv(edge_weights_file_category)
edge_weights_array = data_edge['weight'].values
edge_weights_array = edge_weights_array.astype(np.float32)
edge_weight = torch.tensor(edge_weights_array).to(torch.float32)
graph_type = 'category'
data_graph_reconstruction = generate_data_model(edge_index,edge_weight,poi_file,graph_type)

#contrastive learning
model_contrastive_learning = TripletContrastiveModel(input_dim, hidden_dim, output_dim, dropout_rate, temperature, margin, num_heads)
optimizer = torch.optim.Adam(list(model_graph_reconstruction.parameters()) + list(model_contrastive_learning.parameters()), lr=0.001)
edge_index = extract_edge_index(edge_weights_file_distance)
edge_weight = extract_normalize_weights(edge_weights_file_distance)
graph_type = 'distance'
data_contrastive_learning = generate_data_model(edge_index,edge_weight,poi_file,graph_type)
num_epochs = 100
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
    self_attention = SelfAttention(fused_embedding_size, num_heads)
    output = self_attention(merged_embedding)
    linear_layer = nn.Linear(output.shape[1], output_dim)
    final_node_embeddings = linear_layer(output)
    loss = recon_loss(final_node_embeddings, data_graph_reconstruction.edge_index)
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
