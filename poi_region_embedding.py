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
import pickle


class POI2Region(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super(POI2Region, self).__init__()
        self.PMA = PMA(dim=hidden_channels, num_heads=num_heads, num_seeds=1, ln=False)
        self.conv = GCNConv(hidden_channels, hidden_channels, cached=True, bias=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, zone, region_adjacency):
        region_emb = x.new_zeros((zone.max()+1, x.size()[1]))
        for index in range(zone.max() + 1):
            poi_index_in_region = (zone == index).nonzero(as_tuple=True)[0]
            region_emb[index] = self.PMA(x[poi_index_in_region].unsqueeze(0)).squeeze()
        region_emb = self.conv(region_emb, region_adjacency)
        region_emb = self.prelu(region_emb)
        return region_emb


poi2region=POI2Region(64, 4)
pos_poi_emb = torch.load('/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/poi_embedding_nyc_2000_TCL_final_no_scheduler_no_weight.tensor')
region_emb_to_save = torch.FloatTensor(0)
pickle_file_path = '/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/Data/NYC/nyc_data.pkl'
with open(pickle_file_path, 'rb') as file:
    city_dict = pickle.load(file)
region_id = torch.tensor(city_dict['region_id'], dtype=torch.int64)
region_adjacency = torch.tensor(city_dict['region_adjacency'], dtype=torch.int64)
region_emb_to_save = poi2region(pos_poi_emb, region_id, region_adjacency)
torch.save(region_emb_to_save, f'/home/iailab41/sheikhz0/POI-Embeddings-Own-Approach/region_embedding_new_param_TCL_2000_with_no_weight')