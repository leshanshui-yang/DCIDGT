# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('...')

!pip install wandb
import wandb

from codes.Component import MyConfig
from codes.Settings import Settings

from argparse import Namespace

import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import copy

def set_seed(seed):
  os.environ['PYTHONHASHSEED']=str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)

## Origin DynAN Model.py Imports
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import networkx as nx

from sklearn import metrics

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPooler, BertAttention, BertIntermediate, BertOutput
from transformers.configuration_utils import PretrainedConfig


BertLayerNorm = torch.nn.LayerNorm
TransformerLayerNorm = torch.nn.LayerNorm

"""# Utils

## DataLoader
"""

## DEBUG
from codes.base_class.dataset import dataset
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
import gc

class DynamicDatasetLoader(dataset):
    c = 0.15
    k = 5
    eps = 0.001
    window_size = 1
    data = None
    batch_size = None
    dataset_name = None
    load_all_tag = False
    compute_s = False
    anomaly_per = 0.1
    train_per = 0.5

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DynamicDatasetLoader, self).__init__(dName, dDescription)

    def load_hop_wl_batch(self):  #load the "raw" WL/Hop/Batch dict
        print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.k) + '_' + str(self.window_size), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k) + '_' + str(self.window_size), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix. (0226)"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation. (0226)"""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. (0226)"""
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj_np = np.array(adj.todense())
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_normalized = self.sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized

    def get_adjs(self, rows, cols, weights, nb_nodes):

        eigen_file_name = 'data/eigen/' + self.dataset_name + '_' + str(self.train_per) + '_' + str(self.anomaly_per) + '_sd' + str(self.seed) + '.pkl'
        if not os.path.exists(eigen_file_name):
            generate_eigen = True
            print('Generating eigen as: ' + eigen_file_name)
        else:
            generate_eigen = False
            print('Loading eigen from: ' + eigen_file_name)
            with open(eigen_file_name, 'rb') as f:
                eigen_adjs_sparse = pickle.load(f)
            eigen_adjs = []
            for eigen_adj_sparse in eigen_adjs_sparse:
                if self.dataset_name == "digg": ####
                    eigen_adjs.append(np.array(eigen_adj_sparse.todense()).astype(np.float16))
                else:
                    eigen_adjs.append(np.array(eigen_adj_sparse.todense()))
            del eigen_adjs_sparse
            collected = gc.collect()
            print(f"Garbage collector: collected {collected} objects.")


        adjs = []
        if generate_eigen:
            eigen_adjs = []
            eigen_adjs_sparse = []

        for i in range(len(rows)):
            adj = sp.csr_matrix((weights[i], (rows[i], cols[i])), shape=(nb_nodes, nb_nodes), dtype=np.float32)
            adjs.append(self.preprocess_adj(adj))
            if self.compute_s:
                if generate_eigen:
                    eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
                    for p in range(adj.shape[0]):
                        eigen_adj[p,p] = 0.
                    eigen_adj = self.normalize(eigen_adj)
                    eigen_adjs.append(eigen_adj)
                    eigen_adjs_sparse.append(sp.csr_matrix(eigen_adj))

            else:
                eigen_adjs.append(None)

        if generate_eigen:
            with open(eigen_file_name, 'wb') as f:
                pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)

        return adjs, eigen_adjs

    def load(self):
        """Load dynamic network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))
        with open('data/percent/' + self.dataset_name + '_' + str(self.train_per) + '_' + str(self.anomaly_per) + '_sd' + str(self.seed) + '.pkl', 'rb') as f:
            rows, cols, labels, weights, headtail, train_size, test_size, nb_nodes, nb_edges = pickle.load(f)

        degrees = np.array([len(x) for x in headtail])
        num_snap = test_size + train_size

        edges = [np.vstack((rows[i], cols[i])).T for i in range(num_snap)]
        adjs, eigen_adjs = self.get_adjs(rows, cols, weights, nb_nodes)

        labels = [torch.LongTensor(label) for label in labels]

        snap_train = list(range(num_snap))[:train_size]
        snap_test = list(range(num_snap))[train_size:]

        idx = list(range(nb_nodes))
        index_id_map = {i:i for i in idx}
        idx = np.array(idx)

        return {'X': None, 'A': adjs, 'S': eigen_adjs, 'index_id_map': index_id_map, 'edges': edges,
                'y': labels, 'idx': idx, 'snap_train': snap_train, 'degrees': degrees,
                'snap_test': snap_test, 'num_snap': num_snap}

"""## TADDY PE - Utils.py"""

## WL dict
def WL_setting_init(node_list, link_list):
    node_color_dict = {}
    node_neighbor_dict = {}

    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in link_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    return node_color_dict, node_neighbor_dict

def compute_zero_WL(node_list, link_list):
    WL_dict = {}
    for i in node_list:
        WL_dict[i] = 0
    return WL_dict

## batching + hop + int + time
def compute_batch_hop(node_list, edges_all, num_snap, Ss, k=5, window_size=1):

    batch_hop_dicts = [None] * (window_size-1)
    s_ranking = [0] + list(range(k+1))

    Gs = []
    for snap in range(num_snap):
        G = nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(edges_all[snap])
        Gs.append(G)


    indexes = [] ####

    for snap in range(window_size - 1, num_snap):
        batch_hop_dict = {}
        edges = edges_all[snap]

        current_indexes = []
        for edge in edges:
            edge_idx = str(snap) + '_' + str(edge[0]) + '_' + str(edge[1])
            batch_hop_dict[edge_idx] = []
            for lookback in range(window_size):
                # s = np.array(Ss[snap-lookback][edge[0]] + Ss[snap-lookback][edge[1]].todense()).squeeze()
                s = Ss[snap - lookback][edge[0]] + Ss[snap - lookback][edge[1]]
                s[edge[0]] = -1000 # don't pick myself
                s[edge[1]] = -1000 # don't pick myself
                top_k_neighbor_index = s.argsort()[-k:][::-1]

                indexs = np.hstack((np.array([edge[0], edge[1]]), top_k_neighbor_index))

                for i, neighbor_index in enumerate(indexs):
                    try:
                        hop1 = nx.shortest_path_length(Gs[snap-lookback], source=edge[0], target=neighbor_index)
                    except:
                        hop1 = 99
                    try:
                        hop2 = nx.shortest_path_length(Gs[snap-lookback], source=edge[1], target=neighbor_index)
                    except:
                        hop2 = 99
                    hop = min(hop1, hop2)
                    batch_hop_dict[edge_idx].append((neighbor_index, s_ranking[i], hop, lookback))

            current_indexes.append(edge) ####

        batch_hop_dicts.append(batch_hop_dict)
        indexes.append(np.stack(current_indexes)) ###
    return batch_hop_dicts#, indexes

# Dict to embeddings
def dicts_to_embeddings(feats, batch_hop_dicts, wl_dict, num_snap, use_raw_feat=False, return_indexes=False):

    raw_embeddings = []
    wl_embeddings = []
    hop_embeddings = []
    int_embeddings = []
    time_embeddings = []

    all_edge_indexes = [] ####

    for snap in range(num_snap):


        current_edge_indexes = [] ####

        batch_hop_dict = batch_hop_dicts[snap]

        if batch_hop_dict is None:
            raw_embeddings.append(None)
            wl_embeddings.append(None)
            hop_embeddings.append(None)
            int_embeddings.append(None)
            time_embeddings.append(None)

            all_edge_indexes.append(np.array([[0,0]])) ####

            continue

        raw_features_list = []
        role_ids_list = []
        position_ids_list = []
        hop_ids_list = []
        time_ids_list = []

        for edge_idx in batch_hop_dict:

            neighbors_list = batch_hop_dict[edge_idx]
            edge = edge_idx.split('_')[1:]
            edge[0], edge[1] = int(edge[0]), int(edge[1])

            current_edge_indexes.append([edge[0], edge[1]]) ####

            raw_features = []
            role_ids = []
            position_ids = []
            hop_ids = []
            time_ids = []

            for neighbor, intimacy_rank, hop, time in neighbors_list:
                if use_raw_feat:
                    raw_features.append(feats[snap-time][neighbor])
                else:
                    raw_features.append(None)
                role_ids.append(wl_dict[neighbor])
                hop_ids.append(hop)
                position_ids.append(intimacy_rank)
                time_ids.append(time)

            raw_features_list.append(raw_features)
            role_ids_list.append(role_ids)
            position_ids_list.append(position_ids)
            hop_ids_list.append(hop_ids)
            time_ids_list.append(time_ids)

        if use_raw_feat:
            raw_embedding = torch.FloatTensor(raw_features_list)
        else:
            raw_embedding = None
        wl_embedding = torch.LongTensor(role_ids_list)
        hop_embedding = torch.LongTensor(hop_ids_list)
        int_embedding = torch.LongTensor(position_ids_list)
        time_embedding = torch.LongTensor(time_ids_list)

        raw_embeddings.append(raw_embedding)
        wl_embeddings.append(wl_embedding)
        hop_embeddings.append(hop_embedding)
        int_embeddings.append(int_embedding)
        time_embeddings.append(time_embedding)


        all_edge_indexes.append(np.array(current_edge_indexes)) ####


    if return_indexes:
      return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, all_edge_indexes
    else:
      return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

"""## Encoders - Edge / Light Encoder / BERT"""

class EdgeEncoding(nn.Module):
    def __init__(self, config):
        super(EdgeEncoding, self).__init__()
        self.config = config

        self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.time_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)

        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, init_pos_ids=None, hop_dis_ids=None, time_dis_ids=None):
        # print("-"*16);print(init_pos_ids.shape);print(hop_dis_ids.shape);print(time_dis_ids.shape)
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)
        time_embeddings = self.hop_dis_embeddings(time_dis_ids)

        embeddings = position_embeddings + hop_embeddings + time_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LiteTFEnc(nn.Module):
    def __init__(self, config):
        super(LiteTFEnc, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([LiteTFLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
        # print("-"*16);print(hidden_states.shape)
        outputs = (hidden_states,)
        return outputs


class LiteTFLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class LiteBaseModel(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(LiteBaseModel, self).__init__(config)
        self.config = config

        self.embeddings = EdgeEncoding(config)
        self.encoder = LiteTFEnc(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.raw_feature_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.raw_feature_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, head_mask=None):
        embedding_output = self.embeddings(init_pos_ids=init_pos_ids, hop_dis_ids=hop_dis_ids, time_dis_ids=time_dis_ids)

        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    def run(self):
        pass


class EdgeBertModel(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(EdgeBertModel, self).__init__(config)
        self.config = config
        self.encoder = LiteTFEnc(config)
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, edge_embeddings, head_mask=None):
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(edge_embeddings, head_mask=head_mask)
        return encoder_outputs

    def run(self):
        pass


class NodeBertModel(BertPreTrainedModel):
    data = None

    def __init__(self, config):
        super(NodeBertModel, self).__init__(config)
        self.config = config
        self.encoder = LiteTFEnc(config)
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, edge_embeddings, head_mask=None):
        head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(edge_embeddings, head_mask=head_mask)
        return encoder_outputs

    def run(self):
        pass

"""## Negative Sampling"""

### Edge Features

def compute_normalized_degrees(edge_lists, num_nodes):
  T = len(edge_lists)
  degrees = torch.zeros(T, num_nodes, 2)  # T x |V| x 2

  for t, edges in enumerate(edge_lists):
    for edge in edges:
      source, target = edge
      degrees[t, source, 0] += 1
      degrees[t, target, 1] += 1

    max_degree = degrees[t].max()
    if max_degree > 0:
      degrees[t] /= max_degree

  ### CAUTION
  # degrees = torch.where(degrees == 0, torch.tensor(-0.99), degrees)
  return degrees


def generate_edge_features(edge_lists, edge_snap, t, degrees, tau):
  current_features = []
  for edge in edge_snap:
    source, target = edge
    feature_vector = np.zeros(tau * 2)

    for k in range(tau):
      if t-k-1 >= 0:
        feature_vector[2*k] = degrees[t-k-1, source, 0]
        feature_vector[2*k+1] = degrees[t-k-1, target, 1]

    current_features.append(feature_vector)
  return torch.Tensor(np.stack(current_features))


def negative_sampling(edges, node_list):
    negative_edges = []
    node_list = node_list
    num_node = node_list.shape[0]
    for snap_edge in edges:
        num_edge = snap_edge.shape[0]

        negative_edge = snap_edge.copy()
        fake_idx = np.random.choice(num_node, num_edge)
        fake_position = np.random.choice(2, num_edge).tolist()
        fake_idx = node_list[fake_idx]
        negative_edge[np.arange(num_edge), fake_position] = fake_idx

        negative_edges.append(negative_edge)
    return negative_edges


def min_max_norm(data):
  min_vals = data.min(axis=0)
  max_vals = data.max(axis=0)

  scale = max_vals - min_vals
  scale[scale == 0] = 1

  normalized_data = (data - min_vals) / scale
  return normalized_data


class VectorEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout_rate=0.5):
        super(VectorEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

"""## RWOP Codes"""

def seen_nodes_accum(edges):
  all_v_accum = []
  all_seen_inactive_nodes = []

  current_v_accum = set()
  last_v_accum = set()
  for i,a in enumerate(edges):
    current_v = set(np.unique(a.flatten()))
    current_seen_inactive = current_v_accum - current_v
    current_v_accum = current_v_accum | current_v
    all_v_accum.append(current_v_accum)
    all_seen_inactive_nodes.append(current_seen_inactive)
    print("Snapshot:%2d | Seen Nodes:%4d  ||  Current Seen Inactive Nodes: %4d"%(i, len(current_v_accum), len(current_seen_inactive)))
  return all_v_accum, all_seen_inactive_nodes

from scipy.linalg import orthogonal_procrustes

def alignment(node2vec_feats_raw, edges, lentrain, rwopref='first'):
  T, N, d = node2vec_feats_raw.shape
  all_v_accum, all_seen_inactive_nodes = seen_nodes_accum(edges)
  if rwopref=='lasttrain':
    print("OP with last train")
    embs_op = rwop_last_trained(node2vec_feats_raw, all_seen_inactive_nodes, lentrain)
  elif rwopref=='first':
    print("OP with first train")
  return embs_op

def rwop_last_trained(raw, all_seen_inactive_nodes, lentrain):
  print("RW OP using last trained")

  embs_op = copy.copy(raw)

  print("from last train to first train 鈫?")
  for i in range(lentrain-1, -1, -1): # e.g. train snap = 0~6; i from 6 to 0
    # common_nodes = list(all_v_accum[i])
    nodes_ref = all_seen_inactive_nodes[i+1]
    nodes_current = all_seen_inactive_nodes[i]
    common_nodes = list(set(nodes_ref) & set(nodes_current))
    print("snap %2d, ref: %2d  ||  Based on %4d inactive seen nodes"%(i, i+1, len(common_nodes)))
    e_ref = embs_op[i+1][common_nodes] # valid nodes at i-1
    e_trans_before = raw[i][common_nodes] # same valid nodes at i

    R, _ = orthogonal_procrustes(e_trans_before, e_ref)
    e_trans_after = np.dot(raw[i], R)
    embs_op[i] = e_trans_after

  print("from last train to last test 鈫?")
  for i in range(lentrain+1, len(raw)):
    # common_nodes = list(all_v_accum[i-1])
    nodes_ref = all_seen_inactive_nodes[i-1]
    nodes_current = all_seen_inactive_nodes[i]
    common_nodes = list(set(nodes_ref) & set(nodes_current))
    print("snap %2d, ref: %2d  ||  Based on %4d inactive seen nodes"%(i, i-1, len(common_nodes)))

    e_ref = embs_op[i-1][common_nodes] # valid nodes at i-1
    e_trans_before = raw[i][common_nodes] # same valid nodes at i

    R, _ = orthogonal_procrustes(e_trans_before, e_ref)
    e_trans_after = np.dot(raw[i], R)
    embs_op[i] = e_trans_after

  return embs_op

"""# Model"""

###########################
### LEARNING CODE
###########################


## Dynamic Graph Transformer using Orthogonal Procruste Embedding on Random-walk
## DGTOPER
from torch.optim.lr_scheduler import CyclicLR

class DGTOPER(BertPreTrainedModel):
  def __init__(self, config, config2=None, config3=None, args=None):
    super(DGTOPER, self).__init__(config, args)
    self.args = args
    self.config = config
    self.N = args.N
    self.tau = args.window_size #- 1
    self.loaded_data = loaded_data
    self.seed = args.seed
    self.valrate = args.valrate
    self.patience = args.patience

    self.rwopref = args.rwopref
    self.lentrain = args.lentrain

    self.global_dropout = args.global_dropout
    self.global_droplayer = nn.Dropout(args.global_dropout)

    self.v_feat = args.v_feat
    self.schedular = args.schedular

    if args.v_feat == 'emb':
      self.NodeEmb = nn.Embedding(num_embeddings=args.N, embedding_dim=config2.hidden_size)
    elif args.v_feat == 'deg' or 'rod':
      self.emb_edgefeat = VectorEmbedding(self.tau*2, config.hidden_size)
    elif args.v_feat == 'rw':
      pass
    elif args.v_feat == 'rwop':
      pass
    elif args.v_feat == 'no':
      pass

    self.subg_tf = args.subg_tf
    if self.subg_tf:
      self.taddy_feat = LiteBaseModel(config)
    else:
      self.taddy_feat = EdgeEncoding(config)


    self.node_tf = args.node_tf
    if args.node_tf:
      self.vtf = NodeBertModel(config2)

    self.edge_tf = args.edge_tf
    if args.edge_tf:
      self.etf = EdgeBertModel(config3)

    if self.v_feat == "rod":
      self.cls_y = torch.nn.Linear(config.hidden_size*3, 1)
    elif self.v_feat == "no" or not self.subg_tf:
      self.cls_y = torch.nn.Linear(config.hidden_size, 1)
    else:
      self.cls_y = torch.nn.Linear(config.hidden_size*2, 1)

    self.rw_norm = args.rw_norm

    self.rw_linear = torch.nn.Linear(16*2, config.hidden_size) # emb rw d = 16 / node
    self.rw_dropout = nn.Dropout(p=args.global_dropout)
    self.rw_act = nn.ReLU()

    self.weight_decay = config.weight_decay
    self.init_weights()
    self.wblog = args.wblog


  def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, edgefeat=None, vidx=None,
              rwfeat=None, idx=None, snap=None):

    ## Original TADDY MAIN
    if self.subg_tf:
      outputs = self.taddy_feat(init_pos_ids, hop_dis_ids, time_dis_ids)
      sequence_output = 0
      for i in range(self.config.k+1):
        if self.subg_tf:
          sequence_output += outputs[0][:,i,:]
        else:
          sequence_output += outputs[:,i,:]
      sequence_output /= float(self.config.k+1)

      if self.global_dropout:
        sequence_output = self.global_droplayer(sequence_output)

    ## Node features & Node Transformer
    if self.v_feat == 'emb':
      nodes_src = vidx[:,0]
      nodes_tgt = vidx[:,1]
      emb_v_src = self.NodeEmb(nodes_src)
      emb_v_tgt = self.NodeEmb(nodes_tgt)

      if self.node_tf:
        emb_v_src = self.vtf(emb_v_src.unsqueeze(0))[0].squeeze()
        emb_v_tgt = self.vtf(emb_v_tgt.unsqueeze(0))[0].squeeze()
      nodes_feat = torch.cat((emb_v_src, emb_v_tgt), dim=-1) # d=16*2

    elif self.v_feat == 'deg':
      nodes_feat = self.emb_edgefeat(edgefeat)

    elif self.v_feat in ['rw', 'rwop', 'rod']: ## The OP Processing is done out of forward func in training loop
      node2vec_feat = rwfeat # d=16
      nodes_src = vidx[:,0]
      nodes_tgt = vidx[:,1]
      emb_v_src = node2vec_feat[nodes_src]
      emb_v_tgt = node2vec_feat[nodes_tgt]

      if self.node_tf:
        emb_v_src = self.vtf(emb_v_tgt.unsqueeze(0))[0].squeeze()
        emb_v_tgt = self.vtf(emb_v_tgt.unsqueeze(0))[0].squeeze()
      nodes_feat = torch.cat((emb_v_src, emb_v_tgt), dim=-1) # d=16*2
      nodes_feat = self.rw_linear(nodes_feat)
      nodes_feat = self.rw_act(nodes_feat)
      nodes_feat = self.rw_dropout(nodes_feat)

    if self.v_feat == 'rod':
      nodes_feat2 = self.emb_edgefeat(edgefeat)

    if self.global_dropout and self.node_tf:
      nodes_feat = self.global_droplayer(nodes_feat)

    ## Final Combine
    if self.v_feat == "rod":
      combined_output = torch.cat((sequence_output, nodes_feat, nodes_feat2), dim=-1)
    elif self.v_feat != "no" and self.subg_tf:
      combined_output = torch.cat((sequence_output, nodes_feat), dim=-1)
    elif self.subg_tf:
      combined_output = sequence_output
    elif self.v_feat != "no":
      combined_output = nodes_feat

    ## Additional Edge Transformer
    if self.edge_tf:
      combined_output = self.etf(combined_output.unsqueeze(0))[0].squeeze()
      if self.global_dropout:
        combined_output = self.global_droplayer(combined_output)

    output = self.cls_y(combined_output)
    return output


  def batch_cut(self, idx_list):
      batch_list = []
      for i in range(0, len(idx_list), self.config.batch_size):
          batch_list.append(idx_list[i:i + self.config.batch_size])
      return batch_list


  def evaluate(self, trues, preds):
    roc_aucs = {}
    pr_aucs = {}

    for snap in range(len(self.data['snap_test'])):
        roc_auc = metrics.roc_auc_score(trues[snap], preds[snap])
        roc_aucs[snap] = roc_auc

        pr_auc = metrics.average_precision_score(trues[snap], preds[snap])
        pr_aucs[snap] = pr_auc

    trues_full = np.hstack(trues)
    preds_full = np.hstack(preds)

    roc_auc_full = metrics.roc_auc_score(trues_full, preds_full)
    pr_auc_full = metrics.average_precision_score(trues_full, preds_full)
    return roc_aucs, pr_aucs, roc_auc_full, pr_auc_full


  def generate_embedding(self, edges, return_indexes=False):
      num_snap = len(edges)

      WL_dict = compute_zero_WL(self.data['idx'],  np.vstack(edges[:7]))
      batch_hop_dicts = compute_batch_hop(self.data['idx'], edges, num_snap, self.data['S'], self.config.k, self.config.window_size)
      raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, indexes = \
          dicts_to_embeddings(self.data['X'], batch_hop_dicts, WL_dict, num_snap, return_indexes=True)

      if return_indexes:
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings, indexes
      else:
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings


  def negative_sampling(self, edges):
      negative_edges = []
      node_list = self.data['idx']
      num_node = node_list.shape[0]
      for snap_edge in edges:
          num_edge = snap_edge.shape[0]

          negative_edge = snap_edge.copy()
          fake_idx = np.random.choice(num_node, num_edge)
          fake_position = np.random.choice(2, num_edge).tolist()
          fake_idx = node_list[fake_idx]
          negative_edge[np.arange(num_edge), fake_position] = fake_idx

          negative_edges.append(negative_edge)
      return negative_edges



  def train_model(self, max_epoch):
      optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
      if self.schedular:
        scheduler = CyclicLR(optimizer, base_lr=self.lr/10, max_lr=self.lr,
                    step_size_up=10, mode='triangular', cycle_momentum=False)


      raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
      self.data['raw_embeddings'] = None

      ns_function = self.negative_sampling

      ## Deg for test
      if self.v_feat in ["emb","deg","rw","rwop","rod"]:
        dyn_deg = compute_normalized_degrees(self.data['edges'], self.N)

        test_e_feats = []
        for test_t in range(max(self.data['snap_test'])+1):
          pos_neg_all_edges = np.vstack([self.data['edges'][test_t]])
          current_e_feats = generate_edge_features(self.data['edges'], pos_neg_all_edges, test_t, dyn_deg, self.tau)
          test_e_feats.append(current_e_feats)

      ## RW for all
      if self.v_feat == "rw":
        node2vec_feats = np.load('data/ctdne/%s_%.1f_%s_sd%1d.npy'%(
          args.dataset, args.train_per, str(args.anomaly_per), self.seed))

        if self.rw_norm:
          node2vec_feats = np.array([min_max_norm(arr) for arr in node2vec_feats])


      if self.v_feat in ["rwop", "rod"]:
        node2vec_feats_raw = np.load('data/ctdne/%s_%.1f_%s_sd%1d.npy'%(
          args.dataset, args.train_per, str(args.anomaly_per), self.seed))

        if self.rw_norm:
          node2vec_feats = np.array([min_max_norm(arr) for arr in node2vec_feats_raw])

        node2vec_feats = alignment(node2vec_feats_raw, self.data['edges'], self.lentrain, self.rwopref)


      if self.valrate:
        best_val = -99.99

      patience_counter = 0

      val_test_flag = False

      for epoch in range(max_epoch):

          t_epoch_begin = time.time()

          # -------------------------
          # print("TRAIN - NEGATIVE")
          negatives = ns_function(self.data['edges'][:max(self.data['snap_train']) + 1])
          raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, \
          time_embeddings_neg, negative_indexes = self.generate_embedding(negatives, return_indexes=True)
          self.train()


          ## Deg for train
          if self.v_feat in ["emb","deg","rw","rwop","rod"]:
            train_e_feats = []
            train_all_edges = []
            for train_t in range(max(self.data['snap_train'])+1):
              pos_neg_all_edges = np.vstack([self.data['edges'][train_t], negative_indexes[train_t]])
              current_e_feats = generate_edge_features(self.data['edges'], pos_neg_all_edges, train_t, dyn_deg, self.tau)
              train_e_feats.append(current_e_feats)
              train_all_edges.append(pos_neg_all_edges)


          if self.valrate:
            val_acc = []


          loss_train = 0
          for snap in self.data['snap_train'][:-1]:

            if wl_embeddings[snap] is None:
              continue
            int_embedding_pos = int_embeddings[snap]
            hop_embedding_pos = hop_embeddings[snap]
            time_embedding_pos = time_embeddings[snap]
            y_pos = self.data['y'][snap].float()

            int_embedding_neg = int_embeddings_neg[snap]
            hop_embedding_neg = hop_embeddings_neg[snap]
            time_embedding_neg = time_embeddings_neg[snap]
            y_neg = torch.ones(int_embedding_neg.size()[0])

            int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
            hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
            time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
            y = torch.hstack((y_pos, y_neg)).unsqueeze(1)

            if self.valrate:
              Npos = y_pos.shape[0]
              Nneg = y_neg.shape[0]

              pos_perm = torch.randperm(Npos)
              neg_perm = torch.randperm(Nneg) + Npos

              num_train_pos = int((1-self.valrate) * Npos)
              num_train_neg = int((1-self.valrate) * Nneg)
              num_val_pos = Npos - num_train_pos
              num_val_neg = Nneg - num_train_neg

              train_idx = torch.cat((pos_perm[:num_train_pos], neg_perm[:num_train_neg])).tolist()
              val_idx = torch.cat((pos_perm[-num_val_pos:], neg_perm[-num_val_neg:])).tolist()

            optimizer.zero_grad()


            if self.v_feat in ["emb","rw","rwop","rod"]:
              current_train_edges = torch.Tensor(train_all_edges[snap])
              current_train_edges = current_train_edges.long()

            if self.v_feat == "emb":
              output = self.forward(int_embedding, hop_embedding, time_embedding, vidx=current_train_edges, snap=snap).squeeze()

            elif self.v_feat == "deg":
              edge_deg_feat = train_e_feats[snap]
              output = self.forward(int_embedding, hop_embedding, time_embedding, edgefeat=edge_deg_feat, snap=snap).squeeze()

            elif self.v_feat in ["rw","rwop"]:
              edge_rw_feat = node2vec_feats[snap]
              output = self.forward(int_embedding, hop_embedding, time_embedding, vidx=current_train_edges
                                    , rwfeat=torch.Tensor(edge_rw_feat), snap=snap)#.squeeze()
            elif self.v_feat == "rod":
              edge_deg_feat = train_e_feats[snap]
              edge_rw_feat = node2vec_feats[snap]
              output = self.forward(int_embedding, hop_embedding, time_embedding, edgefeat=edge_deg_feat,
                          vidx=current_train_edges, rwfeat=torch.Tensor(edge_rw_feat), snap=snap)

            else:
              output = self.forward(int_embedding, hop_embedding, time_embedding, snap=snap)

            if self.valrate:
              loss = F.binary_cross_entropy_with_logits(output[train_idx], y[train_idx])
              loss.backward()
              optimizer.step()

              loss_train += loss.detach().item()

              with torch.no_grad():
                output_val_sigmoid = torch.sigmoid(output[val_idx]).squeeze().numpy()
                y_val = y[val_idx].numpy().astype(int)
                val_acc.append(metrics.roc_auc_score(y_val, output_val_sigmoid))

            else:
              loss = F.binary_cross_entropy_with_logits(output, y)
              loss.backward()
              optimizer.step()

              if self.schedular:
                scheduler.step()

              loss_train += loss.detach().item()

          loss_train /= len(self.data['snap_train']) - self.config.window_size + 1
          time_train = time.time() - t_epoch_begin


          ## Valid
          if self.valrate:
            current_val = np.mean(val_acc)
            print('Epoch: {}, loss:{:.3f}, val_auc:{:.3f}, Time: {:4d}s'.format(epoch + 1, loss_train, current_val, int(time_train)))
            if epoch >= args.patience:
              if current_val > best_val:
                best_val = current_val
                patience_counter = 0
                val_test_flag = True
              else:
                patience_counter += 1
                val_test_flag = False
          else:
            print('Epoch: {}, loss:{:.3f}, Time: {:4d}s'.format(epoch + 1, loss_train, int(time_train)))


          ## Test
          if ((epoch + 1) % self.args.print_feq) == 0 or val_test_flag:
              self.eval()
              preds = []
              for snap in self.data['snap_test']:

                  int_embedding = int_embeddings[snap]
                  hop_embedding = hop_embeddings[snap]
                  time_embedding = time_embeddings[snap]


                  with torch.no_grad():

                      if self.v_feat in ["emb","rw","rwop","rod"]:
                        current_edges = torch.Tensor(self.data['edges'][snap]).long()

                      if self.v_feat == "emb":
                        output = self.forward(int_embedding, hop_embedding, time_embedding, vidx=current_edges, snap=snap)

                      elif self.v_feat == "deg":
                        edge_deg_feat = test_e_feats[snap]
                        output = self.forward(int_embedding, hop_embedding, time_embedding, edgefeat=edge_deg_feat, snap=snap)

                      elif self.v_feat in ["rw","rwop"]:
                        edge_rw_feat = node2vec_feats[snap]
                        output = self.forward(int_embedding, hop_embedding, time_embedding, vidx=current_edges
                                              , rwfeat=torch.Tensor(edge_rw_feat), snap=snap)
                      elif self.v_feat == "rod":
                        edge_deg_feat = test_e_feats[snap]
                        edge_rw_feat = node2vec_feats[snap]
                        output = self.forward(int_embedding, hop_embedding, time_embedding, edgefeat=edge_deg_feat,
                                    vidx=current_edges, rwfeat=torch.Tensor(edge_rw_feat), snap=snap)

                      else:
                        output = self.forward(int_embedding, hop_embedding, time_embedding, snap=snap).squeeze()


                      output = torch.sigmoid(output)
                  pred = output.squeeze().numpy()
                  preds.append(pred)

              y_test = self.data['y'][min(self.data['snap_test']):max(self.data['snap_test'])+1]
              y_test = [y_snap.numpy() for y_snap in y_test]

              aucs, pr_aucs, auc_full, pr_auc_full = self.evaluate(y_test, preds)

              if args.wblog:
                wandb.log({"epoch": epoch, "train loss": loss_train, "train time": time_train,
                           "test AUC ROC": auc_full, "test PR AUC": pr_auc_full})

              if val_test_flag:
                print('鈫? Best Val Model - Test Result 鈫?')
                best_model_auc = auc_full
                best_model_prauc = pr_auc_full

              print('TOTAL AUC ROC:{:.4f}'.format(auc_full), 'TOTAL PR  AUC:{:.4f}'.format(pr_auc_full))

          if patience_counter >= self.patience:
            print("Early stopping triggered at epoch ", epoch)
            break



      if args.wblog:
        if self.valrate:
          wandb.summary['Final Test AUC'] = best_model_auc
          wandb.summary['Final Test PRAUC'] = best_model_prauc
        else:
          wandb.summary['Final Test AUC'] = auc_full
          wandb.summary['Final Test PRAUC'] = pr_auc_full
          print(preds)
          print(y_test)
        wandb.finish()



  def run(self):
      self.train_model(self.max_epoch)
      return
      # self.learning_record_dict

"""# Train.py"""

wandb.finish()

debug = False # @param {type:"boolean"}
anorms = [.01, .05, .10]

datasets = ['btc_alpha', 'btc_otc', 'uci', 'digg']
seeds = [1,2,3,4,5]

for sd in seeds:
  for dss in datasets:
    for anomaly_per in anorms:
      args = Namespace()
      args.wblog = True # @param {type:"boolean"}
      args.debug = debug
      wb_project = "DGTOPER-Contrib"
      if args.wblog:
        wandb.finish()

      args.dataset = dss
      args.anomaly_per = anomaly_per

      args.prefix = '...'
      args.v_feat = 'rwop' # @param ['deg', 'rw', 'rwop', 'emb', 'no', 'rod']
      args.postfix = '...' # @param
      if len(args.postfix) > 1:
        args.modelname = 'DGTOPER' + '_' + args.postfix
      else:
        args.modelname = 'DGTOPER' # @param

      args.subg_tf = True # @param {type:'boolean'}
      args.node_tf = True # @param {type:'boolean'}
      args.edge_tf = False # @param {type:'boolean'}
      if args.v_feat=='no':
        args.node_tf = False
        nodes_feat = None
      args.rw_norm = True # @param {type:'boolean'}

      args.rwopref = 'lasttrain' # @param ['first', 'lasttrain', 'no']
      if args.v_feat not in ['rw', 'rwop', 'rod']:
        args.rwopref = 'no'
        args.rw_norm = False

      args.valrate = 0.1 # @param {type:"slider", min:0.0, max:0.5, step:0.05}
      args.patience = 10 # @param {type:"slider", min:0, max:50, step:5}
      ## Default
      args.train_per = 0.5
      args.neighbor_num = 5
      args.window_size = 2 # @param {type:"slider", min:2, max:4, step:1}

      args.embedding_dim = 32#32
      if dss == "uci":
        args.embedding_dim = 8#32

      args.num_hidden_layers = 2
      args.num_attention_heads = 2

      args.embedding_dim_vtf = 16
      args.num_hidden_layers_vtf = 2
      args.num_attention_heads_vtf = 2
      args.weight_decay2 = 0.0005 #5e-4

      if args.v_feat == 'rod':
        args.embedding_dim_etf = 96
      elif args.node_tf and args.subg_tf:
        args.embedding_dim_etf = 64
      else:
        args.embedding_dim_etf = 32
      args.num_hidden_layers_etf = 2
      args.num_attention_heads_etf = 2
      args.weight_decay3 = 0.005

      args.max_epoch = 100
      args.lr = 0.0001

      args.weight_decay = 0
      if args.node_tf:
        args.global_dropout = 0.2

      args.schedular = None

      args.seed = sd
      args.print_feq = 1

      ## Set Seed
      set_seed(args.seed)

      ## Debug use
      if args.debug:
        args.lr *= 100
        args.max_epoch = 300

      ############################
      #### Start
      ############################
      print('$$$$ Start $$$$')
      data_obj = DynamicDatasetLoader()
      data_obj.dataset_name = args.dataset
      data_obj.k = args.neighbor_num
      data_obj.window_size = args.window_size
      data_obj.anomaly_per = args.anomaly_per
      data_obj.train_per = args.train_per
      data_obj.load_all_tag = False
      data_obj.compute_s = True
      data_obj.seed = sd


      ## Generate Hyper param config Class
      conf1 = MyConfig(k=args.neighbor_num, window_size=args.window_size, hidden_size=args.embedding_dim,
                          intermediate_size=args.embedding_dim, num_attention_heads=args.num_attention_heads,
                          num_hidden_layers=args.num_hidden_layers, weight_decay=args.weight_decay)
      conf2 = None
      conf3 = None

      if args.node_tf:
        conf2 = copy.copy(conf1)
        conf2.hidden_size = args.embedding_dim_vtf
        conf2.num_hidden_layers = args.num_hidden_layers_vtf
        conf2.num_attention_heads = args.num_attention_heads_vtf
        conf2.weight_decay = args.weight_decay2

      if args.edge_tf:
        conf3 = copy.copy(conf1)
        conf3.hidden_size = args.embedding_dim_etf
        conf3.num_hidden_layers = args.num_hidden_layers_etf
        conf3.num_attention_heads = args.num_attention_heads_etf
        conf3.weight_decay = args.weight_decay3

      ##
      loaded_data = data_obj.load()
      args.N = loaded_data['idx'].max()+1
      args.lentrain = len(loaded_data['snap_train'])

      method_obj = DGTOPER(conf1, config2=conf2, config3=conf3, args=args)
      method_obj.spy_tag = True
      method_obj.max_epoch = args.max_epoch
      method_obj.lr = args.lr
      method_obj.data = loaded_data

      if args.wblog:
        wandb.init(project=wb_project, name=args.prefix + '_' + args.modelname + '_' +\
                  args.dataset + "_anom%2d"%(args.anomaly_per*100) + "_sd%d"%args.seed, config=vars(args))
      print(args)
      method_obj.run()

      del data_obj
      del method_obj
      gc.collect()

