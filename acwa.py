# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('...')

# from codes.DynamicDatasetLoader import DynamicDatasetLoader
from codes.Component import MyConfig
# from codes.DynADModel import DynADModel
from codes.Settings import Settings
import numpy as np
import torch

from argparse import Namespace
import random


def set_seed(seed):
  os.environ['PYTHONHASHSEED']=str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True)


import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
from sklearn import metrics
from argparse import Namespace

"""# Train Node2Vec"""

import os.path as osp
import sys
from CTDNE import CTDNE
import networkx as nx

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def node2vec_emb(edges, d, L=20, epo=100):
  model = Node2Vec(
    edges,
    embedding_dim=d,
    walk_length=L,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    sparse=True,
  ).to(device)
  loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
  optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

  def train(model):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
      optimizer.zero_grad()
      loss = model.loss(pos_rw.to(device), neg_rw.to(device))
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    return model, total_loss / len(loader)


  for epoch in range(epo):
    model, loss = train(model)

  return model().detach().cpu().numpy()


def ctdne_emb(g, d, L=30, NW=200):
  CTDNE_model = CTDNE(graph, dimensions=d, walk_length=L, num_walks=NW, workers=4)
  model = CTDNE_model.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the CTDNE constructor)
  ## CREATE ZERO ARRAY
  ## REPLACE ZERO VECTORS WITH LEARNT
  return model

## DEBUG

from codes.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
import os
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
            if self.dataset_name == "digg": ####
                adj = sp.csr_matrix((weights[i], (rows[i], cols[i])), shape=(nb_nodes, nb_nodes), dtype=np.float16)
            else:
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





%%time

anorms = [.01, .05, .10]

datasets = ['uci', 'btc_alpha', 'btc_otc', 'digg']
d = 16

seeds = [1,2,3,4,5]

for sd in seeds:
  for anomaly_per in anorms:
    for dataset in datasets:
      args = Namespace()
      ## Customized
      args.dataset = dataset
      args.anomaly_per = anomaly_per

      ## Default
      args.train_per = 0.5
      args.neighbor_num = 5
      # args.window_size = 2
      args.window_size = 3

      args.embedding_dim = 32 #32
      args.num_hidden_layers = 2
      args.num_attention_heads = 2

      args.num_hidden_layers2 = 2
      args.num_attention_heads2 = 4

      args.max_epoch = 100 if args.dataset in ['uci', 'btc_alpha', 'btc_otc'] else 200
      args.lr = 0.001 / 10
      args.weight_decay = 0.0005 #5e-4
      args.weight_decay2 = 0.002 #5e-4


      args.seed = sd
      args.print_feq = 5

      ## Set Seed
      set_seed(args.seed)

      ############################
      #### Start
      ############################
      print('$$$$ Start $$$$')

      start_time1 = time.time()

      data_obj = DynamicDatasetLoader()
      data_obj.dataset_name = args.dataset
      data_obj.k = args.neighbor_num
      data_obj.window_size = args.window_size
      data_obj.anomaly_per = args.anomaly_per
      data_obj.train_per = args.train_per
      data_obj.load_all_tag = False
      data_obj.compute_s = True
      data_obj.seed = sd

      ##
      # method_obj = DynADModel(my_config, args)
      loaded_data = data_obj.load()
      args.N = loaded_data['idx'].max()+1

      e_acum = np.array([args.N-1, args.N-1]) ## Fix max embedding
      embs = []



      end_time1 = time.time()
      print("TADDY-precompute time:", end_time1 - start_time1)


      set_seed(args.seed)


      start_time2 = time.time()

      for i,el in enumerate(loaded_data['edges']):
        print(i)
        start_time3 = time.time()
        current_nodes = np.unique(e_acum.flatten())

        e_acum = np.vstack([e_acum, el])

        graph = nx.from_edgelist(e_acum)
        m = len(graph.edges())
        edge2time = {edge: time for edge,time in zip(graph.edges(),(m*np.random.rand(m)).astype(int))}
        nx.set_edge_attributes(graph,edge2time,'time')

        if args.dataset in ['digg', 'ast']:
          model = ctdne_emb(graph, d=d, NW=200)
        else:
          model = ctdne_emb(graph, d=d)
        emb = np.zeros((args.N, d))
        for node in current_nodes:
          emb[node] = model.wv[model.wv.key_to_index[str(node)]]

        end_time3 = time.time()
        print("ACWA-snapshot time:", end_time3 - start_time3)

        embs.append(emb)

      end_time2 = time.time()
      print("ACWA-precompute time:", end_time2 - start_time2)


      np.save('data/ctdne/%s_%.1f_%s_sd%1d.npy'%(dataset, args.train_per, str(anomaly_per), sd), np.array(embs))
        # break

      del data_obj
      del loaded_data
      gc.collect()
