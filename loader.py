#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from utils import extract_tile


class PatchDataset(data.Dataset):
    def __init__(self, file_path, transform, od_mode=True, label_type=1):
        self.transform = transform
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.data_dir = data['base_dir']
        self.image_list = data['list']
        self.od = od_mode
        self.lt = label_type

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.image_list[index][0])).convert('RGB')
        label = self.image_list[index][self.lt]
        if self.transform!=None:
            img = self.transform(img)
        if self.od:
            img = -torch.log(img + 1.0/255.0)

        return img, label

    def __len__(self):
        return len(self.image_list)

    def get_weights(self):
        num = self.__len__()
        labels = np.zeros((num,), np.int)
        for s_ind, s in enumerate(self.image_list):
            labels[s_ind] = s[self.lt]
        tmp = np.bincount(labels)
        weights = 1.0 / np.asarray(tmp[labels], np.float)

        return weights

class CLPatchesDataset(data.Dataset):
    def __init__(self, file_path, transform, od_mode=True, label_type=1):
        self.transform = transform
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.data_dir = data['base_dir']
        self.image_list = data['list']
        self.od = od_mode
        self.lt = label_type

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.image_list[index][0])).convert('RGB')

        if self.transform!=None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        if self.od:
            img1 = -torch.log(img + 1.0/255.0)
            img2 = -torch.log(img + 1.0/255.0)

        return img1, img2

    def __len__(self):
        return len(self.image_list)

    def get_weights(self):
        num = self.__len__()
        labels = np.zeros((num,), np.int)
        for s_ind, s in enumerate(self.image_list):
            labels[s_ind] = s[self.lt]
        tmp = np.bincount(labels)
        weights = 1.0 / np.asarray(tmp[labels], np.float)

        return weights


class SlideLocalTileDataset(data.Dataset):
    def __init__(self, image_dir, position_list, transform,
            tile_size=512, imsize=224, od_mode=False):
        self.transform = transform

        self.im_dir = image_dir
        self.pos = position_list
        self.od = od_mode
        self.ts = tile_size
        self.imsize = imsize

    def __getitem__(self, index):
        img = extract_tile(self.im_dir, self.ts, self.pos[index][1], self.pos[index][0], self.imsize, self.imsize)
        if len(img) == 0:
            img = np.ones((self.imsize, self.imsize, 3), np.uint8) * 240
        img = Image.fromarray(img[:,:,[2,1,0]]).convert('RGB')
        img = self.transform(img)

        if self.od:
            img = -torch.log(img + 1.0/255.0)

        return img

    def __len__(self):
        return self.pos.shape[0]


class TissueBorderEmbedder():
    def __init__(self, feat_dim, max_len=64, interval=1.0):
        self.interval = interval
        self.max_len = int(max_len / self.interval)

        self.pe = np.zeros((self.max_len, feat_dim), float)
        position = np.arange(0, self.max_len)
        div_term = 1 / (self.max_len ** (np.arange(0, feat_dim) / feat_dim))
        pos_mat = np.matmul(position[:,np.newaxis], div_term[np.newaxis,:])

        self.pe[:, 0::2] = np.sin(pos_mat)[:,0::2]
        self.pe[:, 1::2] = np.cos(pos_mat)[:,1::2]
        

    def embed(self, dist_values):
        index = np.asarray(np.minimum(dist_values/self.interval, self.max_len-1), int)

        return self.pe[index]


class LaGraphLoader(torch.utils.data.Dataset):
    def __init__(self, graph_list_path, max_node_number, task_id=1,
                disable_adj=False, dist_embed_dim=64, 
                ):
        with open(graph_list_path, 'rb') as f:
            data = pickle.load(f)
        self.dl = data['list']
        self.list_dir = data['base_dir']
        self.type_num = 5 if task_id == 1 else 2

        self.maxno = max_node_number
        self.ti=task_id
        self.use_adj = not disable_adj

        with open(self.get_graph_path(0), 'rb') as f:
            graph_data = pickle.load(f)

        self.feat_dim = graph_data['feats'].shape[1]
        self.de_dim = dist_embed_dim
        self.de_table = TissueBorderEmbedder(self.de_dim)

    def __len__(self):
        return len(self.dl)

    def __getitem__(self, idx):
        graph_feat = np.zeros((self.maxno,self.feat_dim))
        graph_adj = np.zeros((self.maxno + 1, self.maxno + 1))
        graph_de = np.zeros((self.maxno, self.de_dim))

        with open(self.get_graph_path(idx), 'rb') as f:
            graph_data = pickle.load(f)

        # node (token) number
        num_node = min(graph_data['feats'].shape[0], self.maxno)

        # node (token) mask
        node_mask = np.zeros((self.maxno + 1, 1), int)
        node_mask[:(num_node+1)] = 1

        # node (token) features
        features = graph_data['feats'][:num_node]
        graph_feat[:num_node] = features
        
        # distance embedding
        dist_embeddings = self.de_table.embed(graph_data['dists'][:num_node])
        graph_de[:num_node] = dist_embeddings

        # adjacency matrix
        adj = graph_data['adj'][:num_node,:num_node] if self.use_adj else np.zeros((num_node,num_node))
        adj_ = np.pad(adj, ((1,0),(1,0)), 'constant', constant_values=(1,1))
        adj_ = adj_ + np.eye(num_node+1)
        degree = np.sum(adj_, axis=1) ** -0.5
        degree = np.diag(degree)
        normed_adj = np.matmul(np.matmul(degree, adj_), degree)
        graph_adj[:(num_node+1), :(num_node+1)] = normed_adj

        # graph label
        graph_label = np.asarray(self.dl[idx][self.ti], int)

        return graph_feat, graph_adj, graph_de, node_mask, graph_label

    def get_graph_path(self, idx):
        return os.path.join(self.list_dir, self.dl[idx][0])

    def get_feat_dim(self):
        return self.feat_dim
        
    def get_weights(self):
        num = self.__len__()
        labels = np.zeros((num,), np.int)
        for s_ind, s in enumerate(self.dl):
            labels[s_ind] = s[self.ti]
        tmp = np.bincount(labels)
        weights = 1.0 / np.asarray(tmp[labels], np.float)

        return weights


class DistributedWeightedSampler(data.DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):

        super(DistributedWeightedSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=False
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
