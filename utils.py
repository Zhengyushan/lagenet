#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2021/11
# author:yushan zheng
# emai:yszheng@buaa.edu.cn

import torch.nn.functional as F
import torch
import numpy as np
import os
import cv2
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt

# The definition of magnification of the endometrial dataset.
# 'Large':40X, 'Medium':20X, 'Small':10X, 'Overview':5X
scales = ['Large', 'Medium', 'Small', 'Overview']

'''
Endometrial dataset configuration
| Index | Abbr.    | Full name
|-------|----------|------------------------------------------------------
|   0   | Normal   | Normal
|   2   | WDEA     | Well-differentiated Endometrioid adenocarcinoma
|   3   | MDEA     | Moderately differentiated Endometrioid adenocarcinoma
|   4   | LDEA     | Low differentiated Endometrioid adenocarcinoma
|   10  | SEIC     | Serous endometrial intraepithelial carcinoma
'''
# lesion names
lesions = ["Normal", "WDEA", "MDEA", "LDEA", "SEIC"]
# labels for the binary task
binary_map = {0:0, 2:1, 3:1, 4:1, 10:1}
# labels for the multi-type task
sub_type_map = {0:0, 2:1, 3:2, 4:3, 10:4}

def merge_config_to_args(args, cfg):
    # dirs
    args.patch_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'patch')
    args.list_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'patch_list')

    args.cnn_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'cnn_model')
    args.feat_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'cnn_feat')

    args.graph_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'graph')
    args.graph_list_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'graph_list')
    args.gcn_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'gcn_model')
    args.lage_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'lagcn_model')
    args.hash_dir = os.path.join(cfg.DATA.DATA_SAVE_DIR, 'hash_model')

    args.slide_dir = cfg.DATA.LOCAL_SLIDE_DIR

    # data
    args.label_id = cfg.DATA.LABEL_ID
    args.test_ratio = cfg.DATA.TEST_RATIO
    args.fold_num = cfg.DATA.FOLD_NUM

    # image
    if 'IMAGE' in cfg:
        args.level = cfg.IMAGE.LEVEL
        args.mask_level = cfg.IMAGE.MASK_LEVEL
        args.imsize = cfg.IMAGE.PATCH_SIZE
        args.tile_size = cfg.IMAGE.LOCAL_TILE_SIZE
        args.rl = args.mask_level-args.level
        args.msize = args.imsize >> args.rl
        args.mhalfsize = args.msize >> 1

    # sampling
    if 'SAMPLE' in cfg:
        args.positive_ratio = cfg.SAMPLE.POS_RAT
        args.negative_ratio = cfg.SAMPLE.NEG_RAT
        args.intensity_thred = cfg.SAMPLE.INTENSITY_THRED
        args.sample_step = cfg.SAMPLE.STEP
        args.max_per_class = cfg.SAMPLE.MAX_PER_CLASS
        
        args.srstep = args.sample_step>>args.rl
        args.filter_size = (args.imsize>>args.rl, args.imsize>>args.rl)
        
    # CNN
    if 'CNN' in cfg:
        args.arch = cfg.CNN.ARCH
        args.pretrained = cfg.CNN.PRETRAINED

    # feature
    if 'FEATURE' in cfg:
        args.step = cfg.FEATURE.STEP
        args.frstep = args.step>>args.rl
        
    # graph
    if 'GRAPH' in cfg:
        args.node_num = cfg.GRAPH.NODE_NUM
        args.max_graph_per_class = cfg.GRAPH.MAX_PER_CLASS

    # lage-net
    if 'LAGENET' in cfg:
        args.lage_depth = cfg.LAGENET.DEPTH
        args.lage_heads = cfg.LAGENET.HEADS
        args.lage_dim = cfg.LAGENET.DIM
        args.lage_mlp_dim = cfg.LAGENET.MLP_DIM
        args.lage_dim_head = cfg.LAGENET.HEAD_DIM
        args.lage_pool = cfg.LAGENET.POOL 

    # hash
    if 'HASH' in cfg:
        args.hash_bits = cfg.HASH.BITS

    return args


def get_sampling_path(args):
    prefix = '[l{}t{}s{}m{}][p{}n{}i{}]'.format(args.level, args.imsize,
                                              args.sample_step, args.max_per_class,
                                              int(args.positive_ratio * 100),
                                              int(args.negative_ratio * 100),
                                              args.intensity_thred)

    return os.path.join(args.patch_dir, prefix)

def get_data_list_path(args):
    prefix = get_sampling_path(args)
    prefix = '{}[f{}_t{}]'.format(prefix[prefix.find('['):], args.fold_num,
                                int(args.test_ratio * 100))

    return os.path.join(args.list_dir, prefix)


def get_cnn_path(args):
    prefix = get_data_list_path(args)
    args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
        args.fold)
    prefix = '{}[{}_td_{}_{}]'.format(prefix[prefix.find('['):], args.arch, 
                args.label_id, args.fold_name)

    return os.path.join(args.cnn_dir, prefix)


def get_feature_path(args):
    if args.pretrained:
        prefix = '[{}_pre][fs{}]'.format(args.arch, args.step)
    else:
        prefix = get_data_list_path(args)
        args.fold_name = 'list_fold_all' if args.fold == -1 else 'list_fold_{}'.format(
                        args.fold)
        prefix = '{}[{}_td_{}][fs{}][{}]'.format(prefix[prefix.find('['):], 
            args.arch, args.label_id, args.step, args.fold_name)

    return os.path.join(args.feat_dir, prefix)


def get_graph_path(args):
    prefix = get_feature_path(args)
    prefix = '{}[grp_n{}_m{}]'.format(prefix[prefix.find('['):], 
        args.node_num, args.max_graph_per_class,)

    return os.path.join(args.graph_dir, prefix)


def get_graph_list_path(args):
    prefix = get_feature_path(args)
    prefix = '{}[grp_n{}_m{}]'.format(prefix[prefix.find('['):], 
        args.node_num, args.max_graph_per_class,)

    return os.path.join(args.graph_list_dir,prefix)


def get_slide_list_local(slide_dir):
    slides = os.listdir(slide_dir)
    slide_list = []
    for s_id, s_guid in enumerate(slides):
        # the slides in our dataset are named by guids
        if len(s_guid) < 36:
            continue

        slide_path = os.path.join(slide_dir, s_guid)
        slide_content = os.listdir(slide_path)
        # # Check data integrity
        # if len(slide_content) < 11:
        #     print(s_id, s_guid, 'is incomlete. skip.')
        #     continue

        slide_list.append(s_guid)

    return slide_list


def get_lage_path(args):
    prefix = get_graph_list_path(args)
    prefix = '{}[d{}_h_{}_de{}dm{}dh{}hb{}_td_{}][{}{}{}]'.format(prefix[prefix.find('['):], 
        args.lage_depth, args.lage_heads, args.lage_dim, args.lage_mlp_dim, args.lage_dim_head, args.hash_bits, args.label_id,
        'd' if not args.disable_distance else '_',
        'a' if args.node_aug else '_',
        'x' if not args.disable_adj else '_',
        )

    return os.path.join(args.lage_dir, prefix)

'''
The function to load the patch in position and size of (x, y, width, height) from the WSI.
Rewrite the function to fit your dataset.
'''
def extract_tile(image_dir, tile_size, x, y, width, height):
    x_start_tile = x // tile_size
    y_start_tile = y // tile_size
    x_end_tile = (x+width) // tile_size
    y_end_tile = (y+height) // tile_size

    tmp_image = np.zeros(
        ((y_end_tile-y_start_tile+1)*tile_size, (x_end_tile-x_start_tile+1)*tile_size, 3),
        np.uint8)

    for y_id, col in enumerate(range(x_start_tile, x_end_tile + 1)):
        for x_id, row in enumerate(range(y_start_tile, y_end_tile + 1)):
            img_path = os.path.join(image_dir, '{:04d}_{:04d}.jpg'.format(row,col))
            if not os.path.exists(img_path):
                return []
            tmp_image[(x_id*tile_size):(x_id+1)*tile_size, (y_id*tile_size):(y_id+1)*tile_size,:] = \
                cv2.imread(img_path)

    x_off = x % tile_size
    y_off = y % tile_size
    output = tmp_image[y_off:y_off+height, x_off:x_off+width]

    return output

'''
The function to segment the tissue area from the background.
'''
def get_tissue_mask(wsi_thumbnail, scale=30):
    hsv = cv2.cvtColor(wsi_thumbnail, cv2.COLOR_RGB2HSV)
    _, tissue_mask = cv2.threshold(hsv[:, :, 2], 210, 255, cv2.THRESH_BINARY_INV)
    tissue_mask[hsv[:, :, 0]<10]=0

    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * scale + 1, 2 * scale + 1)
        )
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, element)

    return tissue_mask

'''
The function to detect the spatial adjacency of the patches in the graph.
'''
def detect_connectivity(positions, down_factor=1):
    power = np.sum(np.multiply(positions, positions), axis=1)
    power = np.repeat(power[np.newaxis, :], positions.shape[0], axis=0)
    dist_map = np.abs(power - 2*np.dot(positions, np.transpose(positions)) + np.transpose(power))
    adj_mat = dist_map <= down_factor*down_factor

    return adj_mat

def calc_distances_to_border(content_mat):
    dist_map = cv2.distanceTransform(
                np.asarray(content_mat, np.uint8), 
                distanceType=cv2.DIST_L2, 
                maskSize=3
                )

    return dist_map    

## Evaluation functions
def retrieval(query, query_label, database, database_label):
    query[query > 0] = 1
    query[query < 0] = -1
    database[database > 0] = 1
    database[database < 0] = -1
    
    q_label_ = torch.zeros(query.size(0), max(database_label)+1
        ).scatter_(1, query_label.unsqueeze(1), 1)
    d_label_ = torch.zeros(database.size(0), max(database_label)+1
        ).scatter_(1, database_label.unsqueeze(1), 1)

    print(query.size(),database.size())
    if len(query.size()) > 2:
        hamming_distance = torch.einsum('nsk,mqk->nmsq', query, database) / 2
        hamming_distance = torch.mean(hamming_distance, dim=(2,3))
    else:
        hamming_distance = torch.matmul(query, database.T)

    sim_mat = torch.matmul(q_label_, d_label_.T).int()
    ret_index = torch.argsort(hamming_distance, axis=1, descending=True)
    _, inv_index = ret_index.sort()
    
    correct = sim_mat.clone().scatter_(
        1, inv_index, sim_mat) > 0

    return ret_index, correct.int()


def mean_average_precision(correct_mat):
    tmp_mat = np.asarray(correct_mat, np.int32)

    ave_p = np.cumsum(tmp_mat, axis=1) / np.arange(1,tmp_mat.shape[1]+1)
    ave_p_tmp = ave_p.copy()
    ave_p_tmp[tmp_mat < 1] = 0

    mean_ave_p = np.cumsum(ave_p_tmp, axis=1) / (np.cumsum(tmp_mat, axis=1) + 0.00001)

    return np.mean(mean_ave_p, axis=0)


def mean_reciprocal_rank(correct_mat):
    first_hit = np.argmax(correct_mat, axis=1)
    first_hit = np.asarray(first_hit + 1, np.float)

    return np.mean(1.0 / first_hit)


def average_precision(correct_mat, ret_num=None):
    data = correct_mat if ret_num == None else correct_mat[:,:ret_num]
    return np.mean(data)

def recall_at_n(correct_mat, ret_num):
    recall = np.max(correct_mat[:,:ret_num], axis=1)
    return np.mean(recall)


def accuracy(output, target, topk=(1,2)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def multi_auc(labels, predicts, num_classes):
    label_onehot = torch.zeros(predicts.size(0), num_classes
        ).scatter_(1, labels.unsqueeze(1), 1)

    label_onehot = label_onehot.numpy()
    preds = predicts.numpy()

    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], preds[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), preds.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
    
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    
    return roc_auc_dict