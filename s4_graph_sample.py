#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2021/11
# author:yushan zheng
# emai:yszheng@buaa.edu.cn

import os
import cv2
import argparse
import numpy as np
from yacs.config import CfgNode
import pickle
from sklearn.cluster import AgglomerativeClustering

from utils import *

parser = argparse.ArgumentParser('Sampling graph data')
parser.add_argument('--cfg', type=str,
    default='',
    help='The path of yaml config file')
parser.add_argument('--slide-dir', type=str,
    default='')
parser.add_argument('--slide-list', type=str, default='',
                    help='The list of slide guids used in the dataset')


parser.add_argument('--task-id', type=str, default='', help='task id on MoticGallery')
parser.add_argument('--feat-dir', type=str, default='')
parser.add_argument('--graph-dir', type=str, default='')

parser.add_argument('--level', type=int, default=1)
parser.add_argument('--mask-level', type=int, default=3)

parser.add_argument('--test-ratio', type=float, default=0.3)
parser.add_argument('--fold-num', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)

# CNN
parser.add_argument('--arch', type=str, default='efficientnet-b0')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained cnn model')
parser.add_argument('--imsize', type=int, default=224)
parser.add_argument('--step', type=int, default=112)

parser.add_argument('--num-classes', type=int, default=6)
parser.add_argument('--max-per-graph', type=int, default=1024, 
            help='The upper bound of patches in a graph.')
parser.add_argument('--abnormal-ratio', type=float, default=0.2, 
            help='The upper bound of patches in a graph.')
parser.add_argument('--distance-dim', type=int, default=16)
parser.add_argument('--num-workers', type=int, default=4)

parser.add_argument('--intensity-thred', type=int, default=25, help='')


def main(args):
    np.random.seed(1)
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    print(cfg)

    sample_graph(args)
    make_list(args)


def sample_graph(args):
    slide_list = get_slide_list_local(args.slide_dir)
    if args.slide_list:
        with open(args.slide_list, 'rb') as f:
            dataset_list = pickle.load(f)
        slide_list_ = []
        for slide_guid in dataset_list:
            if slide_guid in slide_list:
                slide_list_.append(slide_guid)
            else:
                print(slide_guid)
        slide_list = slide_list_

    print('slide num', len(slide_list))
    num_classes = len(sub_type_map)

    graph_dir = get_graph_path(args)

    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    feat_path = get_feature_path(args)
    for s_id, s_guid in enumerate(slide_list):
        slide_graph_dir = os.path.join(graph_dir, s_guid)
        slide_graph_summary_path = os.path.join(slide_graph_dir, 'summary')
        if os.path.exists(slide_graph_summary_path):
            print(s_id, s_guid, 'finished. skip.')
            continue

        wsi_feathre_path = os.path.join(feat_path, s_guid + '.pkl')
        if not os.path.exists(wsi_feathre_path):
            print(s_id, s_guid, 'feature does not exists')
            continue

        with open(wsi_feathre_path, 'rb') as f:
            feat_data = pickle.load(f)

        feat = feat_data['feats']
        score = feat_data['probs']
        content_mat = feat_data['content']
        
        dist_mat = calc_distances_to_border(content_mat)
        patch_dist = dist_mat[content_mat]
        patch_score = score[content_mat]
        patch_pos = np.transpose(np.asarray(np.where(content_mat)))
        adj_mat = detect_connectivity(patch_pos, 1)
        # cv2.imwrite('/home/zhengyushan/code/xsa-net/iobuf/'+ s_guid + '.png', np.asarray(content_mat*255, np.uint8))
        
        # clustering
        n_cluster = int(feat.shape[0]/args.node_num)

        estimator = AgglomerativeClustering(n_clusters=n_cluster, connectivity=adj_mat)
        feat_ = np.concatenate((feat, patch_pos*500/args.node_num), axis=1)
        estimator.fit(feat_)
        label_pred = estimator.labels_
        graph_size = np.bincount(label_pred)

        sub_graph_num = max(label_pred)+1

        slide_path = os.path.join(args.slide_dir, s_guid)
        mask = cv2.imread(os.path.join(slide_path, 'AnnotationMask.png'), 0)
        positive_mat = cv2.blur(
            (mask > 0)*255, ksize=args.filter_size, anchor=(0, 0))
        positive_mat = positive_mat[::args.frstep, ::args.frstep] > 128
        mask_mat = mask[::args.frstep, ::args.frstep]

        positive_indexes = positive_mat[content_mat]
        patch_label = mask_mat[content_mat]
        
        graph_tl = np.zeros((sub_graph_num,), float)
        for l_ind in range(sub_graph_num):
            graph_tl[l_ind] = np.mean(positive_indexes[label_pred==l_ind])
        
        pos_graph_id = np.where(graph_tl > 0.3)[0]
        neg_graph_id = np.where(graph_tl < 0.001)[0]

        if len(pos_graph_id) > args.max_graph_per_class:
            pos_graph_id = pos_graph_id[np.random.choice(
                len(pos_graph_id), args.max_graph_per_class, replace=False)]

        if len(neg_graph_id) > args.max_graph_per_class // (num_classes-1):
            neg_graph_id = neg_graph_id[np.random.choice(
                len(neg_graph_id), args.max_graph_per_class // (num_classes-1), replace=False)]

        if not os.path.exists(slide_graph_dir):
            os.makedirs(slide_graph_dir)

        graph_list = np.hstack((pos_graph_id, neg_graph_id))

        counter = 0
        graph_size_list = []
        for l_ind in graph_list:
            sub_node_id = np.where(label_pred==l_ind)[0]
            sub_pos = patch_pos[sub_node_id]
            adj = detect_connectivity(sub_pos)
            
            graph = {'adj':adj,
                    'pos':sub_pos,
                    'feats':feat[sub_node_id],
                    'probs':patch_score[sub_node_id],
                    'feats_label':patch_label[sub_node_id],
                    'dists':patch_dist[sub_node_id],
                    'tl':graph_tl[l_ind],
                    'label':max(patch_label[sub_node_id])
                    }
            
            # print('distances', max(graph['dists']), 'tl', graph['tl'], 'label', graph['label'])   
            graph_save_path = os.path.join(slide_graph_dir, str(counter))         
            with open(graph_save_path, 'wb') as f:
                pickle.dump(graph, f)
            counter += 1
            graph_size_list.append(sub_pos.shape[0])

        print(s_id, s_guid, 'graph number', counter, 'min size', min(graph_size_list),\
            'median size', np.median(graph_size_list), 'max size', max(graph_size_list))

        label_plot = np.zeros(content_mat.shape, float)
        label_plot[content_mat] = label_pred

        with open(slide_graph_summary_path, 'wb') as f:
            graph_summary = {
                'num':counter,
                'index_mat':label_plot,
                'content_mat':content_mat,
                'mask_mat':mask_mat,
                'size':graph_size_list
            }
            pickle.dump(graph_summary, f)
        


def make_list(args):

    dataset_path = get_graph_path(args)
    graph_list_path = get_graph_list_path(args)
    if not os.path.exists(graph_list_path):
        os.makedirs(graph_list_path)

    dataset_split_path = os.path.join(get_data_list_path(args), 'split.pkl')
    with open(dataset_split_path, 'rb') as f:
        folds = pickle.load(f)

    config_path = os.path.join(graph_list_path, 'list_config.csv')
    if os.path.exists(config_path):
        print('The list exists. Delete <list_config.csv> to remake the list.')
        return 0

    sample_list = []
    slide_count = 0
    for f_id, fold_list in enumerate(folds):
        sub_set_name = 'test' if f_id == args.fold_num else 'fold_{}'.format(
            f_id)

        sample_list_fold = []
        class_slide_counter = np.zeros(len(sub_type_map))
        class_image_counter = np.zeros(len(sub_type_map))

        for s_id, s_guid in enumerate(fold_list):
            slide_graph_dir = os.path.join(dataset_path, s_guid)
            slide_graph_list = os.listdir(slide_graph_dir)
            graph_labels = []
            for g_name in slide_graph_list:
                if g_name == 'summary':
                    continue
                graph_path = os.path.join(slide_graph_dir, g_name)
                with open(graph_path, 'rb') as f:
                    data = pickle.load(f)
                    g_label = data['label']

                sample_str = [
                         os.path.join(s_guid, g_name), 
                         sub_type_map[g_label], 
                         binary_map[g_label],
                         data['pos'].shape[0],
                        ]

                sample_list_fold.append(sample_str)
                graph_labels.append(sub_type_map[g_label])
                class_image_counter[sub_type_map[g_label]] += 1
                
            class_slide_counter[max(graph_labels)] += 1
            print(s_id, s_guid, 'graph number', len(slide_graph_list) - 1)

        slide_count += len(fold_list)

        with open(config_path, 'a') as f:
            print_str = '{}, slide number: '.format(sub_set_name)
            for num in class_slide_counter:
                print_str += '{},'.format(num)
            print_str += ' image number: '
            for num in class_image_counter:
                print_str += '{},'.format(num)
            print_str += '\n'
            f.write(print_str)
        print(print_str)
        sample_list.append(sample_list_fold)

    for f_id in range(args.fold_num+1):
        f_name = 'list_fold_all' if f_id == args.fold_num else 'list_fold_{}'.format(
            f_id)
        val_set = sample_list[f_id]

        train_set = []
        if f_id == args.fold_num:
            for train_f_id in range(args.fold_num+1):
                train_set += sample_list[train_f_id]
        else:
            train_index = np.hstack(
                (np.arange(0, f_id), np.arange(f_id+1, args.fold_num)))
            for train_f_id in train_index:
                train_set += sample_list[train_f_id]

        train_set_shuffle = []
        for tss in np.random.permutation(len(train_set)):
            train_set_shuffle.append(train_set[tss])
        test_set = sample_list[-1]

        sub_list_path = os.path.join(graph_list_path, f_name)
        if not os.path.exists(sub_list_path):
            os.makedirs(sub_list_path)

        with open(os.path.join(sub_list_path, 'train'), 'wb') as f:
            pickle.dump({'base_dir': dataset_path,
                         'list': train_set_shuffle}, f)

        if len(val_set):
            with open(os.path.join(sub_list_path, 'val'), 'wb') as f:
                pickle.dump({'base_dir': dataset_path, 'list': val_set}, f)
        if len(test_set):
            with open(os.path.join(sub_list_path, 'test'), 'wb') as f:
                pickle.dump({'base_dir': dataset_path, 'list': test_set}, f)

    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
