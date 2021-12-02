#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import numpy as np
import pickle
import os
import cv2
import argparse
from multiprocessing import Pool
from yacs.config import CfgNode
from utils import *


parser = argparse.ArgumentParser('Sampling patches for CNN trianing')
parser.add_argument('--cfg', type=str, default='',
                    help='The path of yaml config file')
parser.add_argument('--slide-dir', type=str, default='',
                    help='The path of slide data')
parser.add_argument('--slide-list', type=str, default='',
                    help='The list of slide guids used in the dataset')

parser.add_argument('--tile-size', type=int, default=512,
                    help='The size of local tile')
parser.add_argument('--level', type=int, default=1,
                    help='The layer index of the slide pyramid to sample')
parser.add_argument('--mask-level', type=int, default=3,
                    help='The layer index of the annotation mask')
parser.add_argument('--save-mask', action='store_true',
                    help='Save a mask for each sample if it is set as True.\
                        It is designed for image segmentation tasks.')

parser.add_argument('--test-ratio', type=float, default=0.3,
                    help='The ratio of slides used for testing')
parser.add_argument('--fold-num', type=int, default=5,
                    help='The number of folds for cross-validation')

parser.add_argument('--positive-ratio', type=float, default=0.5,
                    help='The image with positive pixels above the ratio will be labeled as positive sample')
parser.add_argument('--negative-ratio', type=float, default=0.05,
                    help='The image with negative pixels below the ratio will be labeled as negative sample')
parser.add_argument('--intensity-thred', type=int, default=25,
                    help='The threshold to recognize foreground regions')

parser.add_argument('--imsize', type=int, default=224,
                    help='The size of patch images')
parser.add_argument('--sample-step', type=int, default=112,
                    help='The minimal pixel interval of two patches in the sampling process')

parser.add_argument('--max-per-class', type=int, default=1000,
                    help='The upper bound of images per class from one slide.')
parser.add_argument('--num-workers', type=int, default=20,
                    help='The processors used for parallel sampling.')


def main(args):
    np.random.seed(1)
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

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

    args.dataset_path = get_sampling_path(args)
    print('slide num', len(slide_list))

    sampling_list = [(i, args) for i in slide_list]
    if args.num_workers < 2:
        # sampling the data using single thread
        for s in slide_list:
            sampling_slide((s, args))
    else:
        # sampling the data in parallel
        with Pool(args.num_workers) as p:
            p.map(sampling_slide, sampling_list)

    list_path = get_data_list_path(args)
    dataset_split_path = os.path.join(list_path, 'split.pkl')
    if not os.path.exists(dataset_split_path):
        np.random.shuffle(slide_list)
        test_list = slide_list[:int(len(slide_list)*args.test_ratio)]
        train_list = slide_list[int(len(slide_list)*args.test_ratio):]

        folds = []
        for f_id in range(args.fold_num):
            folds.append(train_list[f_id::args.fold_num])
        folds.append(test_list)

        if not os.path.exists(list_path):
            os.makedirs(list_path)
        with open(dataset_split_path, 'wb') as f:
            pickle.dump(folds, f)

    make_list(args)

    return 0


def sampling_slide(slide_info):
    slide_guid = slide_info[0]
    args = slide_info[1]

    time_file_path = os.path.join(args.dataset_path, slide_guid, 'info.txt')
    if os.path.exists(time_file_path):
        print(slide_guid, 'is already sampled. skip.')
        return 0

    slide_path = os.path.join(args.slide_dir, slide_guid)
    image_dir = os.path.join(slide_path, scales[args.level])
    mask = cv2.imread(os.path.join(slide_path, 'AnnotationMask.png'), 0)

    Overview = cv2.imread(os.path.join(slide_path, 'Overview.jpg'))
    tissue_mask = get_tissue_mask(Overview)
    content_mat = cv2.blur(tissue_mask, ksize=args.filter_size, anchor=(0, 0))
    content_mat = content_mat[::args.srstep, ::args.srstep]

    positive_mat = cv2.blur(
        (mask > 0)*255, ksize=args.filter_size, anchor=(0, 0))
    positive_mat = positive_mat[::args.srstep, ::args.srstep]

    # the left-top position of benign patches
    bn_lt = np.transpose(
        np.asarray(
            np.where((positive_mat < args.negative_ratio * 255)
                     & (content_mat > args.intensity_thred)), np.int32))
    if bn_lt.shape[0] > args.max_per_class:
        bn_lt = bn_lt[np.random.choice(
            bn_lt.shape[0], args.max_per_class, replace=False)]

    if bn_lt.shape[0] > 0:
        slide_save_dir = os.path.join(args.dataset_path, slide_guid, '0')
        if not os.path.exists(slide_save_dir):
            os.makedirs(slide_save_dir)

        extract_and_save_tiles(image_dir, slide_save_dir, bn_lt,
                               args.tile_size, args.imsize, args.sample_step)

    class_list = np.unique(mask[mask > 0])
    for c in class_list:
        class_index_mat = cv2.blur(
            (mask == c)*255, ksize=args.filter_size, anchor=(0, 0))
        class_index_mat = class_index_mat[::args.srstep, ::args.srstep]

        # the left-top position of tumor patches
        tm_lt = np.transpose(
            np.asarray(
                np.where((class_index_mat > args.positive_ratio * 255)
                         & (content_mat > args.intensity_thred)), np.int32))

        if tm_lt.shape[0] > args.max_per_class:
            tm_lt = tm_lt[np.random.choice(
                tm_lt.shape[0], args.max_per_class, replace=False)]

        slide_save_dir = os.path.join(args.dataset_path, slide_guid, str(c))
        if not os.path.exists(slide_save_dir):
            os.makedirs(slide_save_dir)

        extract_and_save_tiles(image_dir, slide_save_dir, tm_lt,
                               args.tile_size, args.imsize, args.sample_step)

        if args.save_mask:
            extract_and_save_masks(mask, slide_save_dir, tm_lt, args)

    with open(time_file_path, 'w') as f:
        f.write('Sampling finished')


def make_list(args, min_file_size=5 * 1024):
    """
    Attributes:
        min_file_size : The minimum size of the jpeg considered in the training.
            5*1024=5Kb: The histopathology image with no substantial content generally 
                        in size of under 5Kb when compressed in jpeg format.
    """
    dataset_path = get_sampling_path(args)
    list_path = get_data_list_path(args)

    dataset_split_path = os.path.join(list_path, 'split.pkl')
    if not os.path.exists(dataset_split_path):
        raise AssertionError('Run sampling function first.')

    with open(dataset_split_path, 'rb') as f:
        folds = pickle.load(f)

    config_path = os.path.join(list_path, 'list_config.csv')
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
            slide_dir = os.path.join(dataset_path, s_guid)
            class_list = os.listdir(slide_dir)
            for c in class_list:
                c_dir = os.path.join(slide_dir, c)
                if os.path.isfile(c_dir):
                    continue
                c_index = sub_type_map[int(c)]
                b_index = binary_map[int(c)]
                class_slide_counter[c_index] += 1

                image_list = os.listdir(c_dir)
                image_list_tmp = []
                if c_index == 0 & len(image_list) > args.max_per_class:
                    for use_img in np.random.permutation(args.max_per_class):
                        image_list_tmp.append(image_list[use_img])
                    image_list = image_list_tmp

                for img in image_list:
                    if img[-3:] == 'jpg':
                        img_path = os.path.join(c_dir, img)
                        # The file size of jpeg image
                        if os.path.getsize(img_path) < min_file_size:
                            continue

                        sample_str = [os.path.join(s_guid, c, img),
                                      c_index, b_index]
                        if args.save_mask:
                            sample_str.append(os.path.join(
                                s_guid, c, img[:-4] + '_mask.png'))
                        sample_str.append(slide_count + s_id)
                        sample_list_fold.append(sample_str)
                        class_image_counter[c_index] += 1

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

        sub_list_path = os.path.join(list_path, f_name)
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


def extract_and_save_tiles(image_dir, slide_save_dir, position_list, tile_size,
                           imsize, step):
    print(slide_save_dir, position_list.shape[0])
    for pos in position_list:
        img = extract_tile(image_dir, tile_size, pos[1] * step, pos[0] * step,
                           imsize, imsize)
        if len(img) > 0:
            cv2.imwrite(
                slide_save_dir + '/{:04d}_{:04d}.jpg'.format(pos[1], pos[0]),
                img)


def extract_and_save_masks(slide_mask, slide_save_dir, position_list, args):
    for pos in position_list:
        y = pos[0] * args.rstep
        x = pos[1] * args.rstep
        img = slide_mask[y:(y + args.msize), x:(x + args.msize)]
        if len(img) > 0:
            img = cv2.resize(img, (args.imsize, args.imsize),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                slide_save_dir +
                '/{:04d}_{:04d}_mask.png'.format(pos[1], pos[0]), img*50)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
