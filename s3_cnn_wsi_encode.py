#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2020/12
# author:Yushan Zheng
# emai:yszheng@buaa.edu.cn

import sys
sys.path.append('./thirdparty')

from torch.utils.data import DistributedSampler
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.utils.data
import torch.multiprocessing as mp
import torch.optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import os
import argparse
import builtins
import shutil
import time
import random
import warnings
import pickle
import cv2
import numpy as np
from yacs.config import CfgNode

from loader import SlideLocalTileDataset
from efficientnet_pytorch import EfficientNet

from utils import *

parser = argparse.ArgumentParser('Extract cnn freatures of whole slide images')
parser.add_argument('--cfg', type=str, default='',
                    help='The path of yaml config file')
parser.add_argument('--slide-dir', type=str,
                    default='/media/disk1/medical/gastric_slides')
parser.add_argument('--slide-list', type=str, default='',
                    help='The list of slide guids used in the dataset')
parser.add_argument('--fold', type=int, default=0,
                    help='To identify the cnn used for feature extraction.\
                         a value -1 identify the cnn trained by all the training set data.\
                         It is useless when pretrained is set as True')

parser.add_argument('--arch', type=str, default='efficientnet-b0')
parser.add_argument('--num-classes', default=6, type=int, metavar='N',
                    help='class number of in the dataset')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--fold-num', type=int, default=5)
parser.add_argument('--max-per-class', type=int, default=1000,
                    help='The upper bound of images from one slide.')

parser.add_argument('--positive-ratio', type=float, default=0.5,
                    help='The image with positive pixels above the ratio will be labeled as positive sample')
parser.add_argument('--negative-ratio', type=float, default=0.05,
                    help='The image with negative pixels below the ratio will be labeled as negative sample')
parser.add_argument('--intensity-thred', type=int, default=25,
                    help='The threshold to recognize foreground regions')

parser.add_argument('--tile-size', type=int, default=512,
                    help='The size of local tile')
parser.add_argument('--level', type=int, default=1,
                    help='The layer index of the slide pyramid to sample')
parser.add_argument('--mask-level', type=int, default=3,
                    help='The layer index of the annotation mask')

parser.add_argument('--od-input', action='store_true', default=False)

parser.add_argument('--imsize', type=int, default=224,
                    help='The size of window.')
parser.add_argument('--step', type=int, default=112,
                    help='The step of window sliding for feature extraction.')
parser.add_argument('--batch-size', type=int, default=256)


def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    save_dir = get_feature_path(args)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.num_classes = len(sub_type_map) if args.label_id == 1 else 2

    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print("Use GPU: {} for encoding".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if 'efficientnet' in args.arch:
        if args.pretrained:
            model = EfficientNet.from_pretrained(
                args.arch, num_classes=args.num_classes)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(
                args.arch, num_classes=args.num_classes)
        # image_size = EfficientNet.get_image_size(args.arch)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=args.num_classes)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(os.path.join(get_cnn_path(
            args), 'model_best.pth.tar'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print('Load model time', time.time() - start_time)

    if args.pretrained:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    model.eval()
    inference_module = model.module if args.distributed else model

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

    for s_id, s_guid in enumerate(slide_list):
        if args.distributed:
            # skip the slides the other gpus are working on
            if not s_id % ngpus_per_node == gpu:
                continue

        slide_save_path = os.path.join(get_feature_path(args), s_guid + '.pkl')
        if os.path.exists(slide_save_path):
            print('skip', slide_save_path)
            continue

        slide_path = os.path.join(args.slide_dir, s_guid)
        image_dir = os.path.join(slide_path, scales[args.level])

        tissue_mask = get_tissue_mask(cv2.imread(
            os.path.join(slide_path, 'Overview.jpg')))
        content_mat = cv2.blur(
            tissue_mask, ksize=args.filter_size, anchor=(0, 0))
        content_mat = content_mat[::args.frstep, ::args.frstep] > args.intensity_thred
        content_tile_lefttop = np.transpose(
            np.asarray(np.where(content_mat), np.int32))

        if content_tile_lefttop.shape[0] > 0:
            slide_dataset = SlideLocalTileDataset(image_dir, content_tile_lefttop*args.step, transform,
                                                  args.tile_size, args.imsize)
            slide_loader = torch.utils.data.DataLoader(
                slide_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False)

            print(s_id, s_guid, content_tile_lefttop.shape[0])
            features = []
            probs = []
            for images in slide_loader:
                images = images.cuda(args.gpu, non_blocking=True)
                with torch.no_grad():
                    # TODO: add densenet inference
                    if 'efficientnet' in args.arch:
                        x = inference_module.extract_features(images)
                        ft = inference_module._avg_pooling(x)
                        ft = ft.flatten(start_dim=1)
                        output = inference_module._fc(ft)
                    elif 'densenet' in args.arch:
                        x = inference_module.features(images)
                        x = F.relu(x, inplace=True)
                        ft = F.adaptive_avg_pool2d(x, (1, 1))
                        ft = ft.flatten(start_dim=1)
                        output = inference_module.classifier(ft)
                    else:
                        raise NotImplementedError(
                            "You may need to implement the feature extraction functions for your chosen CNN backbone."
                            )

                    output = torch.softmax(output, dim=1)
                    features.append(ft.cpu().numpy())
                    probs.append(output.cpu().numpy())

            probs = np.concatenate(probs, axis=0)
            features = np.concatenate(features, axis=0)

            prob_mat = np.zeros(
                (content_mat.shape[0], content_mat.shape[1], args.num_classes),
                np.float)
            prob_mat[content_mat] = probs

            with open(slide_save_path, 'wb') as f:
                pickle.dump({'feats': features,
                             'probs': prob_mat,
                             'content': content_mat,
                             'args': args
                             }, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
