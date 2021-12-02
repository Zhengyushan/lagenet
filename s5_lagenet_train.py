#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2021/11
# author:yushan zheng
# emai:yszheng@buaa.edu.cn

import argparse
import os
import pickle
import shutil
import time
import numpy as np
from yacs.config import CfgNode

from tabulate import tabulate
from sklearn import metrics

import torch
import torch.nn.functional as F

from model import LAGENet
from loader import LaGraphLoader
from utils import *


def arg_parse():
    parser = argparse.ArgumentParser(description='GCN-Hash arguments.')

    parser.add_argument('--cfg', type=str,
            default='',
            help='The path of yaml config file')
    parser.add_argument('--dataset', type=str,
                        default='')
    parser.add_argument('--result-dir', type=str, default='./data')
    parser.add_argument('--fold', type=int, default=-1, help='use all data for training if it is set -1')

    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers to load data.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0.01,
                        help='Weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum.')
    parser.add_argument('--shuffle-train', default=False, action='store_true')

    parser.add_argument('--label-id', type=int, default=1,
                        help='1: five categoriy task, 2: binary task')

    parser.add_argument('--eval-model', type=str, default='',
                        help='provide a path of a trained model to evaluate the performance')
    parser.add_argument('--eval-freq', type=int, default=20)
    parser.add_argument('--print-freq', type=int, default=10)

    parser.add_argument('--max-nodes', type=int, default=128,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')

    # Ablation settings
    parser.add_argument('--disable-adj', action='store_true', default=False)
    parser.add_argument('--disable-distance', action='store_true', default=False)
    parser.add_argument('--node-aug', action='store_true', default=False)

    # LAGE-Net config
    parser.add_argument('--lage-depth', type=int, default=None)
    parser.add_argument('--lage-heads', type=int, default=None)
    parser.add_argument('--lage-dim', type=int, default=None)
    parser.add_argument('--lage-mlp-dim', type=int, default=None)
    parser.add_argument('--lage-dim-head', type=int, default=None)
    parser.add_argument('--hash-bits', type=int, default=None)

    parser.add_argument('--redo', default=False, action='store_true')

    return parser.parse_args()


def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    args.num_classes = len(sub_type_map) if args.label_id == 1 else 2
    graph_model_path = get_lage_path(args)
    print(graph_model_path)

    if not args.redo and os.path.exists(graph_model_path + '.csv')\
            and (len(args.eval_model) == 0):
        print(graph_model_path, 'exists. skip')
        return 0
    
    if not os.path.exists(graph_model_path):
        os.makedirs(graph_model_path)

    graph_list_dir = os.path.join(get_graph_list_path(args), args.fold_name)
    # train graph data
    train_set = LaGraphLoader(
            os.path.join(graph_list_dir, 'train'),
            max_node_number=args.max_nodes,
            task_id=args.label_id,
            disable_adj=args.disable_adj,
            dist_embed_dim= args.lage_dim
            )

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        train_set.get_weights(), len(train_set), replacement=True
        )

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=args.shuffle_train,
            num_workers=args.num_workers, sampler=train_sampler)
    dataset_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=args.shuffle_train,
            num_workers=args.num_workers, sampler=None)
    # validation graph data
    val_path = os.path.join(graph_list_dir, 'val')
    if not os.path.exists(val_path):
        valid_loader = None
    else:
        valid_set = LaGraphLoader(val_path,
            max_node_number=args.max_nodes,
            task_id=args.label_id,
            disable_adj=args.disable_adj,
            dist_embed_dim= args.lage_dim
            )

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None
            )
        
    # test graph data
    test_path = os.path.join(graph_list_dir, 'test')
    if not os.path.exists(test_path):
        test_loader = None
    else:
        test_set = LaGraphLoader(test_path,
            max_node_number=args.max_nodes,
            task_id=args.label_id,
            disable_adj=args.disable_adj,
            dist_embed_dim= args.lage_dim
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None
            )

    args.input_dim = train_set.get_feat_dim()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LAGENet(
        num_patches=args.max_nodes,
        patch_dim=args.input_dim,
        num_classes=args.num_classes, 
        num_hash_bits=args.hash_bits,
        dim=args.lage_dim, 
        depth=args.lage_depth, 
        heads=args.lage_heads, 
        mlp_dim=args.lage_mlp_dim, 
        dim_head=args.lage_dim_head, 
        pos_embed=args.disable_distance,
        pool = args.lage_pool, 
    )

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        print('Do not support multi-GPU training.')

    if args.eval_model and valid_loader is not None:
        model_params = torch.load(args.eval_model, map_location='cpu')
        model.load_state_dict(model_params['state_dict'])

        label, predict = dataset_inference(valid_loader, model, args)
        accs = accuracy(predict, label)

        label = label.numpy()
        predict = predict.numpy()

        confuse_mat = metrics.confusion_matrix(label, np.argmax(predict, axis=1))
        confuse_mat = np.asarray(confuse_mat,np.float)
        for y in range(args.num_classes):
            nc = np.sum(label==y) if np.sum(label==y)>0 else 1
            confuse_mat[y,:] = confuse_mat[y,:]/nc

        with open(os.path.join(graph_model_path, 'eval.csv'), 'w') as f:
            for cn in range(confuse_mat.shape[0]):
                f.write(',{:.3f}'.format(confuse_mat[cn,cn]))
            f.write('\n')

        with open(os.path.join(graph_model_path,  'eval.pkl'), 'wb') as f:
            pickle.dump({'acc':accs[0].numpy(), 'cm':confuse_mat, 'num':np.bincount(label)}, f)

        return 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.7)

    with open(graph_model_path + '.csv', 'a') as f:
        f.write('epc, T, acc, mic, mac, V, val, mic, mac,  p@5, p@20, MAP, MRR, R5, T, acc, mic, mac,  p@5, p@20, MAP, MRR, R5, SUB\n')

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        train(train_loader, model, optimizer, epoch, args)
        scheduler.step()

        best_acc1 = 0
        if (epoch+1) % args.eval_freq == 0:
            train_acc, train_preds, train_codes, train_labels = dataset_inference(dataset_loader, model, args)
            val_acc,val_preds, val_codes, val_labels = dataset_inference(valid_loader, model, args)
            test_acc, test_preds, test_codes, test_labels = dataset_inference(test_loader, model, args)

            # evaluate classification
            _, train_auc = evaluate_cls(train_preds, train_labels, args.num_classes, prefix='Train')
            _, val_auc = evaluate_cls(val_preds, val_labels, args.num_classes, prefix='Val')
            test_cm, test_auc = evaluate_cls(test_preds, test_labels, args.num_classes, prefix='Test')
            
            # evaluate retrieval
            val_rm = evaluate_ret(val_codes[::10], val_labels[::10], train_codes, train_labels, prefix='Val')
            test_rm = evaluate_ret(test_codes[::10], test_labels[::10], train_codes, train_labels, prefix='Test')

            with open(graph_model_path + '.csv', 'a') as f:
                f.write('{:02d}, T, {:.3f},{:.3f},{:.3f}, V, {:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}, T, {:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}, SUB'.format(
                    epoch, 
                        train_acc, train_auc['micro'], train_auc['macro'],
                        val_acc, val_auc['micro'], val_auc['macro'], val_rm['ap5'], val_rm['ap20'], val_rm['map'], val_rm['mrr'], val_rm['r5'],
                        test_acc, test_auc['micro'], test_auc['macro'], test_rm['ap5'], test_rm['ap20'], test_rm['map'], test_rm['mrr'], test_rm['r5'],
                        ))
                for cn in range(test_cm.shape[0]):
                    f.write(',{:.3f}'.format(test_cm[cn, cn]))
                f.write('\n')

            model_path = os.path.join(graph_model_path, 'lage_model_ep{}.pth.tar'.format(epoch + 1))
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, model_path)

            shutil.copy(model_path, os.path.join(graph_model_path, 'checkpoint.pth.tar'))
            code_path = os.path.join(graph_model_path, 'lage_code_ep{}.pkl'.format(epoch + 1))
            with open(code_path, 'wb') as f:
                pickle.dump(
                    {'train_code':train_codes, 'val_code':val_codes, 'test_code':test_codes,
                     'train_label':train_labels, 'val_label':val_labels, 'test_label':test_labels,}, f
                )

def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    cls_losses = AverageMeter('Loss1', ':.4e')
    hash_losses = AverageMeter('Loss2', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, cls_losses, hash_losses, top1,
                             top2, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            feats = data[0].float().cuda(args.gpu, non_blocking=True)
            adj = data[1].float().cuda(args.gpu, non_blocking=True)
            de = data[2].float().cuda(args.gpu, non_blocking=True)
            masks = data[3].int().cuda(args.gpu, non_blocking=True)
            target = data[4].cuda(args.gpu, non_blocking=True)
            one_hot = torch.zeros(feats.size(0), args.num_classes).cuda(args.gpu).scatter_(1, target.unsqueeze(1), 1)

        # compute output
        preds, hash_codes = model(feats, adj, de, masks)
        '''
            Here, we append a classification loss function to accelerate 
            the training of the network and meanwhile enable it to identify the tissue.
        '''
        cls_loss = cls_loss_func(preds, target)
        hash_loss = hash_loss_func(hash_codes, one_hot, model.get_weights())
        loss = hash_loss + cls_loss

        # measure accuracy and record loss
        acc1, acc2 = accuracy(preds, target, topk=(1, 2))
        cls_losses.update(cls_loss.item(), feats.size(0))
        hash_losses.update(hash_loss.item(), feats.size(0))
        top1.update(acc1[0], feats.size(0))
        top2.update(acc2[0], feats.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def evaluate_cls(y_preds, y_labels, num_classes=None, prefix='Eval'):
    if num_classes is None:
        num_classes = max(y_labels) + 1
    auc = multi_auc(y_labels, y_preds, args.num_classes)

    y_labels = y_labels.numpy()
    y_preds = y_preds.numpy()
    confuse_mat = metrics.confusion_matrix(
        y_labels, np.argmax(y_preds, axis=1))
    confuse_mat = np.asarray(confuse_mat, float)

    values = [prefix, auc['micro'], auc['macro']]
    headers = ['Classification', 'micro auc', 'macro auc']
    for y in range(max(y_labels)+1):
        confuse_mat[y, :] = confuse_mat[y, :]/np.sum(y_labels == y)
        values.append(confuse_mat[y, y])
        headers.append(str(y))

    print(tabulate([values,], headers, tablefmt="grid"))

    return confuse_mat, auc


def evaluate_ret(query_code, query_label, db_code, db_label, prefix='Eval'):
    _, correct = retrieval(query_code, query_label, db_code, db_label)
    correct = correct.numpy()
    ap_5 = average_precision(correct, ret_num=5)
    ap_20 = average_precision(correct, ret_num=20)
    mAP = mean_average_precision(correct)[-1]
    mRR = mean_reciprocal_rank(correct)
    r5 = recall_at_n(correct, ret_num=5)

    print(tabulate([[prefix, ap_5, ap_20, mAP, mRR, r5]],
                   headers=['CBIR', 'p@5', 'p@20', 'MAP', 'MRR', 'R5'], tablefmt="grid")
          )

    return {'ap5': ap_5, 'ap20': ap_20, 'map': mAP, 'mrr': mRR, 'r5': r5}


def dataset_inference(dataset_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    cls_losses = AverageMeter('Loss1', ':.4e')
    hash_losses = AverageMeter('Loss2', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(dataset_loader), batch_time, cls_losses, hash_losses, top1, top2,
                             prefix='Inference: ')

    # switch to evaluate mode
    model.train()
    y_preds = []
    y_labels = []
    y_codes = []
    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataset_loader):
            if args.gpu is not None:
                feats = data[0].float().cuda(args.gpu, non_blocking=True)
                adj = data[1].float().cuda(args.gpu, non_blocking=True)
                de = data[2].float().cuda(args.gpu, non_blocking=True)
                masks = data[3].int().cuda(args.gpu, non_blocking=True)
                target = data[4].cuda(args.gpu, non_blocking=True)
                one_hot = torch.zeros(feats.size(0), args.num_classes).cuda(args.gpu).scatter_(1, target.unsqueeze(1), 1)

            # compute output
            preds, hash_codes = model(feats, adj, de, masks)
            cls_loss = cls_loss_func(preds, target)
            hash_loss = hash_loss_func(hash_codes, one_hot, model.get_weights())

            y_preds.append(preds.cpu().data)
            y_labels.append(target.cpu().data)
            y_codes.append(hash_codes.cpu().data)

            # measure accuracy and record loss
            acc1, acc2 = accuracy(preds, target, topk=(1, 2))
            cls_losses.update(cls_loss.item(), feats.size(0))
            hash_losses.update(hash_loss.item(), feats.size(0))
            top1.update(acc1[0], feats.size(0))
            top2.update(acc2[0], feats.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
              .format(top1=top1, top2=top2))

    y_preds = torch.cat(y_preds)
    y_codes = torch.cat(y_codes)
    y_labels = torch.cat(y_labels)
    
    return top1.avg, y_preds, y_codes, y_labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def cls_loss_func(pred, label):
    return F.cross_entropy(pred, label)


def hash_loss_func(features, label, hash_weights=None):
    hash_bits = features.size()[1]

    cross_sim_mat = torch.matmul(features, features.T) / hash_bits
    cross_label_mat = torch.matmul(label, label.T)
    cross_label_mat[cross_label_mat<1]=-1

    loss = (cross_sim_mat - cross_label_mat).pow(2).mean()
    loss_org = (hash_weights.matmul(hash_weights.T) - torch.eye(hash_bits).to(hash_weights.device)).pow(2).mean()

    return loss + 0.01*loss_org


if __name__ == "__main__":
    args = arg_parse()
    main(args)
