'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models

from custom_gen_dataset import CustomDataset

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import tqdm

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',help='number of data loading workers')
# Optimization options
parser.add_argument('--root_path', default='/mnt/blob/datasets/generate_results/stylegan/', type=str, help='the input dataset path')
parser.add_argument('--train_file', default='/mnt/blob/datasets/generate_results/stylegan_trainlist.txt', type=str, help='train file list path')
parser.add_argument('--test_file', default='/mnt/blob/datasets/generate_results/stylegan_testlist.txt', type=str, help='test file list path')

parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn_mulcla')
parser.add_argument('--train_size', type=int, default=192, help='size of training')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--add_gt', type=int, default=0, help='if we use gt for training')
parser.add_argument('--num_class', type=int, default=8)
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

if args.add_gt == 1:
    args.add_gt = True
else:
    args.add_gt = False

# Validate dataset

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_loss = 1  # best test accuracy

def main():
    global best_loss
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset ')
    # if args.dataset == 'cifar10':
    #     dataloader = datasets.CIFAR10
    #     num_classes = 10
    # else:
    #     dataloader = datasets.CIFAR100
    #     num_classes = 100


    trainset = CustomDataset(root_path=args.root_path, train_file=args.train_file, test_file=args.test_file, train_size=args.train_size, isTrain=True, if_gt=args.add_gt)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = CustomDataset(root_path=args.root_path, train_file=args.train_file, test_file=args.test_file, train_size=args.train_size, isTrain=False, if_gt=args.add_gt)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    # if args.arch.startswith('resnext'):
    #     model = models.__dict__[args.arch](
    #                 cardinality=args.cardinality,
    #                 num_classes=num_classes,
    #                 depth=args.depth,
    #                 widen_factor=args.widen_factor,
    #                 dropRate=args.drop,
    #             )
    # elif args.arch.startswith('densenet'):
    #     model = models.__dict__[args.arch](
    #                 num_classes=num_classes,
    #                 depth=args.depth,
    #                 growthRate=args.growthRate,
    #                 compressionRate=args.compressionRate,
    #                 dropRate=args.drop,
    #             )
    # elif args.arch.startswith('wrn'):
    #     model = models.__dict__[args.arch](
    #                 num_classes=num_classes,
    #                 depth=args.depth,
    #                 widen_factor=args.widen_factor,
    #                 dropRate=args.drop,
    #             )
    # elif args.arch.endswith('resnet'):
    #     model = models.__dict__[args.arch](
    #                 num_classes=num_classes,
    #                 depth=args.depth,
    #                 block_name=args.block_name,
    #             )
    # else:
    #     model = models.__dict__[args.arch](num_classes=num_classes)

    model = models.__dict__[args.arch](num_classes=args.num_class, input_size = args.train_size)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    

    # criterion = torch.nn.MSELoss()

    criterion = torch.nn.BCELoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'gen_image-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f' % (test_loss))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        start = time.time()
        train_loss = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        end = time.time()
        print("training time cost "+str(int(end-start))+"s")
        start = time.time()
        test_loss = test(testloader, model, criterion, epoch, use_cuda)
        end = time.time()
        print("test time cost "+str(int(end-start))+"s")

        # append logger file
        logger.append([state['lr'], train_loss, test_loss])

        # save model
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': test_loss,
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best loss:')
    print(best_loss)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    print("training epoch "+str(epoch)+" -------------- ")
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(trainloader))
    for batch_idx, item in enumerate(trainloader):
        inputs = item['image']
        targets = item['label']
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets.type_as(outputs))

        # measure accuracy and record loss

        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
        #             batch=batch_idx + 1,
        #             size=len(trainloader),
        #             data=data_time.avg,
        #             bt=batch_time.avg,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg,
        #             top1=top1.avg,
        #             top5=top5.avg,
        #             )
        # bar.next()
    # bar.finish()
        
        if batch_idx % 100 == True:
            print(" Loss: {loss:.4f}".format(loss=losses.avg))
    return losses.avg

def test(testloader, model, criterion, epoch, use_cuda):
    print("test epoch "+str(epoch)+" -------------- ")
    global best_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(testloader))

    file_path = "/mnt/blob/datasets/generation_results/score_results/ffhq/m2/ffhq2cat_192_8bin_center.txt"
    with open(file_path, 'wt') as f:

        for batch_idx, item in enumerate(testloader):
            print(batch_idx)
            inputs = item['image']
            targets = item['label']
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    
        # compute output
            with torch.no_grad():
                outputs = model(inputs, split_patch = False)
        
            # image_file = open("/mnt/blob/datasets/FFHQ/selected1.txt")
            image_file = open(args.test_file)
            image_list = image_file.readlines()

            outputs_sum = outputs.sum(1)

            for index in range(0,100):
                a = float(outputs_sum[index])
                a = int(1000000*a)
                ori_image_name = image_list[100*batch_idx+index].replace("\n", "")
                c = str(a) + "_" + ori_image_name
                f.write(c)
                f.write("\n")



        loss = criterion(outputs, targets.type_as(outputs))

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        # top1.update(prec1.item(), inputs.size(0))
        # top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
        #             batch=batch_idx + 1,
        #             size=len(testloader),
        #             data=data_time.avg,
        #             bt=batch_time.avg,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg,
        #             top1=top1.avg,
        #             top5=top5.avg,
        #             )
        # bar.next()
    # bar.finish()
    
    print(" test loss: {loss:.4f}".format(loss=losses.avg))
    return losses.avg

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
