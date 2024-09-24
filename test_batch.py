import argparse
import builtins
import os
import sys
import random
import shutil
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from sklearn.metrics import classification_report
from nets.utils import get_model, load_pretrain, load_resume, ExponentialMovingAverage
# from losses.single_center_loss import SingleCenterLoss
from datasets.utils import get_train_dataset, get_val_dataset
# from datasets.imbalanced_sampler import BalanceClassSampler, DistributedSamplerWrapper
# from utils import reduce_tensor, AverageMeter, ProgressMeter, concat_all_gather
from logger import create_logger
import logging


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--val_list', metavar='PATH', type=str, default=None,
                    help='path to val dataset')
parser.add_argument('--val_root', metavar='PATH', type=str, default='',
                    help='path to val dataset')
parser.add_argument('--score_list', metavar='PATH', type=str, default='',
                    help='path to score')

# face dataset
parser.add_argument('--use_face', action='store_true',
                    default=False, help='use face csv dataset')
parser.add_argument('--face_crop_extend_ratio', default=0.3, type=float,
                    help='used for FaceDataset ')

# landmark dataset
parser.add_argument('--use_landmark', action='store_true',
                    default=False, help='use eye csv dataset')
parser.add_argument('--landmark_base_scale', default=0.0, type=float, help='base scale')
parser.add_argument('--landmark_rotate', default=10, type=float, help='random degree')
parser.add_argument('--landmark_translation', default=0.06, type=float, help='random translation')
parser.add_argument('--landmark_scale', default=0.06, type=float, help='random scale')

parser.add_argument('--arch', metavar='ARCH', type=str,
                    help='model architecture')
parser.add_argument('--input_size', default=224, type=int, help='network input size')
parser.add_argument('--num_classes', default=1000, type=int, metavar='N',
                    help='number of dataset classes number')
parser.add_argument('--class_index', default=1, type=int,
                    help='save score class index')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--reset_epoch', action='store_true')
parser.add_argument('--test_crop', action='store_true')
parser.add_argument('--test_five_crop', action='store_true')

# ema models
parser.add_argument('--ema', action='store_true', default=False, help='using ema model')
parser.add_argument('--model_ema_decay', type=float, default=0.9999)
parser.add_argument('--model_ema_steps', type=int, default=32)
parser.add_argument('--protocol', type=str,
                        help='choose protocol')

def main(args):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    time.sleep(2) # ensure dir create success
    create_logger(output_dir='/tmp', dist_rank=dist.get_rank())

    if int(os.environ["LOCAL_RANK"]) != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    logging.info('params: {}'.format(args))
    logging.info('world_size:{}, rank:{}, local_rank:{}'.format(world_size, rank, int(os.environ["LOCAL_RANK"])))

    model = None
    model_ema = None
    logging.info("=> creating model '{}'".format(args.arch))
    arch = args.arch
    model = get_model(args.arch, args.num_classes)

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    if args.ema:
        model_ema = ExponentialMovingAverage(model, device="cuda", decay=args.model_ema_decay)
        logging.info('use ema')

    if args.resume:
        load_resume(args, model, model_ema)
    if model_ema is not None:
        model = model_ema

    # validate dataset1
    val_dataset = get_val_dataset(args.val_root, args.val_list, args.input_size, args, return_path=True)
    logging.info('val_dataset: {}'.format(len(val_dataset)))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size//world_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
    logging.info('val_loader:{}'.format(len(val_loader)))

    model.eval()

    lines = []
    with torch.no_grad():
        for i, (images, target, paths) in enumerate(val_loader):
            if i % 10 == 0:
                print(f'batch {i}/{len(val_loader)}')
            images = images.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)
            target = target.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)

            if args.test_five_crop:
                bs, nc, c, h, w = images.size()
                outputs = model(images.view(-1, c, h, w))
                outputs = outputs.view(bs, nc, -1).mean(1)
            elif arch.startswith("mobilenet") or arch.startswith("shufflenet"):
                outputs = model(images)
            else:
                _, outputs = model(images)

            scores = torch.softmax(outputs, dim=1).data.cpu().numpy()[:,args.class_index]
            for path, score in zip(paths, scores):
                lines.append(f'{path} {score:.6f}\n')


    if int(os.environ["LOCAL_RANK"]) == 0:
        save_dir = os.path.dirname(args.score_list)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(args.score_list, 'w') as fid:
            fid.writelines(lines)
            print(f'write to {args.score_list}')


if __name__ == '__main__':
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    main(args)
