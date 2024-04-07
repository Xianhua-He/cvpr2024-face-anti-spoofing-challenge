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
# from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from nets.utils import get_model, load_pretrain, load_resume, ExponentialMovingAverage, auto_resume_helper
from datasets.utils import get_train_dataset, get_val_dataset
from datasets.imbalanced_sampler import BalanceClassSampler, DistributedSamplerWrapper
from torch.utils.data import DataLoader
from datasets.DataLoaderX import DataLoaderX
from utils import reduce_tensor, AverageMeter, ProgressMeter, concat_all_gather
from logger import create_logger
import logging
from losses.single_center_loss import SingleCenterLoss
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# datasets
parser.add_argument('--train_list', metavar='PATH', type=str, default=None,
                    help='path to train dataset')
parser.add_argument('--train_root', metavar='PATH', type=str, default='',
                    help='path to train dataset')
parser.add_argument('--val_list', metavar='PATH', type=str, default=None,
                    help='path to val dataset')
parser.add_argument('--val_root', metavar='PATH', type=str, default='',
                    help='path to val dataset')
parser.add_argument('--imbalanced_sampler', action='store_true',
                    default=False, help='add imbalanced dataset sampler')

# landmark dataset
parser.add_argument('--use_landmark', action='store_true',
                    default=False, help='use eye csv dataset')
parser.add_argument('--landmark_base_scale', default=0.0, type=float, help='base scale')
parser.add_argument('--landmark_rotate', default=10, type=float, help='random degree')
parser.add_argument('--landmark_translation', default=0.06, type=float, help='random translation')
parser.add_argument('--landmark_scale', default=0.06, type=float, help='random scale')

# models
parser.add_argument('--arch', metavar='ARCH', type=str, help='model architecture')
parser.add_argument('--input_size', default=224, type=int, help='network input size')
parser.add_argument('--num_classes', default=1000, type=int, metavar='N',
                    help='number of dataset classes number')
parser.add_argument('--syncbn', action='store_true', default=False, help='syncbn')

# ema models
parser.add_argument('--ema', action='store_true', default=False, help='using ema model')
parser.add_argument('--model_ema_decay', type=float, default=0.99)
parser.add_argument('--model_ema_steps', type=int, default=32)

# lr
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--accumulate_step', default=1, type=int)
parser.add_argument('--schedule', default='40,60', type=str,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
parser.add_argument('--warmup_steps', default=0, type=int)
parser.add_argument('--total_steps', default=0, type=int)

# train
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs to adjust lr')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, help='total batch size')

# misc
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--fp16', action='store_true')

parser.add_argument('--save_freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 1)')
parser.add_argument('--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrain', default=None, type=str, metavar='PATH',
                    help='use pre-trained model')
parser.add_argument('--autoresume', action='store_true')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--saved_model_dir', default=None, type=str, metavar='PATH',
                    help='saved model dir')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test_crop', action='store_true')
parser.add_argument('--test_five_crop', action='store_true')

# loss
parser.add_argument('--single_center_loss', action='store_true',
                        help='use single center loss')
parser.add_argument('--single_center_loss_weight', default=0.001, type=float)

parser.add_argument('--protocol', type=str, help='choose protocol')
parser.add_argument('--live_weight', default=1.0, type=float, help='live sample cross entropy loss weight')

best_acc1 = 0

def main(args):
    global best_acc1

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if rank == 0:
        if not os.path.exists(args.saved_model_dir):
            os.makedirs(args.saved_model_dir)
        # writer = SummaryWriter(log_dir=args.saved_model_dir)
    time.sleep(3) # ensure dir create success
    create_logger(output_dir=args.saved_model_dir, dist_rank=dist.get_rank())

    if int(os.environ["LOCAL_RANK"]) != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    logging.info('params: {}'.format(args))
    logging.info('world_size:{}, rank:{}, local_rank:{}'.format(world_size, rank, int(os.environ["LOCAL_RANK"])))

    # 设置一个随机数种子
    seed_value = 240
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info('seed_value: {}'.format(seed_value))
    def worker_init_fn(worker_id):
        np.random.seed(seed_value + worker_id)


    # train dataset
    train_dataset = get_train_dataset(args.train_root, args.train_list, args.input_size, args)
    logging.info('train_dataset:{}'.format(len(train_dataset)))

    if args.imbalanced_sampler:
        logging.info('with imbalanced sampler')
        train_sampler = DistributedSamplerWrapper(BalanceClassSampler(train_dataset, mode='upsampling'))
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoaderX(
        local_rank=int(os.environ["LOCAL_RANK"]), dataset=train_dataset, batch_size=args.batch_size//world_size,
        shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)
    logging.info('train_loader:{}'.format(len(train_loader)))

    # validate dataset
    val_dataset = get_val_dataset(args.val_root, args.val_list, args.input_size, args)
    logging.info('val_dataset: {}'.format(len(val_dataset)))

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoaderX(
        local_rank=int(os.environ["LOCAL_RANK"]), dataset=val_dataset, batch_size=args.batch_size//world_size,
        shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    logging.info('val_loader:{}'.format(len(val_loader)))

    args.warmup_steps = len(train_loader) * args.warmup_epochs
    args.total_steps = len(train_loader) * args.epochs
    logging.info(f'warmup_steps: {args.warmup_steps}')
    logging.info(f'total_steps: {args.total_steps}')

    # create model
    model = None
    model_ema = None
    logging.info("=> creating model '{}', fp16:{}".format(args.arch, args.fp16))
    model = get_model(args.arch, args.num_classes, args.fp16)

    if args.pretrain:
        load_pretrain(args.pretrain, model)

    model.cuda()
    if args.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('use syncbn')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    criterions = {}

    weights = torch.tensor([args.live_weight, 1.0])
    criterions['criterion_ce'] = nn.CrossEntropyLoss(weight=weights).cuda(int(os.environ["LOCAL_RANK"]))


    # criterion = nn.CrossEntropyLoss().cuda(int(os.environ["LOCAL_RANK"]))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                    eps=1.0e-08, betas=[0.9, 0.999],
                                    lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise 'unkown optimizer'

    scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=400)

    if args.ema:
        model_ema = ExponentialMovingAverage(model, device="cuda", decay=args.model_ema_decay)
        logging.info('use ema')

    if args.autoresume:
        resume_file = auto_resume_helper(args.saved_model_dir)
        if resume_file:
            if args.resume:
                logging.warning(f"auto-resume changing resume file from {args.resume} to {resume_file}")
            args.resume = resume_file
            logging.info(f'auto resuming from {resume_file}')
        else:
            logging.info(f'no checkpoint found in {args.saved_model_dir}, ignoring auto resume')

    if args.resume:
        best_acc1 = load_resume(args, model, model_ema, optimizer, scaler)

    if args.evaluate:
        # validate(val_loader, model, model_ema, criterion, args)
        validate(val_loader, model, model_ema, criterions, args)
        return


    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss, train_acc1 = train(train_loader, model, model_ema, criterions, optimizer, epoch, scaler, args)

        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            val_loss, val_acc1 = validate(val_loader, model, model_ema, criterions, args)
            if rank == 0:
                state = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'state_dict_ema': model_ema.state_dict() if model_ema is not None else None,
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    'scaler' : scaler.state_dict()
                }
                if not os.path.exists(args.saved_model_dir):
                    os.makedirs(args.saved_model_dir)
                ema_str = 'ema_' if args.ema else ''
                save_path = os.path.join(args.saved_model_dir, f'{ema_str}{args.arch}_epoch{epoch:0>3d}_acc1_{val_acc1:.4f}.pth')
                torch.save(state, save_path)

                if val_acc1 > best_acc1:
                    best_acc1 = val_acc1
                    shutil.copyfile(save_path, os.path.join(args.saved_model_dir, f'{ema_str}{args.arch}_bestacc1_{best_acc1:.4f}_epoch{epoch:0>3d}.pth'))

                #writer.add_scalar('Train_Loss/epochs', train_loss, epoch)
                #writer.add_scalar('Train_acc1/epochs', train_acc1, epoch)
                #writer.add_scalar('Val_Loss/epochs', val_loss, epoch)
                #writer.add_scalar('Val_acc1/epochs', val_acc1, epoch)
                #writer.add_scalar('lr/epochs', lr, epoch)
    dist.destroy_process_group()


def train(train_loader, model, model_ema, criterions, optimizer, epoch, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ce = AverageMeter('Loss_ce', ':.5f')
    losses_scl = AverageMeter('Loss_scl', ':.5f')

    losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, top1, losses_ce, losses_scl, losses] if args.single_center_loss 
            else [batch_time, data_time, top1, losses_ce, losses],)

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        global_step = epoch * len(train_loader) + i
        lr = adjust_learning_rate(optimizer, global_step, epoch, args)

        images = images.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)
        target = target.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)

        feats, output = model(images)
        loss_ce = criterions['criterion_ce'](output, target)
        if args.single_center_loss:
            if 'criterion_scl' not in criterions:
                D = feats.shape[1]
                # logger.info(f'Create single center loss, features dim: {D}')
                criterions['criterion_scl'] = SingleCenterLoss(m=0.3, D=D, use_gpu=True)
            loss_scl = criterions['criterion_scl'](feats, target)
            loss = loss_ce + args.single_center_loss_weight * loss_scl

            losses_scl.update(loss_scl.item(), images.size(0))
        else:
            loss = loss_ce

        # with torch.cuda.amp.autocast(args.fp16):
        #     output = model(images)
        #     loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))
        top1.update(acc1[0], images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses.update(loss.item(), images.size(0))

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        loss /= args.accumulate_step
        if args.fp16:
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulate_step == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (i + 1) % args.accumulate_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
        if (i + 1) % args.accumulate_step == 0:
            optimizer.zero_grad()

        if args.ema and (global_step + 1) % args.model_ema_steps == 0:
            model_ema.update_parameters(model)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info(f'epoch {epoch} lr {lr:.6f} ' + progress.display(i))
    return losses.avg, top1.avg



def validate(val_loader, model, model_ema, criterions, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses_ce = AverageMeter('Loss_ce', ':.5f')
    # losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses_ce, top1])

    # used for classification_report
    y_true = []
    y_pred = []

    if model_ema is not None:
        model_ema.eval()
    else:
        model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            images = images.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)
            target = target.cuda(int(os.environ["LOCAL_RANK"]), non_blocking=True)

            if model_ema is not None:
                _, output = model_ema(images)
            else:
                _, output = model(images)

            # concat multi-gpu data
            output = concat_all_gather(output)
            target = concat_all_gather(target)

            # measure accuracy and record loss
            loss_ce = criterions['criterion_ce'](output, target)
            # losses_ce.update(loss_ce.item(), output.size(0))
            # loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))
            losses_ce.update(loss_ce.item(), output.size(0))
            # losses.update(loss.item(), output.size(0))
            top1.update(acc1[0], output.size(0))

            # for classification_report
            y_true.extend(target.cpu().to(torch.int64).numpy())
            _, preds = output.topk(1, 1, True, True)
            y_pred.extend(preds.cpu().to(torch.int64).numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        logging.info(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

        # report = classification_report(y_true, y_pred, target_names=['0_活体', '1_攻击'], output_dict=True)
        report = classification_report(y_true, y_pred, output_dict=True)
        logging.info('{}'.format(report))

    return losses_ce.avg, top1.avg
    # return losses.avg, top1.avg


def save_checkpoint(state, save_dir ,filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)


def adjust_learning_rate(optimizer, global_step, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if global_step < args.warmup_steps:
        minlr = lr * 0.01
        lr = minlr + (lr - minlr) * global_step / (args.warmup_steps - 1)
    else:
        if args.cos:
            lr *= 0.5 * (1. + math.cos(math.pi * (global_step - args.warmup_steps) / (args.total_steps - args.warmup_steps)))
        else:  # stepwise lr schedule
            milestones = args.schedule.split(',')
            milestones = [int(milestone) for milestone in milestones]
            for milestone in milestones:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
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


if __name__ == '__main__':
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    main(args)
