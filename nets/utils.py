import sys
import torch
import torch.nn as nn
import os
import logging
# import torchvision.models as models
import nets.resnet as resnet
import nets.mobilenetv2 as mobilenetv2
import nets.mobilenetv3 as mobilenetv3
# import nets.convnext as convnext
# import nets.coatnet as coatnet
# import nets.maxvit as maxvit
import nets.swin_transformer_v2 as swin_transformer_v2
# import nets.dwspgnet as dwspgnet
# from nets.efficientnet import EfficientNet
# from nets.mobileone import mobileone
from nets.shufflenetv2 import ShuffleNetV2


# model_names = sorted(name for name in resnet.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(resnet.__dict__[name]))

def get_model(arch, num_classes=2, fp16=False):
    if arch.startswith('resnet'):
        model = resnet.__dict__[arch](num_classes=num_classes, fp16=fp16)
    elif arch.startswith('mobilenet_v2'):
        model = mobilenetv2.__dict__[arch](pretrained=False, num_classes=num_classes)
    elif arch.startswith('mobilenet_v3'):
        model = mobilenetv3.__dict__[arch](pretrained=False, num_classes=num_classes)
    elif arch.startswith('efficient'):
        model = EfficientNet.from_name(arch, num_classes=num_classes)
    elif arch.startswith('convnext'):
        model = convnext.__dict__[arch](pretrained=False, num_classes=num_classes, fp16=fp16)
    elif arch.startswith('coatnet'):
        model = coatnet.__dict__[arch](num_classes=num_classes, fp16=fp16)
    elif arch.startswith('max_vit'):
        model = maxvit.__dict__[arch](num_classes=num_classes, fp16=fp16)
    elif arch.startswith('mobileone'): # mobileone-s0 -- mobleone-s4
        variant = arch.split('-')[1] # s0,..,s4, t0, t1, t2
        model = mobileone(variant=variant, num_classes=num_classes)
    elif arch.startswith('swin_v2'):
        model = swin_transformer_v2.__dict__[arch](num_classes=num_classes, fp16=fp16)
    elif arch.startswith('dwspgnet'):
        model = dwspgnet.__dict__[arch](num_classes=num_classes)
    elif arch.startswith('shufflenetv2a'):
        variant = arch.split('_')[1] # 0.5x, 0.75x, 1.0x, 1.5x, 2.0x
        model = ShuffleNetV2(num_classes=num_classes, model_size=variant, use_conv_last=False, use_pooling=False)
    elif arch.startswith('shufflenetv2'):
        variant = arch.split('_')[1] # 0.5x, 0.75x, 1.0x, 1.5x, 2.0x
        model = ShuffleNetV2(num_classes=num_classes, model_size=variant, use_conv_last=True, use_pooling=True)
    else:
        logging.info('arch not supported.')
        sys.exit(-1)
    return model


def load_pretrain(pretrain, model):
    if os.path.isfile(pretrain):
        logging.info("=> loading pretrained weights from '{}'".format(pretrain))
        state_dict = torch.load(pretrain, map_location="cpu")

        if 'model_state' in state_dict.keys():
            state_dict = state_dict['model_state']
        elif 'state_dict_ema' in state_dict.keys() and state_dict['state_dict_ema'] is not None:
            state_dict = state_dict['state_dict_ema']
            logging.info(f'use ema pretrain')
        elif 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict.keys():
            state_dict = state_dict['model']

        if 'swin' in pretrain:
            rpe_mlp_keys = [k for k in state_dict.keys() if "rpe_mlp" in k]
            for k in rpe_mlp_keys:
                state_dict[k.replace('rpe_mlp', 'cpb_mlp')] = state_dict.pop(k)

        if 'moco' in pretrain:
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                del state_dict[k]
        else:
            for k in list(state_dict.keys()):
                if k.startswith('module.module.'):
                    state_dict[k[len("module.module."):]] = state_dict[k]
                    del state_dict[k]
                    k = k[len("module.module."):]
                if k.startswith('module.'):
                    state_dict[k[len("module."):]] = state_dict[k]
                    del state_dict[k]
                    k = k[len("module."):]
                if k.startswith('encoder.'):
                    state_dict[k[len("encoder."):]] = state_dict[k]
                    del state_dict[k]
                    k = k[len("encoder."):]
                if k not in model.state_dict().keys():
                    logging.info('checkpoint key not exist in model {}'.format(k))
                    continue
                if model.state_dict()[k].shape != state_dict[k].shape:
                    logging.info('skip {}, shape dismatch, model: {} vs pretrain:{}'.format(k, model.state_dict()[k].shape, state_dict[k].shape))
                    del state_dict[k]
                    continue

        ret = model.load_state_dict(state_dict, strict=False)
        logging.info('missing_keys:{}'.format(ret.missing_keys))
        logging.info('unexpected_keys:{}'.format(ret.unexpected_keys))
    else:
        logging.info("=> no pretrained weights found at '{}'".format(pretrain))
    return model


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('.pth')]
    logging.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logging.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def load_resume(args, model, model_ema, optimizer=None, scaler=None):
    best_acc1 = 0
    if os.path.isfile(args.resume):
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        loc = 'cuda:{}'.format(int(os.environ["LOCAL_RANK"]))
        checkpoint = torch.load(args.resume, map_location=loc)

        model.load_state_dict(checkpoint['state_dict'])
        logging.info("=>loaded model")

        if model_ema is not None and 'state_dict_ema' in checkpoint:
            model_ema.load_state_dict(checkpoint['state_dict_ema'])
            logging.info("=>loaded ema model")

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            logging.info("=>loaded optimizer")

        if scaler is not None and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            logging.info("=>loaded scaler")

        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
        sys.exit()
    return best_acc1


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)

    def update_parameters(self, model):
        for p_swa, p_model in zip(self.module.state_dict().values(), model.state_dict().values()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device)))
        self.n_averaged += 1
