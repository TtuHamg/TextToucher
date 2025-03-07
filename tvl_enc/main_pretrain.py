# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import sys 
import warnings
warnings.filterwarnings('ignore')
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from timm.data.loader import MultiEpochsDataLoader
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from transformer_utils import handle_flash_attn

from tvl_enc import tvl 
from tvl_enc.tvl import ModalityType
from loss import TVLLoss

from engine_pretrain import train_one_epoch, evaluate
from tacvis import TacVisDataset, TacVisDatasetV2, RGB_AUGMENTS, TAC_AUGMENTS, TAC_AUGMENTS_BG, TAC_AUGMENTS_BG_CJ
from tac_text_dataset import TacTextDataset

import wandb
os.environ["WANDB_MODE"] = "offline"
def get_args_parser():
    parser = argparse.ArgumentParser('Tactile encoder pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')    
    parser.add_argument('--use_tac_text_loss', action='store_true', default=False, help="Use special tactile language loss")
    parser.add_argument('--tactile_model', type=str, default='resnet18', choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224", "resnet18"], 
                        help="Tactile encoder model")
    parser.add_argument('--common_latent_dim', type=int, default=None, help="Common latent dimension for all modalities, if is None, use open clip latent dimension")
    # in https://arxiv.org/pdf/2106.10270.pdf, they ablate with (drop_rate = 0.0, drop_path_rate = 0.0) or (drop_rate = 0.1, drop_path_rate = 0.1) as the two configurations
    parser.add_argument('--drop_rate', type=float, default=0.0, help="dropout before cls layer in tactile encoder")
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help="drop path for tactile encoder")
    parser.add_argument('--disable_vision_text_loss', action="store_true", default=False, help="Disable vision text loss")
    parser.add_argument('--disable_tactile_text_loss', action="store_true", default=False, help="Disable tactile vision loss")
    parser.add_argument(
        '--find_unused_parameters', action='store_true',
        help="distributed ddp find unused parameters")
    parser.set_defaults(find_unused_parameters=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--log_name', default=None, type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    parser.add_argument('--multi_epochs_dataloader', action='store_true', help='Use MultiEpochsDataLoader to prevent reinitializing dataloader per epoch')
    parser.add_argument("--active_modality_names", nargs="+", type=str, default=["vision", "tactile"], 
                        choices=["vision", "text", "audio", "thermal", "depth", "imu", "tactile"],
                        help="Modalities that are used for training")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--enable_flash_attention2', action='store_true', default=False, help="Use flash attntion 2")
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--json_file', default='/Dataset/hct/hct_caption.json',
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    torch.backends.cudnn.determinstic = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)    
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f'device: {device}')
    print(f'seed: {seed}')

    cudnn.benchmark = True
    
    handle_flash_attn(args)

    # handle the active modalities
    modality_types = []
    modalities = ["vision", "text", "tactile"]
    for modality_name in args.active_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            modality_types.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")

            
    dataset = TacTextDataset(args.json_file, TAC_AUGMENTS, device=device)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
        print("Sampler= %s" % str(sampler))
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataloader_cls = MultiEpochsDataLoader if args.multi_epochs_dataloader else torch.utils.data.DataLoader
    data_loader = dataloader_cls(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    
    # define the model
    model = tvl.TVL(tactile_model=args.tactile_model, active_modalities=modality_types, common_latent_dim=args.common_latent_dim)
    loss = TVLLoss(
        active_modalities=modality_types, use_tac_text_loss=args.use_tac_text_loss, 
        disable_vision_text_loss=args.disable_vision_text_loss, disable_tactile_text_loss=args.disable_tactile_text_loss,
    )
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], 
            find_unused_parameters=args.find_unused_parameters
        )
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # mask ratio sampler 
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy_1 = 0.0
    max_accuracy_5 = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, loss, data_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir:
            if train_stats["average_acc1"] >= max_accuracy_1:
                max_accuracy_1 = train_stats["average_acc1"]
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, metric="average_acc1")
            if train_stats["average_acc5"] >= max_accuracy_5:
                max_accuracy_5 = train_stats["average_acc5"]
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, metric="average_acc5")
            # save latest model
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_latest_model_only=True)
    
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.log_name is not None: 
        args.output_dir = os.path.join(args.output_dir, args.log_name)
    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_name is not None and misc.is_main_process():
        wandb.init(entity="project_vit", project="tvl", config=args, name=args.log_name, sync_tensorboard=True)
    main(args)
