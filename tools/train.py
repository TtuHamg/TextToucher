import argparse
import datetime
import os
import sys
import time
import types
import warnings
import re
import json
from copy import deepcopy
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from mmcv.runner import LogBuffer
from torch.utils.data import RandomSampler

from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import (
    AspectRatioBatchSampler,
    BalancedAspectRatioBatchSampler,
)
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import (
    set_random_seed,
    read_config,
    init_random_seed,
    DebugUnderflowOverflow,
)
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from collections import Counter
warnings.filterwarnings("ignore")  # ignore warning

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

# TODO: fix bug: NotImplementedError:Using RTX 3090 or 4000 series dosen’t support faster communication via P2P or IB。
# os.environ["NCCL_SOCKET_IFNAME"] = "enp0s31f6"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "TextToucherBlock"

def train():
    if config.get("debug_nan", False):
        DebugUnderflowOverflow(model)
        logger.info(
            "NaN debugger registered. Start to detect overflow during training."
        )
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    # Now you train the model
    # with torch.autograd.detect_anomaly():
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start = time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            if load_vae_feat:
                z = batch[+0]
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=config.mixed_precision == "fp16"
                    ):
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()
            clean_images = z * config.scale_factor
            y = batch[1]
            y_mask = batch[2]
            data_info = batch[3]
            if config.bg_embed > 0:
                bg_index = batch[4]
            else:
                bg_index = None

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            grad_norm = None
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model,
                    clean_images,
                    timesteps,
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, bg_index=bg_index),
                )
                loss = loss_term["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients and config.gradient_clip is not None:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients:
                    pass

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time * (total_steps - start_step - global_step - 1)
                        )
                    )
                )
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(avg_time * (len(train_dataloader) - step - 1))
                    )
                )
                # avg_loss = sum(loss_buffer) / len(loss_buffer)
                log_buffer.average()
                if args.debug:
                    info = (
                        f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, "
                        f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({model.h}, {model.w}), "
                    )
                else:
                    info = (
                        f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, "
                        f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({model.module.h}, {model.module.w}), "
                    )
                info += ", ".join(
                    [f"{k}:{v:.4f}" for k, v in log_buffer.output.items()]
                )
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                
                
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            global_step += 1
            data_time_start = time.time()

            synchronize()
            if accelerator.is_main_process:
                if (
                    (epoch - 1) * len(train_dataloader) + step + 1
                ) % config.save_model_steps == 0:
                    os.umask(0o000)
                    save_checkpoint(
                        os.path.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=(epoch - 1) * len(train_dataloader) + step + 1,
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                    )
            synchronize()

        synchronize()
        if accelerator.is_main_process:
            if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
                os.umask(0o000)
                save_checkpoint(
                    os.path.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )
        synchronize()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # parser.add_argument("config", type=str, help="config")
    parser.add_argument(
        "--config",
        default="./configs/TextToucher_config/TextToucher_xl2_img512_TVL.py",
        type=str,
        help="config",
    )
    parser.add_argument(
        "--cloud", action="store_true", default=False, help="cloud or local machine"
    )
    parser.add_argument(
        "--work-dir",
        # default="..output/trained_model",#TODO:annotated
        help="the dir to save logs and models",
    )
    parser.add_argument("--resume-from", help="the dir to resume the training")
    parser.add_argument(
        "--load-from", default=None, help="the dir to load a ckpt for training"
    )
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="touch_clear_description_to_tactile",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = "/data/data"
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True,
        )
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 8
        config.valid_num = 100

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = (
        InitProcessGroupKwargs()
    )  # like init_process_group in pytorch.distributed
    init_handler.timeout = datetime.timedelta(
        seconds=5400
    )  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = "FSDP"
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullStateDictConfig,
        )

        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=False, rank0_only=False
            ),
        )
    else:
        init_train = "DDP"
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches = (False,)

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,  # deal with unequal size for processes
        kwargs_handlers=[init_handler],
    )

    logger = get_root_logger(os.path.join(config.work_dir, "train_log.log"))

    config.seed = init_random_seed(config.get("seed", None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, "config.py"))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, "pred_sigma", True)
    learn_sigma = getattr(config, "learn_sigma", True) and pred_sigma
    model_kwargs = {
        "window_block_indexes": config.window_block_indexes,
        "window_size": config.window_size,
        "use_rel_pos": config.use_rel_pos,
        "lewei_scale": config.lewei_scale,
        "config": config,
        "model_max_length": config.model_max_length,
    }

    # build models
    train_diffusion = IDDPM(
        str(config.train_sampling_steps),
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.snr_loss,
    )
    model = build_model(
        config.model,
        config.grad_checkpointing,
        config.get("fp32_attention", False),
        input_size=latent_size,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        **model_kwargs,
    ).train()
    logger.info(
        f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    if config.load_from is not None:
        if args.load_from is not None:
            config.load_from = args.load_from
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get("load_ema", False)
        )
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")

    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).cuda()

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(
        config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type, config = config,
    )
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(
            sampler=RandomSampler(dataset),
            dataset=dataset,
            batch_size=config.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=True,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.valid_num,
        )
        # used for balanced sampling
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(
            dataset, batch_sampler=batch_sampler, num_workers=config.num_workers
        )
    else:
        if config.resample:
            with open(os.path.join(config.data_root, 'hct', config.data.json_file)) as f:
                data_files = json.load(f)
            bg_index_list = []
            for item in data_files:
                bg_index_list.append(int(re.search(r"data\d", item['tactile_feat']).group(0)[-1]) - 1)
            bg_index_static = Counter(bg_index_list)
            train_weights = [ 1/bg_index_static[idx]  for idx in bg_index_list]   
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, num_samples=len(dataset), replacement=False)
            train_dataloader = torch.utils.data.DataLoader(
                dataset,
                num_workers=config.num_workers,
                batch_size=config.train_batch_size,
                sampler=train_sampler
            )
        else:
            train_dataloader = build_dataloader(
                dataset,
                num_workers=config.num_workers,
                batch_size=config.train_batch_size,
                shuffle=True,
            )

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get("auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train_batch_size
            * get_world_size()
            * config.gradient_accumulation_steps,
            config.optimizer,
            **config.auto_lr,
        )
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(
        config, optimizer, train_dataloader, lr_scale_ratio
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    if config.resume_from is not None and config.resume_from["checkpoint"] is not None:
        start_epoch, missing, unexpected = load_checkpoint(
            **config.resume_from,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    train()
