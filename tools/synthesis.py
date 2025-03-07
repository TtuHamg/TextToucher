import argparse
import torch
import os
import json
import sys

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion.model.nets import TextToucher_XL_2
from tools.utils import synthesis_dataset
from metrics.metric import calculate_metrics
from diffusion.utils.misc import read_config

def parse_args():
    parser = argparse.ArgumentParser(description="Synthesis Touch Datasets")
    parser.add_argument("--model_path", type=str, default=True)
    parser.add_argument("--vae_path", type=str, default="")
    parser.add_argument("--sampling_algo", type=str, default="dpm-solver")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--vae_device", type=str, required=True)
    parser.add_argument("--eval_prompts", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=25)

    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--bg_idx", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--steps", type=int, default=25)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config)
    print(args)

    latent_size = 64
    model = TextToucher_XL_2(input_size=latent_size, lewei_scale=1, config=config, model_max_length=config.model_max_length).to(args.device)
    state_dict = torch.load(args.model_path, map_location="cpu")
    del state_dict["state_dict"]["pos_embed"]
    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    print("Missing keys: ", missing)
    print("Unexpected keys", unexpected)

    eval_prompts = args.eval_prompts

    synthesis_dataset(
        args.save_dir,
        model,
        args.vae_path,
        args.eval_prompts,
        args.device,
        args.vae_device,
        args.sampling_algo,
        args.batch_size,
        bg_idx=args.bg_idx,
        seed=args.seed,
        cfg_scale=args.cfg,
        steps = args.steps
    )
    del model
    torch.cuda.empty_cache()