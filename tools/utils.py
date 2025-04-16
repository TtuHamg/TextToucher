import sys
import os
import torch
import re

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import random
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.utils import save_image
from pathlib import Path
from metrics import metric
from diffusion.model.nets import TextToucher_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion import IDDPM, DPMS
from diffusers.models import AutoencoderKL
from PIL import Image
from diffusion.utils.misc import read_config

def compare_vis(gt_dir, fake_dir, interval, plt_col=4):
    start, end = interval
    num = end - start
    plt_row = num // plt_col + 1
    
    fig, axes = plt.subplots(plt_row, plt_col, figsize=(4*plt_col, 2*plt_row))
    axes = axes.flatten()
    for i, index in enumerate(np.arange(start, end)):
        gt_img = os.path.join(gt_dir, f"{index:04d}_tac.jpg")
        fake_img = os.path.join(fake_dir, f"{index:04d}.jpg")
        gt_img = np.array(Image.open(gt_img).resize((512,512),0))
        fake_img = np.array(Image.open(fake_img))
        cat_img = Image.fromarray(np.concatenate((gt_img, fake_img), axis = 1))
        axes[i].set_xticks([])  
        axes[i].set_yticks([])  
        axes[i].set_title(index)
        axes[i].axis('off')  
        axes[i].imshow(cat_img)
    return fig, axes


def organize_eval_partion_datasets(save_dir, json_file, part, repeat_num):
    """
        organize eval data_part, use clear prompt!
    """
    with open(json_file) as f:
        data = json.load(f)
        
    log_dict = {}
    eval_index = []

    for item in data: 
        folder_name = item['tactile_img'].split('tactile')[0]
        if folder_name not in log_dict.keys():
            log_dict[folder_name] = 1
            eval_index.append(item)
        elif log_dict[folder_name] < repeat_num:
            log_dict[folder_name] += 1
            eval_index.append(item)
    if len(eval_index)<5000:
        raise ValueError(f"repeat_num: {repeat_num} is not enough, length is {len(eval_index)}.")
    eval_index = eval_index[:5000]
    with open(os.path.join(os.path.dirname(json_file), f"tvl_clear_cut_{part}.json"), 'w') as f: 
        f.write(json.dumps(eval_index))

    tactille_eval_folder = os.path.join(save_dir, f'tvl_eval_{part}/tactile')
    prompt_eval_folder = os.path.join(save_dir, f'tvl_eval_{part}/prompt_clear')
    os.makedirs(tactille_eval_folder, exist_ok=True)
    os.makedirs(prompt_eval_folder, exist_ok=True)

    root_dir = os.path.dirname(json_file)
    for index, item in enumerate(eval_index):
        tactile_path = os.path.join(root_dir, item['tactile_img'])
        prompt_path = os.path.join(root_dir, item['txt_feat'])
        shutil.copy(tactile_path, os.path.join(tactille_eval_folder,f"{index}_tac.jpg"))
        shutil.copy(prompt_path, os.path.join(prompt_eval_folder, f"{index}_prompt.npz"))
        
    feature_list = []
    mask_list = []
    for file in resort_file(prompt_eval_folder):
        embed = np.load(os.path.join(prompt_eval_folder, file))
        feature_list.append(torch.tensor(embed['caption_feature'][None]))
        mask_list.append(torch.tensor(embed['attention_mask']))
    
    param = {'caption_embs_list': feature_list, 'emb_masks_list': mask_list}
    torch.save(param,os.path.join(save_dir, f'tvl_eval_{part}/prompt_clear_embed.ckpt'))
        

def organize_caption_from_json(json_file, caption_file):
    with open(json_file) as f: 
        data = json.load(f)
        
    with open(caption_file) as f: 
        caption = json.load(f) 
    
    
    result_with_caption = []
    for item in data: 
        item_caption = {}
        obj_folder = item['tactile_img'].split('tactile')[0]
        item_caption['tactile_img'] = item['tactile_img']
        item_caption['tactile_feat'] = item['tactile_feat']
        item_caption['txt_feat'] = re.sub( "prompt_clear", "prompt_caption",item['txt_feat'])
        item_caption['prompt'] = f"the touch of {caption[obj_folder]} is " + item['prompt']
        result_with_caption.append(item_caption)
    with open(os.path.join(os.path.dirname(json_file), 'tvl_caption.json'), 'w') as f: 
        for item in result_with_caption:
            json.dump(item, f,indent=4)
            f.write(',\n')

        
        
        



def delete_file(path):
    for file in os.listdir(path):
        if int(re.search(r"\d+", file).group()) % 5 != 0:
            os.remove(os.path.join(path, file))


def organize_eval_datasets(save_dir, json_file):
    """organize eval dataset from json_file, use uniform sample (each object sample 7 images)
    note: exist object only has one image

    Args:
        save_dir (_type_): _description_
        json_file (_type_): _description_
    """
    tactile_dir = os.path.join(save_dir, "tactile")
    prompt_dir = os.path.join(save_dir, "prompt")
    os.makedirs(tactile_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    with open(json_file) as f:
        data = json.load(f)

    root_dir = os.path.dirname(json_file)
    for index, item in tqdm(enumerate(data)):
        shutil.copy(
            os.path.join(root_dir, item["tactile_img"]),
            os.path.join(tactile_dir, f"{index}_tac.jpg"),
        )
        shutil.copy(
            os.path.join(root_dir, item["txt_feat"]),
            os.path.join(prompt_dir, f"{index}_prompt.npz"),
        )



def split_eval_datasets(json_file, sample_number):
    with open(json_file) as f:
        data = json.load(f)
    print(len(data))
    random.shuffle(data)
    eval_list = []
    statistics_dict = {}

    for item in data:
        folder = item["tactile_img"].split("tactile")[0]
        if folder not in statistics_dict.keys():
            eval_list.append(item)
            statistics_dict[folder] = 1
        elif statistics_dict[folder] < 7:
            eval_list.append(item)
            statistics_dict[folder] += 1
    try:
        print(len(eval_list))
        eval_list = eval_list[0:sample_number]
    except Exception as e:
        print(e)
    
    save_dir = os.path.dirname(json_file)
    file_name = f"{Path(json_file).stem}_eval.json"
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(eval_list, f, indent=4)


def resort_file(path):
    path_list = os.listdir(path)
    path_list.sort(key=lambda x: int(re.search(r"\d+", x).group(0)))
    return path_list


def synthesis_dataset(
    save_dir,
    model,
    vae_path,
    eval_prompts,
    device,
    vae_device,
    sampling_algo="dpm-solver",
    batch_size=100,
    image_size=512,
    cfg_scale=4.5,
    seed=43,
    bg_idx='',
    steps=50,
):
    torch.random.manual_seed(seed)
    os.umask(0o000)
    os.makedirs(save_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(vae_path).to(vae_device)

    if os.path.isfile(eval_prompts):
        state_dict = torch.load(eval_prompts, map_location="cpu")
        caption_embs_list = state_dict["caption_embs_list"]
        emb_masks_list = state_dict["emb_masks_list"]
    else:
        caption_embs_list = []
        emb_masks_list = []
        for npz_path in tqdm(resort_file(eval_prompts)):
            txt_info = np.load(os.path.join(eval_prompts, npz_path))
            caption_embs = torch.from_numpy(txt_info["caption_feature"])
            caption_embs = caption_embs.float()[:, None].to(device)
            emb_masks = torch.from_numpy(txt_info["attention_mask"]).to(device)

            caption_embs_list.append(caption_embs)
            emb_masks_list.append(emb_masks)

    if os.path.isfile(bg_idx):
        bg_idx = torch.load(bg_idx, map_location='cpu')

    samples_num = len(caption_embs_list)
    iter_num = samples_num // batch_size
    latent_size = image_size // 8
    hw = torch.tensor(
        [[image_size, image_size]], dtype=torch.float, device=device
    ).repeat(batch_size, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(batch_size, 1)
    sample_steps_dict = {"iddpm": steps, "dpm-solver": steps, "sa-solver": steps}
    sample_steps = sample_steps_dict[sampling_algo]

    model.eval()
    model.to(device)
    null_y = (
        model.y_embedder.y_embedding[None].repeat(batch_size, 1, 1)[:, None].to(device)
    )
    with torch.no_grad():
        for iter in tqdm(np.linspace(0, samples_num, iter_num + 1)):
            if samples_num == iter:
                break
            caption_embs = torch.cat(
                caption_embs_list[int(iter) : int(iter) + batch_size], dim=0
            ).to(device)
            emb_masks = torch.cat(
                emb_masks_list[int(iter) : int(iter) + batch_size], dim=0
            ).to(device)
            bg_idx_batch = bg_idx[int(iter) : int(iter) + batch_size]
            if sampling_algo == "iddpm":
                # Create sampling noise:
                n = batch_size
                z = torch.randn(n, 4, latent_size, latent_size, device=device).repeat(
                    2, 1, 1, 1
                )
                model_kwargs = dict(
                    y=torch.cat([caption_embs, null_y]),
                    cfg_scale=cfg_scale,
                    data_info={"img_hw": hw, "aspect_ratio": ar},
                    mask=emb_masks,
                    bg_index = bg_idx_batch
                )
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=device,
                )
                samples, _ = samples.to(vae_device).chunk(2, dim=0)
            elif sampling_algo == "dpm-solver":
                # Create sampling noise:
                n = batch_size
                z = torch.randn(n, 4, latent_size, latent_size, device=device)
                model_kwargs = dict(
                    data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks, bg_index = bg_idx_batch
                )
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                ).to(vae_device)
            del z

            samples_ = vae.decode(samples / 0.18215).sample.cpu()
            torch.cuda.empty_cache()
            del samples
            # Save images:
            for i, sample in enumerate(samples_):
                save_path = os.path.join(save_dir, f"{(int(iter)+i):04d}.jpg")
                save_image(
                    sample, save_path, nrow=1, normalize=True, value_range=(-1, 1)
                )
                
def extract_bg_idx(json_file,save_path):
    with open(json_file) as f:
        data = json.load(f)
    bg_idx_list = []
    for item in data:
        bg_idx = int(re.search(r'\d', item['tactile_img']).group(0))-1
        bg_idx_list.append(bg_idx)
    torch.save(torch.IntTensor(bg_idx_list), save_path)