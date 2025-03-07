import sys 
import os
from pathlib import Path
import json
import re
import shutil
import argparse
import torch
import numpy as np
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from PIL import Image
from diffusion.model.t5 import T5Embedder
from torchvision import transforms as T


def reorganize_description(root_dir: str, save_dir: str):
    prompt = {}
    for data_folder in os.listdir(root_dir):
        if data_folder.find("data") >= 0:
            with open(os.path.join(root_dir, data_folder, "finetune.json")) as f:
                data = json.load(f)

            for item in data:
                img_path = item["image"]
                touch_path = item["tactile"]
                name = os.path.basename(item["image"])
                prompt[os.path.join(data_folder,touch_path)] = item["conversations"][1]["value"]
                
    with open(f"{save_dir}/descriptions.json", "w") as f:
        json.dump(prompt, f)


def extract_text_feature(prompt_json_path: str, t5_path: str):
    root_dir=os.path.dirname(prompt_json_path)
    t5 = T5Embedder(dfice="cuda:1", local_cache=True, cache_dir=t5_path, model_max_length=120)
    with open(prompt_json_path) as f:
        data = json.load(f)
    for image_path, caption in tqdm(data.items()):
        pattern=r'tactile'
        prompt_feature_path=re.sub(pattern,'prompt_clear',image_path)
        prompt_folder=os.path.dirname(prompt_feature_path)
        os.makedirs(os.path.join(root_dir,prompt_folder),exist_ok=True)
        
        with torch.no_grad():
            caption = caption.strip()
            if isinstance(caption, str):
                caption = [caption]

            save_path = os.path.join(root_dir, prompt_folder, Path(prompt_feature_path).stem)
            if os.path.exists(f"{save_path}.npz"):
                print(f"error: exist {save_path}.npz")
                return
            try:
                caption_emb, emb_mask = t5.get_text_embeddings(caption)
                emb_dict = {
                    'caption_feature': caption_emb.float().cpu().data.numpy(),
                    'attention_mask': emb_mask.cpu().data.numpy(),
                }
                np.savez_compressed(save_path, **emb_dict)
            except Exception as e:
                print(e)
                
                
def extract_text_feature(tvl_json_path: str, t5_path: str):
    root_dir=os.path.dirname(tvl_json_path)
    t5 = T5Embedder(device="cuda:0", local_cache=True, cache_dir=t5_path, model_max_length=120)
    with open(tvl_json_path) as f:
        data = json.load(f)
    
    # for item in tqdm(data[:5000]):
    for item in tqdm(data):
        prompt_folder=os.path.dirname(item['txt_feat'])
        os.makedirs(os.path.join(root_dir,prompt_folder),exist_ok=True)
        
        with torch.no_grad():
            caption = item['prompt'].strip()
            if isinstance(caption, str):
                caption = [caption]

            save_path = os.path.join(root_dir, item['txt_feat'])
            try:
                caption_emb, emb_mask = t5.get_text_embeddings(caption)
                emb_dict = {
                    'caption_feature': caption_emb.float().cpu().data.numpy(),
                    'attention_mask': emb_mask.cpu().data.numpy(),
                }
                np.savez_compressed(save_path, **emb_dict)
            except Exception as e:
                print(e)
                
def extract_touch_feature(root_dir:str,vae_path:str,image_resize:int,device:str):
    vae=AutoencoderKL.from_pretrained(vae_path).to(device)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(image_resize),  # Image.BICUBIC
        T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])
    
    for data_folder in tqdm(os.listdir(root_dir)):
        if os.path.isdir(os.path.join(root_dir, data_folder)) and data_folder.find("data") >= 0:
            with open(os.path.join(root_dir, data_folder, "finetune.json")) as f:
                data = json.load(f)
                
                for item in tqdm(data):
                    touch_path = item["tactile"]
                    touch_abs_path=os.path.join(root_dir,data_folder,item['tactile'])
                    pattern=r'tactile'
                    touch_feature_path=re.sub(pattern,'touch_feature',touch_abs_path)
                    touch_feature_folder=os.path.dirname(touch_feature_path)
                    os.makedirs(touch_feature_folder,exist_ok=True)
                    try:
                        img=Image.open(touch_abs_path)
                        img=transform(img).to(device)[None]
                        
                        with torch.no_grad():
                            posterior = vae.encode(img).latent_dist
                            z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy().squeeze()
                        np.save(f"{os.path.join(touch_feature_folder,Path(touch_feature_path).stem)}.npy", z)
                    except Exception as e:
                        print(e)
                        print(item)
def reorganize_dataset(root_dir:str,save_dir:str):
    dataset=[]

    with open(f"{root_dir}/description.json") as f:
        prompt_json=json.load(f)

    for data_folder in os.listdir(root_dir):
        if data_folder.find('data')>=0:
            data_folder_path=os.path.join(root_dir,data_folder)
            for exp in os.listdir(data_folder_path):
                tactile_feat_folder_path=os.path.join(data_folder_path,exp,'touch_feature')
                if os.path.exists(tactile_feat_folder_path):
                    for file in os.listdir(tactile_feat_folder_path):
                        sample_json={}
                        sample_json['tactile_img']=os.path.join(data_folder,exp,'tactile',f"{Path(file).stem}.jpg")
                        sample_json['tactile_feat']=os.path.join(data_folder,exp,'touch_feature',file)
                        sample_json['txt_feat']=os.path.join(data_folder,exp,'prompt_clear',f"{Path(file).stem}.npz")
                        sample_json['prompt']=prompt_json[sample_json['tactile_img']]
                        dataset.append(sample_json)
    with open(f"{save_dir}/hct.json",'w') as f:
        json.dump(dataset,f)