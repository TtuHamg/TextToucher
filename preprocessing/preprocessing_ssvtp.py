import sys 
import os
from pathlib import Path
import json
import re
import shutil
from diffusion.model.t5 import T5Embedder
import torch
import numpy as np
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms as T


def extract_text_feature(txt_dir: str, t5_path: str):
    
    t5 = T5Embedder(device="cuda:7", local_cache=True, cache_dir=t5_path, model_max_length=120)
    os.makedirs(os.path.join(os.path.dirname(txt_dir), 'prompt_feat'),exist_ok=True)
    for txt_file in tqdm(os.listdir(txt_dir)):
        with open(os.path.join(txt_dir, txt_file)) as f:
            description = f.readline().strip()
            description = "the feeling of " + description
        
        with torch.no_grad():
            if isinstance(description, str):
                description = [description]
            
            save_path = os.path.join(os.path.dirname(txt_dir), 'prompt_feat', f'{Path(txt_file).stem}.npz')
            if os.path.exists(save_path):
                print(f"error: exist {save_path}.npz")
                return
            try:
                caption_emb, emb_mask = t5.get_text_embeddings(description)
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
        T.Resize((image_resize,image_resize)),  # Image.BICUBIC
        # T.CenterCrop(image_resize),
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])
    
    os.umask(0o000)
    save_dir = os.path.join(os.path.dirname(root_dir), 'tac_feat')
    os.makedirs(save_dir, exist_ok=True)
    for tac_file in tqdm(os.listdir(root_dir)):
        try: 
            img_tac = Image.open(os.path.join(root_dir, tac_file))
            img_tac=transform(img_tac).to(device)[None]
            with torch.no_grad():
                posterior = vae.encode(img_tac).latent_dist
                z = torch.cat([posterior.mean, posterior.std], dim=1).detach().cpu().numpy().squeeze()
            np.save(f"{os.path.join(save_dir, Path(tac_file).stem)}.npy", z)  
        except Exception as e: 
            print(e)
            print(tac_file)
        
        
        
        
def reorganize_dataset(root_dir:str,save_dir:str):
    dataset=[]

    tac_path = os.path.join(root_dir, 'images_tac')
    for tac_file in os.listdir(tac_path):

        index = re.findall(r'\d+', tac_file)[0]
        sample_json = {}
        sample_json['tactile_img'] = os.path.join('images_tac', f"image_{index}_tac.jpg")
        sample_json['tactile_feat'] = os.path.join("tac_feat", f"image_{index}_tac.npy")
        sample_json['txt_feat'] = os.path.join("prompt_feat", f"labels_{index}.npz")
        with open(os.path.join(root_dir, "text", f"labels_{index}.txt")) as f:
            prompt = f.readline().strip()
            prompt = "the feeling of " + prompt
        sample_json['prompt'] = prompt
        dataset.append(sample_json)

    with open(f"{save_dir}/ssvtp.json",'w') as f:
        json.dump(dataset,f)
