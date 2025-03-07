import os
import json
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from diffusion.data.builder import get_data_path, DATASETS
from diffusers.utils.torch_utils import randn_tensor
from diffusion.utils.logger import get_root_logger


@DATASETS.register_module()
class TVLData(Dataset):
    def __init__(
        self,
        root,
        json_file,
        transform,
        load_vae_feat,
        resolution=512,
        input_size=64,
        patch_size=2,
        max_length=120,
        config=None,
        prompt_clear=False,
        **kwargs,
    ):
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.max_length = max_length
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.img_samples = []
        self.prompt_samples = []
        self.ori_imgs_nums = 0

        with open(os.path.join(self.root, json_file)) as f:
            json_dataset = json.load(f)

        for json_file in json_dataset:
            self.img_samples.extend([os.path.join(self.root, json_file["tactile_img"])])
            self.txt_feat_samples.extend(
                [os.path.join(self.root, json_file["txt_feat"])]
            )
            self.vae_feat_samples.extend(
                [os.path.join(self.root, json_file["tactile_feat"])]
            )
            self.prompt_samples.extend([json_file["prompt"]])
            self.ori_imgs_nums += 1

        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader

        logger = (
            get_root_logger()
            if config is None
            else get_root_logger(os.path.join(config.work_dir, "train_log.log"))
        )
        
        self.bg_embed = config.bg_embed

    def getdata(self, index):
        img_path = self.img_samples[index]
        npz_path = self.txt_feat_samples[index]
        npy_path = self.vae_feat_samples[index]
        prompt = self.prompt_samples[index]

        data_info = {
            "img_hw": torch.tensor(
                [torch.tensor(self.resolution), torch.tensor(self.resolution)],
                dtype=torch.float32,
            ),
            "aspect_ratio": torch.tensor(1.0),
        }

        img = self.loader(npy_path) if self.load_vae_feat else self.loader(img_path)
        txt_info = np.load(npz_path)
        txt_feat = torch.from_numpy(txt_info["caption_feature"])
        attention_mask = torch.ones(1, 1, txt_feat.shape[1])
        if "attention_mask" in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info["attention_mask"])[None]
        if txt_feat.shape[1] != self.max_length:
            print("may exist error about text_feat shape!")
            txt_feat = torch.cat(
                [
                    txt_feat,
                    txt_feat[:, -1:].repeat(1, self.max_length - txt_feat.shape[1], 1),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(1, 1, self.max_lenth - attention_mask.shape[-1]),
                ],
                dim=-1,
            )

        if self.transform:
            img = self.transform(img)
        data_info["prompt"] = prompt
        
        
        # add background index in dataaset
        if self.bg_embed > 1:
            bg_index = int(re.search(r"/hct/data\d", npy_path).group(0)[-1]) - 1
            return img, txt_feat, attention_mask, data_info, bg_index
        elif self.bg_embed == 1:
            bg_index = 0
            return img, txt_feat, attention_mask, data_info, bg_index
        
        return img, txt_feat, attention_mask, data_info
    
    def __getitem__(self, idx):
        try:
            return self.getdata(idx)
        except Exception as e:
            print(f"Error details: {str(e)}")
        raise RuntimeError("bad data.")

    def __len__(self):
        return len(self.img_samples)

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(
            mean.shape, generator=None, device=mean.device, dtype=mean.dtype
        )
        return mean + std * sample
