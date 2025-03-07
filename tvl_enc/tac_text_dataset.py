import json
import re
import os
import numpy as np 
from torch.utils.data import Dataset
from tvl_enc.tvl import tokenizer
from collections import OrderedDict
from torchvision import transforms
from PIL import Image 
from tvl_enc.tvl import ModalityType
from tqdm import tqdm

class TacTextDataset(Dataset):
    def __init__(self, json_file, transform_tac=None, device='cpu'):
        self.text_files = []
        self.tac_files = []
        self.transform_tac = transform_tac
        self.device = device
        with open(json_file) as f: 
            data = json.load(f)
        
        dir = os.path.dirname(json_file)
        for item in tqdm(data):
            # tvl
            vision_img = item['tactile_img'].replace('tactile', 'vision')
            ## ssvtp
            # vision_img = item['tactile_img'].replace('_tac', '_rgb')
            try:
                Image.open(os.path.join(dir, vision_img)).convert('RGB')
            except:
                print('Error: ', vision_img)
                continue
            self.text_files.append(os.path.join(dir, item['prompt']))
            self.tac_files.append(os.path.join(dir, vision_img))
    def __len__(self):
        assert len(self.tac_files) == len(self.text_files)
        return len(self.tac_files)
    def __getitem__(self, idx):
        path = self.tac_files[idx]
        img = Image.open(path).convert('RGB')
        if self.transform_tac is not None:
            img = self.transform_tac(img)
        item = OrderedDict()
        item[ModalityType.TACTILE] = img
        text = self.text_files[idx]
        item[ModalityType.TEXT] = tokenizer(text).squeeze()
        return item
    
    
    