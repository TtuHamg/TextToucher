"""
calcuate metrics code are based on 
    https://github.com/Max-Fu/tvl
"""
import sys 
import argparse
import torch 
import json
import re
import os 
import numpy as np  

from tqdm import tqdm 
from itertools import chain, combinations
from torch.utils.data import Dataset
from tvl_enc.tvl import tokenizer
from collections import OrderedDict
from tools import utils
from torchmetrics.regression import CosineSimilarity
from torchvision import transforms
from tvl_enc import tvl 
from tvl_enc.tvl import ModalityType
from PIL import Image 


TAC_MEAN = np.array([0.29174602047139075, 0.2971325588927249, 0.2910404549605639])
TAC_STD = np.array([0.18764469044810236, 0.19467651810273057, 0.21871583397361818])
TAC_PREPROCESS = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=TAC_MEAN,
        std=TAC_STD,
    ),
])

def load_text(
    raw_text: str,
    # prompt: str = "This image gives tactile feelings of ",
    prompt: str = "",
    device: str = None,
    shuffle: bool = False, 
    random_subset: bool = False,
    synonyms_dict: dict = None
):
    keywords = raw_text.replace(".", "").replace("\n", "").lower().split(',')
    keywords = [k.strip() for k in keywords]
    if random_subset:
        ps = chain.from_iterable(combinations(keywords, r) for r in range(len(keywords)+1))
        ps = list(ps)[1:] # drop first element (the empty set)
        text = prompt
        words = ps[np.random.choice(len(ps))]
        # words = [np.random.choice(keywords)]
        if shuffle:
            words = list(words)
            random.shuffle(words)
        if synonyms_dict is not None:
            selected = []
            for w in words:
                if w in synonyms_dict:
                    selected.append(np.random.choice(synonyms_dict[w]))
                else:
                    selected.append(w)
            words = selected
    else:
        text = prompt
        words = keywords
    if len(words) == 1:
        text += words[0]
    elif len(words) == 1:
        text += f"{words[0]} and {words[1]}"
    else:
        text += ", ".join(words[i] for i in range(len(words) - 1)) + f", and {words[-1]}"
    text += "."
    tokens = tokenizer(text)
    if device is not None:
        tokens = tokens.to(device)
    return tokens

class TacTextDataset(Dataset):
    def __init__(self, tac_files, text_json, transform_tac=None, device='cpu'):
        self.tac_files = tac_files
        with open(text_json) as f: 
            text_files = json.load(f)
        
        self.text_files = text_files
        self.transform_tac = transform_tac
    def __len__(self):
        assert len(self.tac_files) == len(self.text_files)
        return len(self.tac_files)
    def __getitem__(self, idx):
        path = self.tac_files[idx]
        img = Image.open(path).convert('RGB')
        if self.transform_tac is not None:
            img = self.transform_tac(img)
        item = OrderedDict()
        item[ModalityType.TACTILE] = img.to(device)
        text = self.text_files[idx]['prompt']
        # text = "This image gives tactile feelings of " + text[re.search(r' is ', text).start()+4:]
        # item[ModalityType.TEXT] = tokenizer(text).squeeze().to(device)
        item[ModalityType.TEXT] = load_text(text).squeeze().to(device)
        return item
    
        
def parse_args():
    parser = argparse.ArgumentParser(description="Calcuate Contrastive Text-Touch Pre-training")
    parser.add_argument("--checkpoint", default=None, required=True, type=str)
    parser.add_argument("--device", default='cpu', type=str,)
    parser.add_argument("--tactile_model", default='vit_tiny_patch16_224', type=str)
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument("--tac_dir", default="", required=True, type=str)
    parser.add_argument("--text_json", default="", required=True, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    device = args.device
    model = tvl.TVL(active_modalities = [ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT] , tactile_model=args.tactile_model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    
    # construct paired touch-text dataset
    # use 0001.png, 0002.png, ... to order touch images
    tac_files = utils.resort_file(args.tac_dir)
    tac_files = [os.path.join(args.tac_dir, file)for file in tac_files]
    TacText_data = TacTextDataset(tac_files, args.text_json, TAC_PREPROCESS, device=device)
    
    dataloader = torch.utils.data.DataLoader(
        TacText_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    cosine_similarity = CosineSimilarity(reduction = 'mean')

    with torch.no_grad():
        for sample in tqdm(dataloader):
            out_dict = model(sample)
            cosine_similarity(out_dict['tactile'], out_dict['text'])
        CTTP_score = cosine_similarity.compute()
    
    if args.save_path is not None:
        with open(args.save_path, 'a') as f:
            f.writelines(f'cttp: {CTTP_score}\r\n')