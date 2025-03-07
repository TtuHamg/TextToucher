import torch 
import os 
import argparse
import torchvision.transforms as TF
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img

def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--real_path",type=str)
    parser.add_argument("--fake_path",type=str)
    parser.add_argument("--device",default="cuda:6",type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    batch_size = 25
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    real_path = args.real_path
    real_files = sorted([os.path.join(real_path, file)for file in os.listdir(real_path)])
    
    real_dataset = ImagePathDataset(real_files, transforms=TF.Compose([TF.ToTensor(), TF.Resize([512,512])]))
    real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    
    
    fake_path = args.fake_path
    fake_files = sorted([os.path.join(fake_path, file) for file in os.listdir(fake_path)])
    
    fake_dataset = ImagePathDataset(fake_files, transforms=TF.Compose([TF.ToTensor(), TF.Resize([512,512])]))
    fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    for real_imgs, fake_imgs in tqdm(zip(real_dataloader, fake_dataloader)):
        lpips_score = lpips(real_imgs.to(device), fake_imgs.to(device))
        ssim_score = ssim(fake_imgs.to(device), real_imgs.to(device))
        psnr_score = psnr(fake_imgs.to(device), real_imgs.to(device))
        
    lpips_score = lpips.compute()
    ssim_score = ssim.compute()
    psnr_score = psnr.compute()
    with open(f"{fake_path}_metrics.txt", 'a') as f:
        f.writelines(f'\nlpips: {lpips_score}')
        f.writelines(f'\nssim: {ssim_score}')
        f.writelines(f'\npsnr: {psnr_score}')
        