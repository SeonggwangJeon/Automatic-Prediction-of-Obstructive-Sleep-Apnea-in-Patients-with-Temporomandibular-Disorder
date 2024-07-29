import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import torch

def GetLoader(transform, x, y, batch = 32, test=True):
    dataset = TMJImageDataset(x,y,transform=transform)
    return DataLoader(dataset, batch_size = batch, shuffle=(not test), num_workers=0)
        
class TMJImageDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = self.x[index, :] 
        y = self.y[index]              
        
        if self.transform is None or len(self.transform) == 0:
            x = transforms.ToTensor()(x)
            return x, y
        else:
            transform = transforms.Compose(self.transform)
            pli = Image.fromarray(np.uint8(x)).convert('RGB')
            x = transform(pli)
            x = np.array(x)
            x = transforms.ToTensor()(x)
            return x,y
        
        
def GetLoader_for_clinic(x, y, batch = 32, test=True):
    dataset = ClinicDataset(x,y)
    return DataLoader(dataset, batch_size = batch, shuffle=(not test), num_workers=0)
        
class ClinicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = self.y[index]
        return x, y