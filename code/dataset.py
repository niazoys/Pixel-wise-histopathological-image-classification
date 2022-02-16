import os
import imageio
import numpy as np
import torch
from utils import RandomContrast
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

class hist_dataset(Dataset):
    
    def __init__(self,path:str,shape=None,training=True):
        self.path=path
        self.training=training
        self.shape=shape
        self.randomCont=RandomContrast(random_state=np.random.RandomState(111))
        self.trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20), 
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

        self.trans_valid = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])     

    def __len__(self):
        self.files = os.listdir(self.path)    
        return len(self.files)

    def __getitem__(self, idx:int):
        img_mask =  np.array(imageio.imread( os.path.join(self.path,self.files[idx])))
        if self.training:
            img_=img_mask[:,:self.shape,:]
            # img=np.array(self.randomCont(img_)).astype(np.float32)
            img=self.trans_train(img_)
        else:
            img_=img_mask[:,:self.shape,:]
            img=self.trans_valid(img_)

        mask=(img_mask[:,self.shape:,2]) 
        
        return img,torch.tensor(mask),img_

class hist_dataset_test(Dataset):
    def __init__(self,path:str):
        self.path=path
        self.transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])     
    def __len__(self):
        self.files =sorted(os.listdir(self.path),key=lambda x: float(x[:-3]))
        return len(self.files)

    def __getitem__(self, idx:int):
        img_ =  np.array(imageio.imread( os.path.join(self.path,self.files[idx])))
        img = self.transform(img_)
        return img,img_