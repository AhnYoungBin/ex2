from torch.utils.data import Dataset
import albumentations as A
import os
import cv2
import torch
import numpy as np
from torchvision import transforms

class Brain_DataSet(Dataset):
    def __init__(self,path, transform=None):
        self.path = path
        self.transform = transform
        self.patients = [file for file in os.listdir(path) if file not in ['data.csv','README.md']]
        self.images, self.masks = [],[]
        for patient in self.patients:
            for file in os.listdir(os.path.join(self.path,patient)):
                if 'mask' not in file.split('.')[0].split('_'):
                    if '(1)' not in file.split('.')[0].split(' '):
                        self.images.append(os.path.join(self.path,patient,file))
        self.images = sorted(self.images)

        for image in self.images:
            self.masks.append(image[:-4]+'_mask.tif')

        print(self.images[0])
        print(self.masks[0])
            

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        image = cv2.imread(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
        ret,mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if self.transform:
            augmented = self.transform(image=image, mask= mask)
            image = augmented['image']
            image = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
            mask = augmented['mask']
            mask = np.reshape(mask,(224,224,1))


            totensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5),(0.5))
            ])
            image = totensor(image)
            mask = totensor(mask)


        return image, mask
