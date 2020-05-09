
#main.py
import torch
import albumentations as A
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
import matplotlib.pyplot as plt
#from data import Brain_DataSet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path ='/content/gdrive/My Drive/lgg-mri-segmentation/kaggle_3m'

albumentations_transform = A.Compose([
    A.Resize(256,256),
    A.RandomCrop(224,224),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5)
])

dataset = Brain_DataSet(path,albumentations_transform)
sample_image, sample_mask = dataset[0]
print('데이터셋 크기: ',len(dataset))
print('이미지 크기 : ',sample_image.size())
print('마스크 크기 : ',sample_mask.size())

train_dataset, val_dataset = random_split(dataset, [3600, 329])

print('학습 데이터셋 크기: ',len(train_dataset))
print('검증 데이터셋 크기: ',len(val_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16)

model = UNet(1).to(device)
criterion = DiceBCELoss()
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_loss = []
val_loss = []

def train():
    model.train()
    running_train_loss = []
    for idx, (inputs, mask) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.float)
        mask = mask.to(device, dtype=torch.float)

        pred_mask = model(inputs)
        loss = criterion(pred_mask, mask)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item())

    epoch_train_loss = np.mean(running_train_loss) 
    print('Train loss: {}'.format(epoch_train_loss))                       
    train_loss.append(epoch_train_loss)

def val():
    model.eval()
    running_val_loss = []
    for idx, (inputs, mask) in enumerate(val_loader):
        inputs = inputs.to(device, dtype=torch.float)
        mask = mask.to(device, dtype=torch.float)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            pred_mask = model.forward(inputs)
            loss = criterion(pred_mask,mask)
            running_val_loss.append(loss.item())

    epoch_val_loss = np.mean(running_val_loss)
    print('Validation loss: {}'.format(epoch_val_loss))                                
    val_loss.append(epoch_val_loss)

for epoch in range(1,26):
    print('Epoch {}/{}'.format(epoch, 25))
    print('-'*20)
    train()
    val()
