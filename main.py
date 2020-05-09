
#main.py
import torch
import albumentations as A
from torch.utils.data import random_split, DataLoader
import numpy as np
import data
import modellist
import argparse
import utils
import torch.optim as optim


device = 'cuda' if torch.cuda.is_available() else 'cpu'
path ='/content/gdrive/My Drive/lgg-mri-segmentation/kaggle_3m'

modellist = modellist.Modellist()

parser = argparse.ArgumentParser(description='Learn by Modeling Segmentation DataSet')
parser.add_argument('modelnum',type=int, help='Select your model number')
parser.add_argument("-show", help="show to model Archtecture",action="store_true")
parser.add_argument('lr',type=float, help='Select opimizer learning rate')
parser.add_argument('epochs',type=int, help='Select train epochs')
args = parser.parse_args()

model = modellist(args.modelnum)
model = model.to(device)
criterion = utils.DiceBCELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
print('criterion : DiceBCELoss')
print('learning_rate : ',args.lr)
print('optimizer : SGD(momentum =0.9 , decay = 5e-4')
print('-'*100)

#show to model archtecture
if args.show:
    print('Model Archtecture')
    print('-'*100)
    print(model)
    print('-'*100)


albumentations_transform = A.Compose([
    A.Resize(256,256),
    A.RandomCrop(224,224),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5)
])

dataset = data.Brain_DataSet(path,albumentations_transform)
sample_image, sample_mask = dataset[0]
print('데이터셋 크기: ',len(dataset))
print('이미지 크기 : ',sample_image.size())
print('마스크 크기 : ',sample_mask.size())

train_dataset, val_dataset = random_split(dataset, [3600, 329])

print('학습 데이터셋 크기: ',len(train_dataset))
print('검증 데이터셋 크기: ',len(val_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16)

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

for epoch in range(1,args.epochs+1):
    print('Epoch {}/{}'.format(epoch, args.epochs))
    print('-'*20)
    train()
    val()
