from threading import current_thread
import torch
from torch._C import device
from torch.distributed.distributed_c10d import init_process_group
import torch.nn as nn
import torch.optim as optim


from datetime import timedelta
import os
import sys
from torch.utils import data

import torchvision
from torchvision import transforms

from models import ResNet18

from utils import get_dataloader, get_dataset, load_checkpoint, save_checkpoint

from hparams import hparams
from trainer import train, test
from swa_util import swa_train_contrib, swa_train_pytorch

## set hparams
print('Hyper-parameters for this exp: \n')
print(list(hparams.keys()))

lr = hparams['lr']
wd = hparams['weight_decay']
tr_bs = hparams['train_batch_size']
ts_bs = hparams['test_batch_size']


data_path = os.getcwd()
ckpt_savePath = os.path.join(data_path, "checkpoints")
os.makedirs(ckpt_savePath, exist_ok=True)

train_transform_ops = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
]

test_transform_ops = [
    transforms.ToTensor()
]

# Get datasets
train_dataset   = get_dataset(db_name="CIFAR10", data_path=data_path, isTrain=True, transform_ops=train_transform_ops)
test_dataset    = get_dataset(db_name="CIFAR10", data_path=data_path, isTrain=False, transform_ops=test_transform_ops)

# Get dataloaders
train_dl        = get_dataloader(dataset=train_dataset, batch_size=tr_bs)
test_dl         = get_dataloader(dataset=test_dataset, batch_size=ts_bs, isShuffle=False)

print('Dataset')
print(f'# Train Images: {len(train_dataset.data)}')
print(f'# Test Images:  {len(test_dataset.data)}')

model = ResNet18()

n_gpu_devices = torch.cuda.device_count()

# DP model
if n_gpu_devices:
    print(f'Found {torch.cuda.device_count()} gpus so using DP')
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.cuda()

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)



# # Simple Train Loop

n_epochs = 5
best_test_acc = 0.
for i in range(n_epochs):
    train(i, model, train_dl, optimizer)
    curr_acc = test(i, model, test_dl)

    if curr_acc > best_test_acc:
        
        print(f'At epoch: {i} best accuracy: {curr_acc} previous best: {best_test_acc}')
        best_test_acc = curr_acc
        model_name = f"best_ckpt.pth"
        save_checkpoint(model, optimizer, i, ckpt_savePath, model_name, n_gpu_devices)
    else:
        print(f'Not Best accuracy at epoch: {i} curr accuracy: {curr_acc} best_so_far: {best_test_acc}')
    
## VERIFY TEST ACCURACY
best_ckpt_path = '/nfs/users/ext_prateek.munjal/projects/pm_research/swa_example/checkpoints/best_ckpt.pth'
load_checkpoint(best_ckpt_path, model, optimizer, n_gpu_devices)

test_acc = test(0, model, test_dl)
print('Test_acc: ', test_acc)


## USES SWA FROM TORCH CONTRIB
## Source: https://github.com/pytorch/contrib

swa_train_contrib(train_dl, test_dl, best_ckpt_path, model, optimizer, swa_settings={
    "start_iter":0,
    "frequency": 10,
    "lr": 5e-5,
    "epochs": 3
}, n_gpus=n_gpu_devices)

## USES SWA FROM PYTORCH OFFICIAL
## Source: https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/

swa_train_pytorch(train_dl, test_dl, best_ckpt_path, model, optimizer, swa_settings={
    "start_iter":0,
    "frequency": 10,
    "lr": 5e-5,
    "epochs": 3
}, n_gpus=n_gpu_devices)