import os
import torch
from torch.utils import data

import torchvision
from torchvision import transforms
import copy

allowed_dbs=["cifar10"]

def get_dataset(db_name:str="CIFAR10", data_path:str=None, isTrain:bool=True, transform_ops:list=[]):
    assert isinstance(db_name, str),f"Expected db_name to be of type `str` but got {type(db_name)}"
    assert isinstance(isTrain, bool),f"Expected isTrain to be of type `bool` but got {type(isTrain)}"
    assert db_name.lower() in allowed_dbs
    assert isinstance(transform_ops, list),f"Expected transform_ops to be of type `list` but got {type(transform_ops)}"

    if data_path is not None:
        assert isinstance(data_path, str), f"Expected data_path to be of type `str` but got {type(data_path)}"
    else:
        data_path = os.getcwd()

    if len(transform_ops)==0:
        transform_ops = [transforms.ToTensor()]
    
    transform_ops = transforms.Compose(transform_ops)

    if db_name.lower() == "cifar10":
    
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=isTrain, download=True, transform=transform_ops)

    return dataset

def get_dataloader(dataset:torch.utils.data.Dataset, batch_size:int=64, n_workers:int=1, isShuffle:bool=True):

    assert isinstance(dataset, torch.utils.data.Dataset),f"Expected dataset to be of type `torch.utils.data.Dataset` but got {type(dataset)}"
    assert isinstance(batch_size, int),f"Expected batch_size to be of type `int` but got {type(batch_size)}"
    assert isinstance(n_workers, int),f"Expected n_workers to be of type `int` but got {type(n_workers)}"
    assert isinstance(isShuffle, bool),f"Expected isShuffle to be of type `bool` but got {type(isShuffle)}"
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=isShuffle, num_workers=n_workers)
    return dataloader

def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer, epoch:int, savePath:str, save_name:str, n_gpus:int=1):
    """Saves a checkpoint."""
    assert model.training is False, "Model should be in eval mode"
    # Ensure that the checkpoint dir exists
    os.makedirs(savePath, exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.module.state_dict() if n_gpus > 1 else model.state_dict()
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'model_state': sd,
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    checkpoint_fpath = os.path.join(savePath, save_name)
    torch.save(checkpoint, checkpoint_fpath)
    return checkpoint_fpath

def load_checkpoint(checkpoint_file:str, model:torch.nn.Module, optimizer:torch.optim.Optimizer=None, n_gpus:int=1):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), \
        'Checkpoint \'{}\' not found'.format(checkpoint_file)

    # Load the checkpoint on CPU to avoid GPU mem spike
    temp_checkpoint = torch.load(checkpoint_file, map_location='cpu')
    checkpoint = copy.deepcopy(temp_checkpoint)
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model

    ms = model.module if n_gpus > 1 else model

    isModuleStrPresent=False
    if 'model_state' in checkpoint:
        checkpoint = checkpoint['model_state']        
        
    for k in checkpoint.keys():
        if k.find("module.") == -1:
            continue
        isModuleStrPresent=True
        break
    
    #remove module
    if isModuleStrPresent:
        print("Loaded checkpoint contains module present in keys.")
        print("So now removing 'module' strings")
        #remove module strings
        from collections import OrderedDict
        new_ckpt_dict = OrderedDict()
        for k,v in checkpoint.items():
            tmp_key = k.replace("module.","")
            new_ckpt_dict[tmp_key] = v
        
        checkpoint = copy.deepcopy(new_ckpt_dict)
        print("Done!!")
        
    ms.load_state_dict(checkpoint)
    ms.cuda(torch.cuda.current_device())

    #ms.load_state_dict(checkpoint['model_state'])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(temp_checkpoint['optimizer_state'])



