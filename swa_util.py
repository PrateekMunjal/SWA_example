import torch
from torch import optim
import torch.nn as nn

import torchcontrib
from torchcontrib.optim import swa ## for swa torchcontrib
from tqdm import tqdm
from trainer import test
import copy
import os
import sys
from utils import load_checkpoint
import warnings

from torch.optim.swa_utils import AveragedModel, SWALR

def swa_train_contrib(trainloader:torch.utils.data.DataLoader, testloader:torch.utils.data.DataLoader, model_path:str, model:torch.nn.Module, optimizer:torch.optim.Optimizer, swa_settings:dict, n_gpus:int=1):
    
    already_dp = False
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        print('model present in DDP mode')
        model = model.module
        print('Now model class: ', type(model))
        # sys.exit(0)
    elif isinstance(model, torch.nn.DataParallel):
        print('model present in DP mode')
        warnings.warn('\n We assume the model passed already utilizes maximum gpus.. if not, just pass the model without wrapping in DP/DDP \n', stacklevel=2)
        already_dp = True
    elif isinstance(model, torch.nn.Module):
        pass
    else:
        raise NotImplementedError
    
    # converting to dataparallel
    if not already_dp and n_gpus:
        model = torch.nn.DataParallel(model, device_ids=range(n_gpus))
        model = model.cuda()


    save_path = os.path.abspath( os.path.join(model_path, os.pardir ))

    swa_start_iter = swa_settings['start_iter']
    swa_freq       = swa_settings['frequency']
    swa_lr         = swa_settings['lr']
    swa_epochs     = swa_settings['epochs']

    loss = torch.nn.CrossEntropyLoss()
    print("SWA_TRAIN on selected device_ids: {}".format(n_gpus))

    current_id = torch.cuda.current_device()

    print("===== In swa_util [torchcontrib] =====")
    
    load_checkpoint(model_path, model, optimizer, n_gpus)
    print("Evaluating model validaton accuracy to confirm whether model is correctly loaded")

    model.eval()
    best_acc = test(0, model, testloader)
    print("Newly loaded model has accuracy: {}".format(best_acc))
    model.train()
    
    swa_optimizer = torchcontrib.optim.SWA(optimizer, swa_start=swa_start_iter, \
        swa_freq=swa_freq, swa_lr=swa_lr)

    print(f"SWA Optimizer: {swa_optimizer}")
    
    print("Training SWA for {} epochs.".format(swa_epochs))
    temp_max_itrs = len(trainloader)
    print(f"len(trainloader): {len(trainloader)}")
    temp_cur_itr = 0
    for epoch in tqdm(range(swa_epochs), desc="Training SWA"):
        temp_cur_itr = 0
        for x,y in tqdm(trainloader, desc="Training Epoch"):
            x = x.cuda(current_id)
            y = y.cuda(current_id)

            output = model(x)

            error = loss(output, y)

            swa_optimizer.zero_grad()
            error.backward()
            swa_optimizer.step()
            # if(temp_cur_itr % 50 == 0):
            # print(f"Iteration [{temp_cur_itr}/{temp_max_itrs}] Done !!")
            
            temp_cur_itr += 1

        #print("Epoch {} Done!! Train Loss: {}".format(epoch, error.item()))

    print("Averaging weights -- SWA")
    swa_optimizer.swap_swa_sgd()
    print("Done!!")

    print("Updating BN")

    swa_optimizer.bn_update(loader=trainloader, model=model)
    print("Done!!")

    #Check val accuracy
    print("Evaluating validation accuracy")
    model.eval()

    swa_acc = test(0, model, testloader)

    print("Validation Accuracy with SWA model: {}".format(swa_acc))

    print("Saving SWA model")
    
    sd = model.module.state_dict() if n_gpus > 1 else model.state_dict()
    #when model is on single gpu then model state_dict contains keyword module
    # So we will simply remove <module> from dictionary.
    isModuleStrPresent=False
    for k in sd.keys():
        if k.find("module.") == -1:
            continue
        isModuleStrPresent=True
        break

    if isModuleStrPresent:
        print("SWA checkpoint contains module present in keys.")
        print("So now removing 'module' strings")
        #remove module strings
        from collections import OrderedDict
        new_ckpt_dict = OrderedDict()
        for k,v in sd.items():
            tmp_key = k.replace("module.","")
            new_ckpt_dict[tmp_key] = v
        
        sd = copy.deepcopy(new_ckpt_dict)
        print("Done!!")

    #sd = model.state_dict()
    # Record the state
    checkpoint = {
        'epoch': -1, # to state the checkpoint is corresponding to SWA
        'model_state': sd,
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    
    checkpoint_file = os.path.join(save_path, "[SWA]valSet_acc_{:.5f}.pyth".format(swa_acc))
    print("---Before SAVING SWA MODEL----")
    torch.save(checkpoint, checkpoint_file)
    print("SAVED SWA model")
    print("SWA Model saved at {}".format(checkpoint_file))
    return

def swa_train_pytorch(trainloader:torch.utils.data.DataLoader, testloader:torch.utils.data.DataLoader, model_path:str, model:torch.nn.Module, optimizer:torch.optim.Optimizer, swa_settings:dict, n_gpus:int=1):
    
    already_dp = False
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        print('model present in DDP mode')
        model = model.module
        print('Now model class: ', type(model))
        # sys.exit(0)
    elif isinstance(model, torch.nn.DataParallel):
        print('model present in DP mode')
        warnings.warn('\n We assume the model passed already utilizes maximum gpus.. if not, just pass the model without wrapping in DP/DDP \n', stacklevel=2)
        already_dp = True
    elif isinstance(model, torch.nn.Module):
        pass
    else:
        raise NotImplementedError
    
    # converting to dataparallel
    if not already_dp and n_gpus:
        model = torch.nn.DataParallel(model, device_ids=range(n_gpus))
        model = model.cuda()
    
    save_path = os.path.abspath( os.path.join(model_path, os.pardir ))

    swa_start_iter = swa_settings['start_iter']
    swa_freq       = swa_settings['frequency']
    swa_lr         = swa_settings['lr']
    swa_epochs     = swa_settings['epochs']

    loss = torch.nn.CrossEntropyLoss()
    print("SWA_TRAIN on selected device_ids: {}".format(n_gpus))

    current_id = torch.cuda.current_device()

    print("===== In swa_util [official pytorch] =====")
    
    load_checkpoint(model_path, model, optimizer, n_gpus)
    print("Evaluating model validaton accuracy to confirm whether model is correctly loaded")

    model.eval()
    best_acc = test(0, model, testloader)
    print("Newly loaded model has accuracy: {}".format(best_acc))
    model.train()

    # I recommend to use exp moving average over simple average function; the torchcontrib implements EMA only.
    # I leave the choice to user now.
    # ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.1 * averaged_model_parameter + 0.9 * model_parameter
    swa_model = AveragedModel(model)#, avg_fn=ema_avg)
    swa_scheduler = SWALR(optimizer,anneal_strategy="linear", anneal_epochs=1, swa_lr=swa_lr)
    
    # swa_optimizer = torchcontrib.optim.SWA(optimizer, swa_start=swa_start_iter, \
    #     swa_freq=swa_freq, swa_lr=swa_lr)

    print(f"SWA Optimizer: {optimizer}")
    
    print("Training SWA for {} epochs.".format(swa_epochs))
    temp_max_itrs = len(trainloader)
    print(f"len(trainloader): {len(trainloader)}")
    temp_cur_itr = 0
    for epoch in tqdm(range(swa_epochs), desc="Training SWA"):
        temp_cur_itr = 0
        for x,y in tqdm(trainloader, desc="Training Epoch"):
            x = x.cuda(current_id)
            y = y.cuda(current_id)

            output = model(x)

            error = loss(output, y)

            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            if(temp_cur_itr!=0 and  (temp_cur_itr % swa_freq == 0) and (temp_cur_itr > swa_start_iter)):
                swa_model.update_parameters(model)
                swa_scheduler.step()
            # print(f"Iteration [{temp_cur_itr}/{temp_max_itrs}] Done !!")
            
            temp_cur_itr += 1

        #print("Epoch {} Done!! Train Loss: {}".format(epoch, error.item()))

    ## NOT needed because swa_model already contains averaged weights
    # print("Averaging weights -- SWA")
    # # swa_optimizer.swap_swa_sgd()
    # print("Done!!")

    print("Updating BN")
    # swa_optimizer.bn_update(loader=trainloader, model=model)
    torch.optim.swa_utils.update_bn(trainloader, swa_model)
    print("Done!!")


    #Check val accuracy
    print("Evaluating validation accuracy")
    swa_model.eval()

    swa_acc = test(0, swa_model, testloader)

    print("Validation Accuracy with SWA model: {}".format(swa_acc))

    print("Saving SWA model")
    
    sd = swa_model.module.state_dict() if n_gpus > 1 else swa_model.state_dict()
    #when model is on single gpu then model state_dict contains keyword module
    # So we will simply remove <module> from dictionary.
    isModuleStrPresent=False
    for k in sd.keys():
        if k.find("module.") == -1:
            continue
        isModuleStrPresent=True
        break

    if isModuleStrPresent:
        print("SWA checkpoint contains module present in keys.")
        print("So now removing 'module' strings")
        #remove module strings
        from collections import OrderedDict
        new_ckpt_dict = OrderedDict()
        for k,v in sd.items():
            tmp_key = k.replace("module.","")
            new_ckpt_dict[tmp_key] = v
        
        sd = copy.deepcopy(new_ckpt_dict)
        print("Done!!")

    #sd = model.state_dict()
    # Record the state
    checkpoint = {
        'epoch': -1, # to state the checkpoint is corresponding to SWA
        'model_state': sd,
        'optimizer_state': optimizer.state_dict(),
    }
    # Write the checkpoint
    
    checkpoint_file = os.path.join(save_path, "[SWA-pytorch]valSet_acc_{:.5f}.pyth".format(swa_acc))
    print("---Before SAVING SWA MODEL----")
    torch.save(checkpoint, checkpoint_file)
    print("SAVED SWA model")
    print("SWA Model saved at {}".format(checkpoint_file))
    return
