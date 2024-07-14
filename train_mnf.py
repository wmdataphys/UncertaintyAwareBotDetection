import os
import json
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataloader.dataloader import CreateLoaders
from pickle import dump
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import pandas as pd
from pickle import load
from models.mnf_models import MNFNet_v3
import torch.nn.functional as F
from torch.utils.data import Subset
from dataloader.create_data import create_dataset

def main(config,resume,method):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    if not os.path.exists(config["output"]["dir"]):
        os.makedirs(config["output"]["dir"])

    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    if not os.path.exists(os.path.join(output_folder,exp_name)):
        os.mkdir(os.path.join(output_folder,exp_name))
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)


    # Load the dataset
    print('Creating Loaders.')
    if method == "BLOC":
        input_shape = 182 ### Hard Coded features
        config['dataset']['path_to_csv'] = config['dataset']['BLOC']
        print("Training on BLOC features.")
    elif method == "BOTOMETER":
        input_shape = 1209#### Hard Coded features
        config['dataset']['path_to_csv'] = config['dataset']['BOTOMETER']
        print("Training on Botometer features.")
    else:
        print("Incorrect method choice. Please choose from: ")
        print("1. BLOC")
        print("2. BOTOMETER")
        exit()

    X_train,X_test,X_val,y_train,y_test,y_val = create_dataset(config['dataset']['path_to_csv'],method=method)
    train_dataset = TensorDataset(torch.tensor(X_train),torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val),torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test),torch.tensor(y_test))
    print(X_train.shape)

    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = config['run_val']
    print("Training Size: {0}".format(len(train_dataset)))
    print("Validation Size: {0}".format(len(val_dataset)))
    print("Testing Size: {0}".format(len(test_dataset)))

    train_loader,val_loader,test_loader = CreateLoaders(train_dataset,val_dataset,test_dataset,config,method=method)

     # Create the model
    
    net = MNFNet_v3(input_shape)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')

    #print(net)


    # Optimizer
    num_epochs=int(config['num_epochs'])
    lr = float(config['optimizer']['lr'])
    weight_decay = float(config['optimizer']['mnf_weight_decay'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,weight_decay=weight_decay)
    num_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)

    

    startEpoch = 0
    global_step = 0

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      weight_decay:',weight_decay)
    print('      num_epochs:', num_epochs)
    print('      KL_scale:',config['optimizer']['KL_scale'])
    print('')

    # Train


    # Define your loss function

    loss_fn = torch.nn.BCELoss() # Utilizes the predefined BNN loss

    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0
        num_samples = config['dataloader']['train']['num_samples']
        for i, data in enumerate(train_loader):
            inputs  = data[0].to('cuda').float()
            y  = data[1].to('cuda').float()
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                logits,sigma = net(inputs)
                logits = logits.unsqueeze(-1).expand(*logits.shape,num_samples)
                sigma = sigma.unsqueeze(-1).expand(*sigma.shape,num_samples)
                sampled_logits = logits + sigma * torch.normal(mean=0.0, std=1.0, size=logits.shape, device=logits.device)
                sampled_p = F.sigmoid(sampled_logits.mean(dim=-1))

            bce = loss_fn(sampled_p,y)
            kl_div = config['optimizer']['KL_scale']*net.kl_div() / len(train_loader)
            loss = bce + kl_div 

            train_acc = (torch.sum(torch.round(sampled_p) == y)).item() / len(y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.shape[0]

            kbar.update(i, values=[("loss", loss.item()),("bce",bce.item()),("kl_loss",kl_div.item()),("Train Accuracy",train_acc)])
            global_step += 1

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])


        ######################
        ## validation phase ##
        ######################
        if run_val:
            net.eval()
            val_loss = 0.0
            val_kl_div = 0.0
            val_acc = 0.0
            val_bce = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs  = data[0].to('cuda').float()
                    y  = data[1].to('cuda').float()
                    logits,sigma = net(inputs)
                    logits = logits.unsqueeze(-1).expand(*logits.shape,num_samples)
                    sigma = sigma.unsqueeze(-1).expand(*sigma.shape,num_samples)
                    sampled_logits = logits + sigma * torch.normal(mean=0.0, std=1.0, size=logits.shape, device=logits.device)
                    sampled_p = F.sigmoid(sampled_logits.mean(dim=-1))

                    bce = loss_fn(sampled_p,y)
                    kl_div = config['optimizer']['KL_scale']*net.kl_div() / len(train_loader)
                    loss = bce + kl_div 
                    val_acc += (torch.sum(torch.round(sampled_p) == y)).item() / len(y)

                    val_loss += loss
                    val_bce += bce
                    val_kl_div += kl_div

                val_kl_div = val_kl_div / len(val_loader)
                val_bce /= len(val_loader)
                val_loss /= len(val_loader)
                val_loss = val_loss.cpu().numpy()
                val_acc = val_acc/len(val_loader)

            history['val_loss'].append(val_loss)

            kbar.add(1, values=[("val_loss" ,val_loss),("val_bce",val_bce.item()),("val_kl_loss",val_kl_div.item()),("val_acc",val_acc)])

            name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.6f}.pth'.format(epoch, val_loss)

        else:
            kbar.add(1,values=[('val_loss',0.)])
            name_output_file = config['name']+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

        filename = os.path.join(output_folder , exp_name , name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        print('')




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-M', '--method', default='BLOC', type=str,
                        help='BLOC or BOTOMETER')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume,args.method)
