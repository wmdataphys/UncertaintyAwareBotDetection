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
from models.mnf_models import MLP
import torch.nn.functional as F
from torch.utils.data import Subset
from dataloader.create_data import create_dataset

def main(config,resume):

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
    
    X_train,X_test,X_val,y_train,y_test,y_val = create_dataset(config['dataset']['path_to_csv'])
    train_dataset = TensorDataset(torch.tensor(X_train),torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val),torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test),torch.tensor(y_test))

    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = config['run_val']
    print("Training Size: {0}".format(len(train_dataset)))
    print("Validation Size: {0}".format(len(val_dataset)))
    print("Testing Size: {0}".format(len(test_dataset)))

    train_loader,val_loader,test_loader = CreateLoaders(train_dataset,val_dataset,test_dataset,config)

     # Create the model
    net = MLP()
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')

    print(net)


    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])

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
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
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

        for i, data in enumerate(train_loader):
            inputs  = data[0].to('cuda').float()
            y  = data[1].to('cuda').float()
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                y_pred = net(inputs)

            loss = loss_fn(y_pred,y)
            train_acc = (torch.sum(torch.round(y_pred) == y)).item() / len(y)

            loss.backward()
            optimizer.step()


            running_loss += loss.item() * inputs.shape[0]

            kbar.update(i, values=[("bce",loss.item()),("Train Accuracy",train_acc)])
            global_step += 1

        scheduler.step()
        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])


        ######################
        ## validation phase ##
        ######################
        if run_val:
            net.eval()
            val_loss = 0.0
            val_acc = 0.0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs  = data[0].to('cuda').float()
                    y  = data[1].to('cuda').float()
                    y_pred = net(inputs)

                    bce = loss_fn(y_pred,y)
                    val_acc += (torch.sum(torch.round(y_pred) == y)).item() / len(y)
                    val_loss += bce
    
                val_loss /= len(val_loader)
                val_acc = val_acc/len(val_loader)

            history['val_loss'].append(val_loss)

            kbar.add(1, values=[("val_bce",val_loss.item()),("val_acc",val_acc)])

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
    parser = argparse.ArgumentParser(description='Hackaton Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume)
