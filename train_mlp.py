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
from loss_utils import BNN_Loss

def main(config,resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    os.mkdir(os.path.join(output_folder,exp_name))
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)


    # Load the dataset
    print('Creating Loaders.')
    pandas_df = pd.read_csv(config['dataset']['path_to_csv'],sep=',',index_col=None)
    X = np.c_[
        pandas_df['e_ecal_over_trk_ratio'].to_numpy(),
        pandas_df['n_towers_40'].to_numpy(),
        pandas_df['eta_pho_closest_to_ebeam'].to_numpy(),
        pandas_df['e_pho_closest_to_ebeam'].to_numpy(),
        pandas_df['dphi_pho_closest_to_ebeam'].to_numpy(),
        pandas_df['obs_e_pz'].to_numpy(),
        pandas_df['obs_e_e'].to_numpy(),
        pandas_df['obs_hfs_pz'].to_numpy(),
        pandas_df['obs_hfs_e'].to_numpy(),
        pandas_df['rot_pt1'].to_numpy(),
        pandas_df['rot_Empz1'].to_numpy(),
        pandas_df['rot_pt2'].to_numpy(),
        pandas_df['obs_pzbal'].to_numpy(),
        pandas_df['obs_ptbal'].to_numpy(),
        pandas_df['obs_dphi'].to_numpy(),
    ]

    x_ = pandas_df['from_tlv_gen_x'].to_numpy()
    y_ = pandas_df['from_tlv_gen_y'].to_numpy()
    Q2_ = pandas_df['from_tlv_gen_Q2'].to_numpy()
    log_S = np.log(Q2_/(x_*y_))
    gen_log_Q2 = pandas_df['gen_log_Q2'].to_numpy()
    #-- targets for regression
    Y_r = np.c_[
        pandas_df['gen_log_x'].to_numpy(),
        pandas_df['gen_log_Q2'].to_numpy(),
        pandas_df['gen_log_y'].to_numpy()
    ]


    GY = pandas_df['from_tlv_gen_y'].to_numpy()
    pth = os.path.join(output_folder,'%s-scalers' % config['name'])

    print("Creating StandardScalers.")
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    scalerY = StandardScaler()
    scalerY.fit(Y_r)
    Y_r = scalerY.transform(Y_r)
    Y_r = np.append(Y_r,np.c_[log_S],axis=1)
    Y_r = np.append(Y_r,np.c_[gen_log_Q2],axis=1)
    print("X: ",X.max(),X.min())
    print("Y: ",Y_r.max(),Y_r.min())
    try:
        os.mkdir(pth)
    except:
        print('\n  Dir %s-scalers already exists\n\n' % pth )


    print('\n\n Saving the input and learning target scalers:\n')
    print('    %s-scalers/input_scaler.pkl' % config['name'] )
    print('    %s-scalers/target_scaler.pkl' % config['name'] )

    dump( scaler, open(os.path.join(pth,'input_scaler.pkl') , 'wb'))
    dump( scalerY, open(os.path.join(pth,'target_scaler.pkl') , 'wb'))



    print("No files specified, using a split of 70/15/15%")
    full_dataset = TensorDataset(torch.tensor(X),torch.tensor(Y_r))
    train_ids = list(np.load(os.path.join(config['dataset']['idx_path'],"athena_train_indices.npy")))
    val_ids = list(np.load(os.path.join(config['dataset']['idx_path'],"athena_val_indices.npy")))
    test_ids = list(np.load(os.path.join(config['dataset']['idx_path'],"athena_test_indices.npy")))

    train_dataset = Subset(full_dataset,train_ids)
    val_dataset = Subset(full_dataset,val_ids)
    test_dataset = Subset(full_dataset,test_ids)
    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = True
    print("Training Size: {0}".format(len(train_dataset)))
    print("Validation Size: {0}".format(len(val_dataset)))
    print("Testing Size: {0}".format(len(test_dataset)))

    train_loader,val_loader,test_loader = CreateLoaders(train_dataset,val_dataset,test_dataset,config)

     # Create the model
    net = MLP(config['model']['blocks'],config['model']['dropout_setval'])
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

    loss_fn = torch.nn.HuberLoss(reduction='mean', delta=config['optimizer']['huber_delta'])

    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs  = data[0].to('cuda').float()
            y  = data[1][:,:3].to('cuda').float()
            log_S_cond = data[1][:,-2].to('cuda').float()
            log_Q2 = data[1][:,-1].to('cuda').float()
            # reset the gradient
            optimizer.zero_grad()

            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                targets = net(inputs)

            loss = loss_fn(targets,y)
            # backprop
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.shape[0]

            kbar.update(i, values=[("loss", loss.item())])

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
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs  = data[0].to('cuda').float()
                    y  = data[1][:,:3].to('cuda').float()
                    log_S_cond = data[1][:,-2].to('cuda').float()
                    log_Q2 = data[1][:,-1].to('cuda').float()

                    targets= net(inputs)
                    val_loss += loss_fn(targets,y)

                val_loss = val_loss.cpu().numpy()/len(val_loader)

            history['val_loss'].append(val_loss)

            kbar.add(1, values=[("val_loss" ,val_loss)])

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
