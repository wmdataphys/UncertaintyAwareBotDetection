import os
import json
import argparse
import torch
import random
import numpy as np
from datetime import datetime
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataloader.dataloader import CreateLoaders
from pickle import dump
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
#from models.maf import MAF
import pandas as pd
from pickle import load
from models.mnf_models import MNFNet_v3, MLP
import torch.nn.functional as F
from torch.utils.data import Subset
from loss_utils import BNN_Loss
from dataloader.create_data import create_dataset

def run_bayes_eval(net,test_loader,device):
    kbar = pkbar.Kbar(target=len(test_loader),width=20, always_stateful=False)
    # This performs sampling for the MNF and runs MLP evaluation
    mypreds_r_MNF = []
    mypreds_r_MNF_std = []
    net.eval()
    feats = []
    mus = []
    samples = 10000
    mypreds_r = []
    true_y = []
    start = time.time()
    for i,data in enumerate(test_loader):

        inputs = data[0].numpy()

        temp = []
        for j in range(len(inputs)):
            temp.append(np.expand_dims(inputs[j],0).repeat(samples,0))

        inputs = torch.tensor(np.concatenate(temp)).to(device).float()
        y = data[1].numpy()
        true_y.append(y)

        with torch.set_grad_enabled(False):
            targets = net(inputs)

        targets = targets.reshape(-1,samples).detach().cpu().numpy()
        mypreds_r_MNF.append(np.mean(targets,axis=1))
        mypreds_r_MNF_std.append(np.std(targets,axis=1))

        kbar.update(i)
    end = time.time()
    print("Elapsed Time: ",end - start)
    mypreds_r_MNF = np.concatenate(mypreds_r_MNF)
    epistemic = np.concatenate(mypreds_r_MNF_std)
    true_y = np.concatenate(true_y)
    print(mypreds_r_MNF.shape,epistemic.shape,true_y.shape)
    print('Time per account: ',(end - start) / len(mypreds_r_MNF))

    preds_frame = pd.DataFrame(mypreds_r_MNF)
    preds_frame.columns = ['y_hat']
    preds_frame['y_hat_sigma'] = epistemic
    preds_frame = preds_frame.dropna()
    preds_frame['y_true'] = true_y

    return preds_frame


def run_mlp_eval(net,test_loader,device):
    kbar = pkbar.Kbar(target=len(test_loader),width=20, always_stateful=False)
    # This performs sampling for the MNF and runs MLP evaluation
    mlp_preds = []
    true_y = []
    for i,data in enumerate(test_loader):
        inputs = data[0].to(device).float()
        y = data[1].numpy()
        true_y.append(y)

        with torch.set_grad_enabled(False):
            targets = net(inputs).detach().cpu().numpy()

        mlp_preds.append(targets)

        kbar.update(i)

    mlp_preds = np.concatenate(mlp_preds)
    true_y = np.concatenate(true_y)

    preds_frame = pd.DataFrame(mlp_preds)
    preds_frame.columns = ['y_hat_mlp']
    preds_frame = preds_frame.dropna()
    preds_frame['y_true_mlp'] = true_y
    return preds_frame

def main(config,mlp_eval):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not mlp_eval:
        print("Running only Bayesian evalution. See argparser.")
    else:
        print("Running Bayesian and DNN evaluation.")

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
    output_folder = config['Inference']['out_dir']

    X_train,X_test,X_val,y_train,y_test,y_val = create_dataset(config['dataset']['path_to_csv'])
    train_dataset = TensorDataset(torch.tensor(X_train),torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val),torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test),torch.tensor(y_test))

    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = False
    print("Training Size: {0}".format(len(train_dataset)))
    print("Validation Size: {0}".format(len(val_dataset)))
    print("Testing Size: {0}".format(len(test_dataset)))

    train_loader,val_loader,test_loader = CreateLoaders(train_dataset,val_dataset,test_dataset,config)
    # Remove datasets/loaders we dont need
    del train_loader,val_loader,train_dataset,val_dataset


     # Load the MNF model
    net = MNFNet_v3()
    net.to(device)
    dict = torch.load(config['Inference']['MNF_model'])
    net.load_state_dict(dict['net_state_dict'])

    bayes_frame = run_bayes_eval(net,test_loader,device)

    if mlp_eval:
        mlp = MLP()
        mlp.to(device)
        dict = torch.load(config['Inference']['DNN_model'])
        mlp.load_state_dict(dict['net_state_dict'])
        mlp_frame = run_mlp_eval(mlp,test_loader,device)

    preds_frame = pd.concat([bayes_frame,mlp_frame],axis=1)
    # Save it
    save_path = os.path.join(config['Inference']['out_dir'],config['Inference']['out_file'])
    preds_frame.to_csv(save_path,sep=',',index=None)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m','--mlp_eval',default=0,type=int,help='Run MLP inference?')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,bool(args.mlp_eval))
