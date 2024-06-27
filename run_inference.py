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
import pandas as pd
from pickle import load
from models.mnf_models import MNFNet_v3, MLP,EpiOnly
import torch.nn.functional as F
from torch.utils.data import Subset
from dataloader.create_data import create_dataset
import torch.nn.functional as F

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
    aleatoric = []
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
            logits,sigmas = net(inputs)
            #targets = net(inputs)

        #targets = targets.reshape(-1,samples).detach().cpu().numpy()
        targets = F.sigmoid(logits)
        p_sigma = F.sigmoid(logits)*(1.0 - F.sigmoid(logits)) * sigmas
        targets = targets.reshape(-1,samples).detach().cpu().numpy()
        p_sigma = p_sigma.reshape(-1,samples).detach().cpu().numpy()
        mypreds_r_MNF.append(np.mean(targets,axis=1))
        mypreds_r_MNF_std.append(np.std(targets,axis=1))
        aleatoric.append(np.mean(p_sigma,axis=1))

        kbar.update(i)
    end = time.time()
    print(" ")
    print("Elapsed Time: ",end - start)
    mypreds_r_MNF = np.concatenate(mypreds_r_MNF)
    epistemic = np.concatenate(mypreds_r_MNF_std)
    aleatoric = np.concatenate(aleatoric)
    true_y = np.concatenate(true_y)
    print(mypreds_r_MNF.shape,epistemic.shape,true_y.shape)
    print('Time per account: ',(end - start) / len(mypreds_r_MNF))
    print(" ")

    preds_frame = pd.DataFrame(mypreds_r_MNF)
    preds_frame.columns = ['y_hat']
    preds_frame['y_hat_sigma'] = epistemic
    preds_frame = preds_frame.dropna()
    preds_frame['y_true'] = true_y
    preds_frame['aleatoric'] = aleatoric

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

def main(config,mlp_eval,method):
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

    # Create directory structure
    output_folder = config['Inference']['out_dir_'+str(method)]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if method == "BLOC":
        input_shape = 193 ### Hard Coded features
        config['dataset']['path_to_csv'] = config['dataset']['BLOC']
        print("Running inference on BLOC features.")
    elif method == "BOTOMETER":
        input_shape = 1209#### Hard Coded features
        config['dataset']['path_to_csv'] = config['dataset']['BOTOMETER']
        print("Running inference on Botometer features.")
    else:
        print("Incorrect method choice. Please choose from: ")
        print("1. BLOC")
        print("2. BOTOMETER")
        exit()

    X_train, X_test, X_val, y_train, y_test, y_val, X_removed_bots, y_removed_bots= create_dataset(config['dataset']['path_to_csv'],leftover_bots=True,method=method)
    train_dataset = TensorDataset(torch.tensor(X_train),torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val),torch.tensor(y_val))
    # Stack the left over bots that we did not use.
    X_test_ = np.concatenate([X_test,X_removed_bots],axis=0)
    y_test_ = np.concatenate([y_test,y_removed_bots])
    subsets = np.concatenate([np.array(['Testing' for i in range(len(X_test))]),np.array(['Bots' for i in range(len(X_removed_bots))])])
    print("Added additional bots removed from training.")
    print(X_test_.shape,y_test_.shape)
    test_dataset = TensorDataset(torch.tensor(X_test_),torch.tensor(y_test_))

    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = False
    print("Training Size: {0}".format(len(train_dataset)))
    print("Validation Size: {0}".format(len(val_dataset)))
    print("Testing Size: {0}".format(len(test_dataset)))

    train_loader,val_loader,test_loader = CreateLoaders(train_dataset,val_dataset,test_dataset,config,method=method)
    # Remove datasets/loaders we dont need
    del train_loader,val_loader,train_dataset,val_dataset


     # Load the MNF model
    net = MNFNet_v3(input_shape)
    #net = EpiOnly()
    net.to(device)
    dict = torch.load(config['Inference']['MNF_model_'+str(method)])
    net.load_state_dict(dict['net_state_dict'])

    bayes_frame = run_bayes_eval(net,test_loader,device)
    bayes_frame = bayes_frame.dropna()
    bayes_frame['method'] = subsets

    if mlp_eval:
        mlp = MLP(input_shape)
        mlp.to(device)
        dict = torch.load(config['Inference']['DNN_model_'+str(method)])
        mlp.load_state_dict(dict['net_state_dict'])
        mlp_frame = run_mlp_eval(mlp,test_loader,device)
        mlp_frame = mlp_frame.dropna()
        preds_frame = pd.concat([bayes_frame,mlp_frame],axis=1)
        print(" ")
    else:
        preds_frame = bayes_frame.copy()

    # Save it
    save_path = os.path.join(config['Inference']['out_dir_'+str(method)],config['Inference']['out_file'])
    
    print("Output file: ",save_path)
    preds_frame.to_csv(save_path,sep=',',index=None)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m','--mlp_eval',default=0,type=int,help='Run MLP inference?')
    parser.add_argument('-M', '--method', default='BLOC', type=str,
                        help='BLOC or BOTOMETER')

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,bool(args.mlp_eval),args.method)
