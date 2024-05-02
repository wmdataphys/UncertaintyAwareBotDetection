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
#from models.maf import MAF
import pandas as pd
from pickle import load
from models.mnf_models import MNFNet_v3, MLP
import torch.nn.functional as F
from torch.utils.data import Subset
from loss_utils import BNN_Loss

def main(config):

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


    #-- targets for regression
    Y_r = np.c_[
        pandas_df['gen_log_x'].to_numpy(),
        pandas_df['gen_log_Q2'].to_numpy(),
        pandas_df['gen_log_y'].to_numpy(),
        pandas_df['from_tlv_gen_y'].to_numpy()
    ]

    print('Max and Mins for inverse transformation uncertainty propogation.')
    print("Max:",Y_r.max(0))
    print("Min:",Y_r.min(0))


    GY = pandas_df['from_tlv_gen_y'].to_numpy()

    z_scaler = load(open(os.path.join(config['scalers']['MLP'],"input_scaler.pkl"),'rb'))

    X_z = z_scaler.transform(X)

    z_scalerY = load(open(os.path.join(config['scalers']['MLP'],"target_scaler.pkl"),'rb'))

    Y_r_z = z_scalerY.transform(Y_r[:,:3])
    print('DNN Statistics')
    print(Y_r_z.shape)
    print("X: ",X_z.max(),X_z.min())
    print("Y: ",Y_r_z.max(),Y_r_z.min())
    print(Y_r_z.shape,X_z.shape)
    print( " ")

    scaler = load(open(os.path.join(config['scalers']['MNF'],"input_scaler.pkl"),'rb'))
    X_ = scaler.transform(X)

    scalerY = load(open(os.path.join(config['scalers']['MNF'],"target_scaler.pkl"),'rb'))
    Y_r_ = scalerY.transform(Y_r[:,:3])
    print('MNF Statistics')
    print(Y_r_.shape)
    print("X: ",X_.max(),X_.min())
    print("Y: ",Y_r_.max(),Y_r_.min())

    X = np.concatenate([X_z,X_],axis=1)
    Y_r = np.concatenate([Y_r_z,Y_r_],axis=1)
    Y_r = np.append(Y_r,np.c_[GY],axis=1)

    print(X.shape)
    print(Y_r.shape)

    full_dataset = TensorDataset(torch.tensor(X),torch.tensor(Y_r))

    train_ids = list(np.load(os.path.join(config['dataset']['idx_path'],"train_indices.npy")))
    val_ids = list(np.load(os.path.join(config['dataset']['idx_path'],"val_indices.npy")))
    test_ids = list(np.load(os.path.join(config['dataset']['idx_path'],"test_indices.npy")))


    train_dataset = Subset(full_dataset,train_ids)
    val_dataset = Subset(full_dataset,val_ids)
    test_dataset = Subset(full_dataset,test_ids)

    print("Training Size: {0}".format(len(train_dataset)))
    print("Validation Size: {0}".format(len(val_dataset)))
    print("Testing Size: {0}".format(len(test_dataset)))

    train_loader,val_loader,test_loader = CreateLoaders(train_dataset,val_dataset,test_dataset,config)
    # Remove datasets/loaders we dont need
    del train_loader,val_loader,train_dataset,val_dataset


     # Load the MNF model
    net = MNFNet_v3()
    net.to('cuda')
    dict = torch.load(config['Inference']['MNF_model'])
    net.load_state_dict(dict['net_state_dict'])

    # Load the MLP model
    mlp = MLP(config['model']['blocks'],config['model']['dropout_setval'])
    mlp.to('cuda') # If no GPU, replace 'cuda' with 'cpu' wherever cuda is seen.
    dict = torch.load(config['Inference']['DNN_model'])
    mlp.load_state_dict(dict['net_state_dict'])




    kbar = pkbar.Kbar(target=len(test_loader),width=20, always_stateful=False)
    # This performs sampling for the MNF and runs MLP evaluation
    mypreds_r_MNF = []
    mypreds_r_MNF_std = []
    Y_r_test = []
    GY_test = []
    net.eval()
    mlp.eval()
    feats = []
    mus = []
    samples = 10000
    mypreds_r = []
    mypreds_r_MNF_aleatoric = []

    for i,data in enumerate(test_loader):

        mlp_inputs = data[0][:,:15].to('cuda').float()

        with torch.set_grad_enabled(False):
            mlp_outputs = mlp(mlp_inputs)

        mypreds_r.append(mlp_outputs.detach().cpu().numpy())
        inputs = data[0][:,15:].numpy()

        temp = []
        for j in range(len(inputs)):
            temp.append(np.expand_dims(inputs[j],0).repeat(samples,0))

        inputs = torch.tensor(np.concatenate(temp)).to('cuda').float()
        y = data[1].to('cuda').float()
        Y_r_test.append(y.detach().cpu().numpy()[:,:6])
        GY_test.append(y.detach().cpu().numpy()[:,6])


        with torch.set_grad_enabled(False):
            targets,log_devs2 = net(inputs)
        for q in range(config['dataloader']['test']['batch_size']):
            de = targets.detach().cpu().numpy()[q*samples:(q+1)*samples]
            da = torch.exp(log_devs2).detach().cpu().numpy()[q*samples:(q+1)*samples]
            mypreds_r_MNF.append(de.mean(0))
            mypreds_r_MNF_std.append(de.std(0))
            mypreds_r_MNF_aleatoric.append(np.sqrt(da.mean(0)))
        #if i == 1000:
        #    break

        kbar.update(i)

    mypreds_r_MNF = np.array(mypreds_r_MNF)
    aleatoric = np.array(mypreds_r_MNF_aleatoric)
    epistemic = np.array(mypreds_r_MNF_std)
    Y_r_test = np.concatenate(Y_r_test)
    GY_test = np.concatenate(GY_test)
    mypreds_r = np.concatenate(mypreds_r)

    preds_frame = pd.DataFrame(mypreds_r_MNF)
    preds_frame.columns = ['x','Q2','y']
    preds_frame = preds_frame.dropna()


    error_dataframe = pd.DataFrame(epistemic)
    error_dataframe.columns = ['x_epistemic','Q2_epistemic','y_epistemic']
    error_dataframe['x_aleatoric'] = aleatoric[:,0]
    error_dataframe['Q2_aleatoric'] = aleatoric[:,1]
    error_dataframe['y_aleatoric'] = aleatoric[:,2]
    error_dataframe = error_dataframe.dropna()


    mlp_frame = pd.DataFrame(mypreds_r)
    mlp_frame.columns = ['x_mlp','Q2_mlp','y_mlp']


    true_frame = pd.DataFrame(Y_r_test)
    true_frame.columns = ['x_true_MLP','Q2_true_MLP','y_true_MLP','x_true_MNF','Q2_true_MNF','y_true_MNF']
    true_frame['tlv_gen_y'] = GY_test

    total_dataframe = pd.concat([preds_frame,error_dataframe,mlp_frame,true_frame],axis=1)


    # Save it
    save_path = os.path.join(config['Inference']['out_dir'],config['Inference']['out_file'])
    total_dataframe.to_csv(save_path,sep=',',index=None)


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config)
