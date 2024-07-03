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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving the model
from sklearn.metrics import classification_report
import warnings
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import matplotlib.pyplot as plt

def main(config,method):

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

    X_train, X_test, X_val, y_train, y_test, y_val, X_removed_accounts, y_removed_accounts, account_type = create_dataset(config['dataset']['path_to_csv'],leftover_accounts=True,method=method)


    # Train Random Forest model
    print('Training Random Forest model.')
    rf_model = RandomForestClassifier(n_estimators=100, random_state=config['seed'],verbose=0)
    rf_model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(output_folder, exp_name, 'random_forest_model.joblib')
    joblib.dump(rf_model, model_path)
    print(f'Model saved to {model_path}')

    # Validate the model
    val_predictions = rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f'Validation Accuracy: {val_accuracy}')
    report = classification_report(y_val, val_predictions, target_names=['Human', 'Bot'])
    print(report)
    print(" ")

    # Evaluate the model on the test set
    test_predictions = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f'Test Accuracy: {test_accuracy}')
    report = classification_report(y_test, test_predictions, target_names=['Human', 'Bot'])
    print(report)
    print(" ")

    print("Testing on only leftover {0}.".format(account_type))
    test_account_predictions = rf_model.predict(X_removed_accounts)
    report = classification_report(y_removed_accounts, test_account_predictions, target_names=['Human', 'Bot'],zero_division=0)
    print(report)
    print(" ")

    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob,drop_intermediate=False)
    auc_ = auc(fpr,tpr)
    print("RF AUC: ",auc_)

    out_dir = config['Inference']['out_dir_'+str(method)]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fig = plt.figure(figsize=(9,6))
    plt.plot(fpr,tpr,color='red', lw=2,linestyle='--', label='RF ROC Curve. (area = %0.3f)' % auc_)
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc="best",fontsize=16)
    plt.xticks(fontsize=18)  # adjust fontsize as needed
    plt.yticks(fontsize=18)  # adjust fontsize as needed
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.grid(True)
    out_path_DLL_ROC = os.path.join(out_dir,"ROC_RF.pdf")
    plt.savefig(out_path_DLL_ROC,bbox_inches='tight')
    plt.close()

    


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='RF Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-M', '--method', default='BLOC', type=str,
                        help='BLOC or BOTOMETER')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.method)