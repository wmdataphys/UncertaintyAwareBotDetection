import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from matplotlib.colors import LogNorm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import joblib
from dataloader.create_data import create_dataset

def plot_auc_mlp(y_pred,y_true,out_folder):
    fpr,tpr,thresholds = roc_curve(y_true,y_pred,drop_intermediate=False)
    auc_ = auc(fpr,tpr)
    print("DNN AUC: ",auc_)
    report = classification_report(y_true, y_pred.round(), target_names=['Human', 'Bot'])
    print(" ")
    print(report)
    print(" ")
    fig = plt.figure(figsize=(9,6))
    plt.plot(fpr,tpr,color='red', lw=2,linestyle='--', label='DNN ROC Curve. (area = %0.3f)' % auc_)
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
    out_path_DLL_ROC = os.path.join(out_folder,"ROC_MLP.pdf")
    plt.savefig(out_path_DLL_ROC,bbox_inches='tight')
    plt.close()

    return (fpr,tpr,auc_)


def plot_auc_bayes(y_pred,sigma,y_true,aleatoric,out_folder,n_strap=1000):
    # Bootstrapping AUC
    fprs = []
    tprs = []
    aucs = []
    scale = np.sqrt(sigma**2 + aleatoric ** 2)
    for i in range(n_strap):
        y_pred_temp = np.random.normal(loc=y_pred, scale=scale)
        fpr_, tpr_, thresholds = roc_curve(y_true, y_pred_temp,drop_intermediate=False)
        fprs.append(fpr_)
        tprs.append(tpr_)
        aucs.append(auc(fpr_, tpr_))

    # This will be shaped like (n_strap,thresholds)
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    # Mean over thresholds
    mean_fpr = np.mean(fprs,axis=0)
    mean_tpr = np.mean(tprs,axis=0)
    tpr_sigma = np.std(tprs,axis=0)
    fpr_sigma = np.std(fprs,axis=0)
    roc_auc = np.mean(aucs)
    roc_auc_sigma = np.std(aucs)

    print("Bayes AUC: ",roc_auc," +=",roc_auc_sigma)
    report = classification_report(y_true, y_pred.round(), target_names=['Human', 'Bot'])
    print(" ")
    print(report)
    print(" ")

    # ROC Curve
    fig= plt.subplots(figsize=(9,6))

    plt.plot(mean_fpr, mean_tpr, color='k', lw=1,linestyle='--' ,label=r'BNN ROC curve (area = {0:.3f} $\pm$ {1:.3f})'.format(roc_auc,roc_auc_sigma))
    plt.fill_between(mean_fpr, mean_tpr - 5 * tpr_sigma, mean_tpr + 5 * tpr_sigma, color='blue', alpha=0.5, label=r'$5\sigma$ TPR Band')
    plt.fill_betweenx(mean_tpr, mean_fpr - 5 * fpr_sigma, mean_fpr + 5 * fpr_sigma, color='red', alpha=0.5, label=r'$5\sigma$ FPR Band')
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc="lower right", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.grid(True)
    out_path_DLL_ROC = os.path.join(out_folder, "ROC_FPR_TPR_Bands.pdf")
    plt.savefig(out_path_DLL_ROC, bbox_inches='tight')
    plt.close()

    fig= plt.subplots(figsize=(9,6))
    plt.plot(mean_fpr, mean_tpr, color='k', lw=1,linestyle='--' ,label=r'BNN ROC curve (area = {0:.3f} $\pm$ {1:.3f})'.format(roc_auc,roc_auc_sigma))
    plt.fill_between(mean_fpr, mean_tpr - 5 * tpr_sigma, mean_tpr + 5 * tpr_sigma, color='blue', alpha=0.5, label=r'$5\sigma$ TPR Band')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.legend(loc="lower right", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.grid(True)
    out_path_DLL_ROC = os.path.join(out_folder, "ROC_TPR_Band.pdf")
    plt.savefig(out_path_DLL_ROC, bbox_inches='tight')
    plt.close()

    fig= plt.subplots(figsize=(9,6))
    plt.plot(mean_fpr, mean_tpr, color='k', lw=1,linestyle='--' ,label=r'BNN ROC curve (area = {0:.3f} $\pm$ {1:.3f})'.format(roc_auc,roc_auc_sigma))
    plt.fill_betweenx(mean_tpr, mean_fpr - 5 * fpr_sigma, mean_fpr + 5 * fpr_sigma, color='red', alpha=0.5, label=r'$5\sigma$ FPR Band')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.grid(True)
    out_path_DLL_ROC = os.path.join(out_folder, "ROC_FPR_Band.pdf")
    plt.savefig(out_path_DLL_ROC, bbox_inches='tight')
    plt.close()

    return (mean_fpr,fpr_sigma,mean_tpr,tpr_sigma,roc_auc,roc_auc_sigma)

def validate_uncertainty(y_pred, sigma,y_true,aleatoric, out_folder):
    # Epistemic uncertainty
    idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[idx]
    sigma_sorted = sigma[idx]
    plt.figure()
    plt.hist2d(y_pred_sorted, sigma_sorted, bins=50, cmap='magma',density=True,norm=LogNorm(),range=[[0,1],[0,0.45]])
    cb = plt.colorbar(label='Log Density')
    cb.set_label('Log Density', fontsize=25)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('Epistemic Uncertainty', fontsize=25)
    #plt.title(r'Epistemic Uncertainty as Function of Probability', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    #plt.grid(True)
    plt.ylim(0,0.45)
    out_path_uncertainty = os.path.join(out_folder, 'epistemic_uncertainty.pdf')
    plt.savefig(out_path_uncertainty, bbox_inches='tight')
    plt.close()

    # Aleatoric Uncertainty 
    idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[idx]
    sigma_sorted = aleatoric[idx]
    plt.figure()
    plt.hist2d(y_pred_sorted, sigma_sorted, bins=50, cmap='magma',density=True,norm=LogNorm(),range=[[0,1],[0,0.45]])
    cb = plt.colorbar(label='Log Density')
    cb.set_label('Log Density', fontsize=25)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('Aleatoric Uncertainty', fontsize=25)
    #plt.title(r'Aleatoric Uncertainty as Function of Probability', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    plt.ylim(0,0.45)
    #plt.grid(True)
    out_path_uncertainty = os.path.join(out_folder, 'aleatoric_uncertainty.pdf')
    plt.savefig(out_path_uncertainty, bbox_inches='tight')
    plt.close()

    # Quadrature Uncertainty 
    quad = np.sqrt(sigma** 2 + aleatoric ** 2)
    idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[idx]
    sigma_sorted = quad[idx]
    plt.figure()
    plt.hist2d(y_pred_sorted, sigma_sorted, bins=50, cmap='magma',density=True,norm=LogNorm(),range=[[0,1],[0,0.45]])
    cb = plt.colorbar(label='Log Density')
    cb.set_label('Log Density', fontsize=25)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel(r'$\sigma_{epi.} \oplus \sigma_{alea.}$', fontsize=25)
    #plt.title(r'Total Uncertainty as Function of Probability', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    plt.ylim(0,0.45)
    #plt.grid(True)
    out_path_uncertainty = os.path.join(out_folder, 'quadrature_uncertainty.pdf')
    plt.savefig(out_path_uncertainty, bbox_inches='tight')
    plt.close()

    # Isolate false negatives
    idx = np.where((y_pred.round() == 0.0) & (y_true == 1.0))[0]
    false_negatives = y_pred[idx]
    sigma_fn = sigma[idx]

    idx = np.argsort(false_negatives)
    false_negatives_sorted = false_negatives[idx]
    sigma_fn_sorted = sigma_fn[idx]
    plt.figure()
    plt.hist2d(false_negatives_sorted, sigma_fn_sorted, bins=50, cmap='magma',density=True,norm=LogNorm())
    cb = plt.colorbar(label='Log Density')
    cb.set_label('Log Density', fontsize=25)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('Epistemic Uncertainty', fontsize=25)
    #plt.title(r'Epistemic Uncertainty as Function of Probability - False Negatives', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    #plt.grid(True)
    out_path_uncertainty = os.path.join(out_folder, 'uncertainty_false_negatives.pdf')
    plt.savefig(out_path_uncertainty, bbox_inches='tight')
    plt.close()


    # Isolate false positives
    idx = np.where((y_pred.round() == 1.0) & (y_true == 0.0))[0]
    false_positives = y_pred[idx]
    sigma_fp = sigma[idx]

    idx = np.argsort(false_positives)
    false_positives_sorted = false_positives[idx]
    sigma_fp_sorted = sigma_fp[idx]
    plt.figure()
    plt.hist2d(false_positives_sorted, sigma_fp_sorted, bins=50, cmap='magma',density=True,norm=LogNorm())
    cb = plt.colorbar(label='Log Density')
    cb.set_label('Log Density', fontsize=25)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('Epistemic Uncertainty', fontsize=25)
    #plt.title(r'Epistemic Uncertainty as Function of Probability - False Positives', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    #plt.grid(True)
    out_path_uncertainty = os.path.join(out_folder, 'uncertainty_false_positives.pdf')
    plt.savefig(out_path_uncertainty, bbox_inches='tight')
    plt.close()

    # Isolate correctly classified samples
    idx = np.where((y_pred.round() == y_true))[0]
    correct_samples = y_pred[idx]
    sigma_correct = sigma[idx]

    idx = np.argsort(correct_samples)
    correct_samples_sorted = correct_samples[idx]
    sigma_correct_sorted = sigma_correct[idx]
    plt.figure()
    plt.hist2d(correct_samples_sorted, sigma_correct_sorted, bins=50, cmap='magma',density=True,norm=LogNorm())
    cb = plt.colorbar(label='Log Density')
    cb.set_label('Log Density', fontsize=25)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('Epistemic Uncertainty', fontsize=25)
    #plt.title(r'Epistemic Uncertainty as Function of Probability - Correctly Identified', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    #plt.grid(True)
    out_path_uncertainty = os.path.join(out_folder, 'uncertainty_correct_samples.pdf')
    plt.savefig(out_path_uncertainty, bbox_inches='tight')
    plt.close()

def plot_loss(path_,method=None,out_dir="./"):
    if method is None:
        print("Please specify method in loss plotting.")
        exit()

    model_dictionary = torch.load(path_)
    train_loss = model_dictionary['history']['train_loss']
    val_loss = model_dictionary['history']['val_loss']

    plt.plot(train_loss,color='red',linestyle='--',linewidth=2,label='Training Loss')
    plt.plot(val_loss,color='blue',linestyle='--',linewidth=2,label='Validation Loss')
    plt.xlabel("Epoch",fontsize=25)
    plt.ylabel('Loss',fontsize=25)
    plt.legend(loc='best')
    plt.title("Loss - {0}".format(str(method)),fontsize=25)
    plt.savefig(os.path.join(out_dir,str(method) + '_loss.pdf'),bbox_inches='tight')
    plt.close()


def load_rf_model_and_compute_roc(model_path, X_test, y_test):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    rf_model = joblib.load(model_path)
    y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, drop_intermediate=False)
    auc_ = auc(fpr, tpr)

    return (fpr, tpr, auc_)


def plot_comparison(DNN_stats,BNN_stats,RF_stats,out_folder,method=None):
    if method == "BLOC":
        title = 'BLOC'
    elif method == "BOTOMETER":
        title = 'Botometer'
    else:
        title = None

    fpr_DNN, tpr_DNN,auc_DNN = DNN_stats
    mean_fpr,fpr_sigma,mean_tpr,tpr_sigma,auc_BNN,auc_BNN_sigma = BNN_stats
    fpr, tpr, auc_RF = RF_stats

    fig= plt.subplots(figsize=(8,8))

    plt.plot(mean_fpr, mean_tpr, color='k', lw=3,linestyle='--' ,label=r'BNN ROC curve (area = {0:.3f} $\pm$ {1:.3f})'.format(auc_BNN,auc_BNN_sigma))
    plt.fill_between(mean_fpr, mean_tpr - 5 * tpr_sigma, mean_tpr + 5 * tpr_sigma, color='grey', alpha=0.5, label=r'$5\sigma$ Band')
    plt.fill_betweenx(mean_tpr, mean_fpr - 5 * fpr_sigma, mean_fpr + 5 * fpr_sigma, color='grey', alpha=0.5, label=None)
    plt.plot(fpr_DNN,tpr_DNN,color='red',lw=3,linestyle='--',label=r'DNN ROC curve (area = {0:.3f})'.format(auc_DNN))
    plt.plot(fpr,tpr,color='blue',lw=3,linestyle='--',label=r'RF ROC curve (area = {0:.3f})'.format(auc_RF))
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=28)
    plt.ylabel('True Positive Rate', fontsize=28)
    plt.title(title,fontsize=32)
    plt.legend(loc="lower right", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.grid(True)
    out_path_DLL_ROC = os.path.join(out_folder, "ROC_Overlayed.pdf")
    plt.savefig(out_path_DLL_ROC, bbox_inches='tight')
    plt.close()

def main(config,mlp_eval,method,comparison):

    out_dir = config['Inference']['out_dir_'+str(method)]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Plots can be found in " + str(out_dir))
    if os.path.exists(os.path.join(config['Inference']['out_dir_'+str(method)],config['Inference']['out_file'])):
        results_ = pd.read_csv(os.path.join(config['Inference']['out_dir_'+str(method)],config['Inference']['out_file']),sep=',',index_col=None)
    else:
        print("Please run inference first.")
        exit()

    results = results_[results_.method == 'Testing']
    bots_only = results_[results_.method == 'Excess']

    predictions = results['y_hat'].to_numpy()
    sigma = results['y_hat_sigma'].to_numpy()
    y_true = results['y_true'].to_numpy()
    aleatoric = results['aleatoric'].to_numpy()
    print("# 0's (Humans): ",len(y_true[y_true == 0]))
    print("# 1's (Bots): ",len(y_true[y_true == 1]))
    print("Total: ",len(y_true))

    BNN_stats = plot_auc_bayes(predictions,sigma,y_true,aleatoric,out_dir)
    validate_uncertainty(predictions,sigma,y_true,aleatoric,out_dir)

    plot_loss(config['Inference']['MNF_model_'+str(method)],"BNN",out_dir=out_dir)

    print("BNN performance on excess accounts:")
    predictions = bots_only['y_hat'].to_numpy()
    sigma = bots_only['y_hat_sigma'].to_numpy()
    y_true = bots_only['y_true'].to_numpy()
    aleatoric = bots_only['aleatoric'].to_numpy()
    print("# 0's (Humans): ",len(y_true[y_true == 0]))
    print("# 1's (Bots): ",len(y_true[y_true == 1]))
    print("Total: ",len(y_true))
    report = classification_report(y_true, predictions.round(), target_names=['Human', 'Bot'],zero_division=0)
    print(report)
    print(" ")
    print("------------------------------------------------------")
    if mlp_eval:
        y_pred_mlp = results['y_hat_mlp'].to_numpy()
        y_true_mlp = results['y_true_mlp'].to_numpy()

        DNN_stats = plot_auc_mlp(y_pred_mlp,y_true_mlp,out_dir)
        plot_loss(config['Inference']['DNN_model_'+str(method)],"DNN",out_dir=out_dir)

        print("MLP performance on excess accounts:")
        y_pred_mlp = bots_only['y_hat_mlp'].to_numpy()
        y_true_mlp = bots_only['y_true_mlp'].to_numpy()
        report = classification_report(y_true_mlp, y_pred_mlp.round(), target_names=['Human', 'Bot'],zero_division=0)
        print(report)

    if comparison and mlp_eval:
        if method == "BLOC":
            input_shape = 182 ### Hard Coded features
            config['dataset']['path_to_csv'] = config['dataset']['BLOC']
        elif method == "BOTOMETER":
            input_shape = 1209#### Hard Coded features
            config['dataset']['path_to_csv'] = config['dataset']['BOTOMETER']
        else:
            print("Incorrect method choice. Please choose from: ")
            print("1. BLOC")
            print("2. BOTOMETER")
            exit()

        print("comparison is True, mlp_eval is True.")
        print("Overlaying BNN, DNN, and RF.")
        X_train, X_test, X_val, y_train, y_test, y_val, X_removed_accounts, y_removed_accounts, account_type = create_dataset(config['dataset']['path_to_csv'],leftover_accounts=True,method=method)
        RF_stats = load_rf_model_and_compute_roc(config['Inference']['RF_model_'+str(method)],X_test,y_test)
        plot_comparison(DNN_stats,BNN_stats,RF_stats,out_dir,method=method)

    if comparison and not mlp_eval:
        print("Please set mlp_eval = 1 to run full comparison.")
        exit()



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Plotting')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m','--mlp_eval',default=0,type=int,help='Run MLP eval?')
    parser.add_argument('-r','--comparison',default=0,type=int,help='Compare RF,DNN,BNN?')
    parser.add_argument('-M', '--method', default='BLOC', type=str,
                        help='BLOC or BOTOMETER')

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,bool(args.mlp_eval),args.method,args.comparison)
