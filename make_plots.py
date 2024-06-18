import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from matplotlib.colors import LogNorm

def plot_auc_mlp(y_pred,y_true,out_folder):
    fpr,tpr,thresholds = roc_curve(y_true,y_pred,drop_intermediate=False)
    auc_ = auc(fpr,tpr)
    print("DNN AUC: ",auc_)
    plt.figure()
    plt.plot(fpr,tpr,color='red', lw=2,linestyle='--', label='ROC Curve. (area = %0.2f)' % auc_)
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic (ROC) Curve - DNN', fontsize=25, pad=20)
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc="best",fontsize=16)
    plt.ylim(0,1)
    plt.xticks(fontsize=18)  # adjust fontsize as needed
    plt.yticks(fontsize=18)  # adjust fontsize as needed
    out_path_DLL_ROC = os.path.join(out_folder,"ROC_MLP.pdf")
    plt.savefig(out_path_DLL_ROC,bbox_inches='tight')
    plt.close()


def plot_auc_bayes(y_pred,sigma,y_true,out_folder,n_strap=1000):
    # Bootstrapping AUC
    fprs = []
    tprs = []
    aucs = []

    for i in range(n_strap):
        y_pred_temp = np.random.normal(loc=y_pred, scale=sigma)
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

    # ROC Curve
    fig,ax = plt.subplots(1,3,figsize=(24,6),sharey=True)
    ax = ax.ravel()

    ax[0].plot(mean_fpr, mean_tpr, color='k', lw=1,linestyle='--' ,label=r'ROC curve (area = {0:.3f} $\pm$ {1:.3f})'.format(roc_auc,roc_auc_sigma))
    ax[0].fill_between(mean_fpr, mean_tpr - 5 * tpr_sigma, mean_tpr + 5 * tpr_sigma, color='blue', alpha=0.5, label=r'$5\sigma$ TPR Band')
    ax[0].fill_betweenx(mean_tpr, mean_fpr - 5 * fpr_sigma, mean_fpr + 5 * fpr_sigma, color='red', alpha=0.5, label=r'$5\sigma$ FPR Band')
    ax[0].plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate', fontsize=25)
    ax[0].set_ylabel('True Positive Rate', fontsize=25)
    ax[0].legend(loc="lower right", fontsize=16)


    ax[1].plot(mean_fpr, mean_tpr, color='k', lw=1,linestyle='--' ,label=r'ROC curve (area = {0:.3f} $\pm$ {1:.3f})'.format(roc_auc,roc_auc_sigma))
    ax[1].fill_between(mean_fpr, mean_tpr - 5 * tpr_sigma, mean_tpr + 5 * tpr_sigma, color='blue', alpha=0.5, label=r'$5\sigma$ TPR Band')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    ax[1].set_xlabel('False Positive Rate', fontsize=25)
    ax[1].set_title('Receiver Operating Characteristic (ROC) Curve - BNN', fontsize=25, pad=20)
    ax[1].legend(loc="lower right", fontsize=16)


    ax[2].plot(mean_fpr, mean_tpr, color='k', lw=1,linestyle='--' ,label=r'ROC curve (area = {0:.3f} $\pm$ {1:.3f})'.format(roc_auc,roc_auc_sigma))
    ax[2].fill_betweenx(mean_tpr, mean_fpr - 5 * fpr_sigma, mean_fpr + 5 * fpr_sigma, color='red', alpha=0.5, label=r'$5\sigma$ FPR Band')
    ax[2].set_xlim([0.0, 1.0])
    ax[2].set_ylim([0.0, 1.05])
    ax[2].plot([0, 1], [0, 1], color='k', lw=1, linestyle='-')
    ax[2].set_xlabel('False Positive Rate', fontsize=25)
    ax[2].legend(loc="lower right", fontsize=16)

    for ax_i in ax:
        ax_i.tick_params(axis='both', which='major', labelsize=18)
        ax_i.tick_params(axis='both', which='minor', labelsize=16)
        ax_i.grid(True)

    plt.subplots_adjust(wspace=0.1)
    out_path_DLL_ROC = os.path.join(out_folder, "ROC.pdf")
    plt.savefig(out_path_DLL_ROC, bbox_inches='tight')
    plt.close()


def validate_uncertainty(y_pred, sigma,y_true, out_folder):
    idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[idx]
    sigma_sorted = sigma[idx]
    plt.figure()
    plt.hist2d(y_pred_sorted, sigma_sorted, bins=50, cmap='magma',density=True,norm=LogNorm())
    cb = plt.colorbar(label='Log Density')
    cb.set_label('Log Density', fontsize=25)
    cb.ax.tick_params(labelsize=18)
    plt.xlabel('Probability', fontsize=25)
    plt.ylabel('Epistemic Uncertainty', fontsize=25)
    plt.title(r'Epistemic Uncertainty as Function of Probability', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    #plt.grid(True)
    out_path_uncertainty = os.path.join(out_folder, 'uncertainty.pdf')
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
    plt.title(r'Epistemic Uncertainty as Function of Probability - False Negatives', fontsize=25, pad=20)
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
    plt.title(r'Epistemic Uncertainty as Function of Probability - False Positives', fontsize=25, pad=20)
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
    plt.title(r'Epistemic Uncertainty as Function of Probability - Correctly Identified', fontsize=25, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0,1)
    #plt.grid(True)
    out_path_uncertainty = os.path.join(out_folder, 'uncertainty_correct_samples.pdf')
    plt.savefig(out_path_uncertainty, bbox_inches='tight')
    plt.close()



def main(config,mlp_eval):

    out_dir = config['Inference']['plot_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Plots can be found in " + str(out_dir))
    if os.path.exists(os.path.join(config['Inference']['out_dir'],config['Inference']['out_file'])):
        results = pd.read_csv(os.path.join(config['Inference']['out_dir'],config['Inference']['out_file']),sep=',',index_col=None)
    else:
        print("Please run inference first.")
        exit()

    predictions = results['y_hat'].to_numpy()
    sigma = results['y_hat_sigma'].to_numpy()
    y_true = results['y_true'].to_numpy()

    plot_auc_bayes(predictions,sigma,y_true,out_dir)
    validate_uncertainty(predictions,sigma,y_true,out_dir)


    if mlp_eval:
        y_pred_mlp = results['y_hat_mlp'].to_numpy()
        y_true_mlp = results['y_true_mlp'].to_numpy()

        plot_auc_mlp(y_pred_mlp,y_true_mlp,out_dir)


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Plotting')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m','--mlp_eval',default=0,type=int,help='Run MLP eval?')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,bool(args.mlp_eval))
