import torch
import torch.functional as F

def BNN_Loss_Phys(preds,log_devs2,y,scaler,log_S,log_Q2):
    unscaled_logs = torch.tensor(scaler.inverse_transform(preds.detach().cpu().numpy())).to('cuda').float()
    phys_loss = (log_Q2 - (log_S + unscaled_logs[:,0] + unscaled_logs[:,2]))**2
    fn = (y - preds) ** 2
    return 0.5*(torch.exp(-log_devs2) * fn + log_devs2).mean(),fn.mean(),phys_loss.mean()


def BNN_Loss(preds,log_devs2,y):
    fn = (y - preds) ** 2
    return 0.5*(torch.exp(-log_devs2) * fn + log_devs2).mean(),fn.mean()
