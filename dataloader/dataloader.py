import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import numpy as np
import torch


# Create dataloaders to iterate.
def CreateLoaders(train_dataset,val_dataset,test_dataset,config,method=None):
    if method is None:
        print("Please specify method to dataloaders.")

    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False)
    test_loader =  DataLoader(test_dataset,
                            batch_size=config['dataloader']['test']['batch_size_'+str(method)],
                            shuffle=False)

    return train_loader,val_loader,test_loader
