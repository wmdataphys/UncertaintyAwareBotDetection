from typing import Any
import torch
from torch import nn
from torch.nn import BatchNorm1d
from models.torch_mnf.layers import MNFLinear

class MNFNet_v3(nn.Sequential):
    """Bayesian DNN with parameter posteriors modeled by normalizing flows."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model."""
        super(MNFNet_v3,self).__init__()
        self.act = nn.SELU()
        self.L1 = MNFLinear(133, 64,**kwargs)
        self.BN1 = BatchNorm1d(64)
        self.L2 = MNFLinear(64, 128, **kwargs)
        self.BN2 = BatchNorm1d(128)
        self.L3 = MNFLinear(128, 256, **kwargs)
        self.BN3 = BatchNorm1d(256)
        self.L4 = MNFLinear(256, 128, **kwargs)
        self.BN4 = BatchNorm1d(128)
        self.L5 = MNFLinear(128, 64, **kwargs)
        self.BN5 = BatchNorm1d(64)
        self.classification_head = MNFLinear(64, 1, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.act(self.BN1(self.L1(x)))
        x = self.act(self.BN2(self.L2(x)))
        x = self.act(self.BN3(self.L3(x)))
        x = self.act(self.BN4(self.L4(x)))
        x = self.act(self.BN5(self.L5(x)))
        cls = self.sigmoid(self.classification_head(x))
        return cls.squeeze(1)

    def kl_div(self) -> float:
        """Compute current KL divergence of the whole model. Given by the sum
        of KL divs. from each MNF layer. Use as a regularization term during training.
        """
        return sum(lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div"))


class MLP(nn.Module):
    def __init__(self,blocks,dropout_setval):
        super(MLP, self).__init__()

        self.init_layer = nn.Linear(15,blocks[0])
        self.init_act = nn.ReLU()
        #[64,128,512,1024,512,128,64,3],
        self.L1 = nn.Linear(blocks[0],blocks[1])  # 64   -> 128
        self.L2 = nn.Linear(blocks[1],blocks[2])  # 128  -> 256
        self.L5 = nn.Linear(blocks[4],blocks[5])  # 256  -> 128
        self.L6 = nn.Linear(blocks[5],blocks[6])  # 128 -> 64
        self.L7 = nn.Linear(blocks[6],blocks[7])  # 64 -> 3
        self.act = nn.SELU()
        self.drop = nn.Dropout(p=dropout_setval)

    def forward(self,x):

        # Input process
        x = self.init_layer(x)
        x = self.init_act(x)
        x = self.drop(x)

        x = self.L1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L5(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L6(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.L7(x)

        return x
