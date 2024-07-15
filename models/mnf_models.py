from typing import Any
import torch
from torch import nn
from torch.nn import BatchNorm1d
from models.torch_mnf.layers import MNFLinear


class EpiOnly(nn.Sequential):
    """Bayesian DNN with parameter posteriors modeled by normalizing flows."""
    def __init__(self, input_shape,**kwargs: Any) -> None:
        """Initialize the model."""
        super(EpiOnly,self).__init__()

        self.act = nn.SELU()


        self.L1 = MNFLinear(input_shape, 256,**kwargs)
        self.BN1 = BatchNorm1d(256)

        self.L2 = MNFLinear(256, 128, **kwargs)
        self.BN2 = BatchNorm1d(128)

        self.L3 = MNFLinear(128, 64, **kwargs)
        self.BN3 = BatchNorm1d(64)
        
        self.classification_head = MNFLinear(64, 1, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.act(self.BN1(self.L1(x)))
        x = self.act(self.BN2(self.L2(x)))
        x = self.act(self.BN3(self.L3(x)))
        cls = self.sigmoid(self.classification_head(x))
        return cls.squeeze(1)
    def kl_div(self) -> float:
        """Compute current KL divergence of the whole model. Given by the sum
        of KL divs. from each MNF layer. Use as a regularization term during training.
        """
        return sum(lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div"))

class MNFNet_v3(nn.Sequential):
    """Bayesian DNN with parameter posteriors modeled by normalizing flows."""

    def __init__(self,input_shape, **kwargs: Any) -> None:
        """Initialize the model."""
        super(MNFNet_v3,self).__init__()

        self.act = nn.SELU()

        self.L1 = MNFLinear(input_shape, 256,**kwargs)
        self.BN1 = BatchNorm1d(256)

        self.L2 = MNFLinear(256, 128, **kwargs)
        self.BN2 = BatchNorm1d(128)

        self.L3 = MNFLinear(128, 64, **kwargs)
        self.BN3 = BatchNorm1d(64)

        self.logit_head = MNFLinear(64, 1, **kwargs)
        self.aleatoric_head = MNFLinear(64,1 ,**kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.act(self.BN1(self.L1(x)))
        x = self.act(self.BN2(self.L2(x)))
        x = self.act(self.BN3(self.L3(x)))
        logits = self.logit_head(x)
        sigma = torch.exp(self.aleatoric_head(x)) # Log variance
        return logits.squeeze(1),sigma.squeeze(1)

    def kl_div(self) -> float:
        """Compute current KL divergence of the whole model. Given by the sum
        of KL divs. from each MNF layer. Use as a regularization term during training.
        """
        return sum(lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div"))


# We should train this and compare it to the BNN as well.
class MLP(nn.Module):
    def __init__(self,input_shape):
        super(MLP, self).__init__()

        self.L1 = nn.Linear(input_shape,256)
        self.BN1 = BatchNorm1d(256)
        self.L2 = nn.Linear(256,128)
        self.BN2 = BatchNorm1d(128)
        self.L3 = nn.Linear(128,64)
        self.BN3 = BatchNorm1d(64)
        self.classification_head = nn.Linear(64,1)

        self.sigmoid = nn.Sigmoid()
        self.act = nn.SELU()

    def forward(self,x):
        x = self.act(self.BN1(self.L1(x)))
        x = self.act(self.BN2(self.L2(x)))
        x = self.act(self.BN3(self.L3(x)))
        cls = self.sigmoid(self.classification_head(x))
        return cls.squeeze(1)
