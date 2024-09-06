import torch
from torch import nn, Tensor
from torch.nn import functional as F
import pandas as pd
#import heatmap

torch.manual_seed(32)

def load_imf_adj(hour:int):
    file_path = f'data/norm_imf/normalized_imf_adj_hour_{hour}.csv'
    imf_mx = pd.read_csv(file_path,header=None)
    return torch.tensor(imf_mx.values, dtype=torch.float32, device='cuda:0')

class MiLearner(nn.Module):
    def __init__(self, n_hist: int, n_in: int, node_dim: int, dropout: float):
        super(MiLearner, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.imf_tensors = [load_imf_adj(hour) for hour in range(24)]
        self.hourly_weights = nn.ParameterList(nn.Parameter(torch.randn(1)) for _ in range(24))


    def forward(self, inputs: Tensor):
        """
        :param inputs: tensor, [B, T, N, F]
        :param supports: tensor, [E, N, N]
        :return: tensor, [E, B, N, N]
        """
        hours = (inputs[:, 0, 0, 1] * 24).to(torch.int)
        imf_adjs = [self.imf_tensors[hour.item()] * torch.clamp(self.hourly_weights[hour.item()],min=0) for hour in hours]
        A_mi = torch.stack(imf_adjs,dim=0)
        return  A_mi


class GraphLearner(nn.Module):
    def __init__(self, supports: Tensor, n_hist: int, n_in: int, node_dim: int, dropout: float, learn_macro: bool, learn_micro: bool):
        super(GraphLearner, self).__init__()
        self.adaptive = nn.Parameter(supports, requires_grad=learn_macro)

        if learn_micro:
            self.mi_learner = MiLearner(n_hist, n_in, node_dim, dropout)

    def forward(self, inputs: Tensor = None) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, F]
        :return: tensor, [E, N, N] or [E, B, N, N]
        """
        supports = self.adaptive

        if hasattr(self, 'mi_learner'):
            #Multi-level Graph Structure Fusion
            supports = supports.unsqueeze(1) + self.mi_learner(inputs)
    
        return  F.normalize(torch.relu(supports), p=1, dim=-1)


