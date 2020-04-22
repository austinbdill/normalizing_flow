import torch

def FreeEnergyBound(z, sum_logdets, density):
    return (-torch.log(density(z)).unsqueeze(1) - sum_logdets).mean()