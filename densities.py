import torch
import numpy as np

def density_1(z):
    z1, z2 = z[:, 0], z[:, 1]
    first_term = 0.5*((torch.norm(z, dim=1)-2)/0.4)**2
    second_term = torch.exp(-0.5*((z1-2)/0.6)**2)
    third_term = torch.exp(-0.5*((z1+2)/0.6)**2)
    total = first_term - torch.log(second_term + third_term+1e-10)
    return torch.exp(-total)

def density_2(z):
    z1, z2 = z[:, 0], z[:, 1]
    w1 = torch.sin(2*np.pi*z1/4)
    total = 0.5*((z2-w1)/0.4)**2
    return torch.exp(-total)

def density_3(z):
    z1, z2 = z[:, 0], z[:, 1]
    w1 = torch.sin(2*np.pi*z1/4)
    w2 = 3*torch.exp(-0.5*((z1-1)/0.6)**2)
    first_term = torch.exp(-0.5*((z2-w1)/0.35)**2)
    second_term = torch.exp(-0.5*((z2-w1+w2)/0.35)**2)
    total = -torch.log(first_term + second_term+1e-10)
    return torch.exp(-total)

def density_4(z):
    z1, z2 = z[:, 0], z[:, 1]
    w1 = torch.sin(2*np.pi*z1/4)
    w2 = 3*torch.exp(-0.5*((z1-1)/0.6)**2)
    w3 = 3*torch.sigmoid((z1-1)/0.3)
    
    first_term = torch.exp(-0.5*((z2-w1)/0.4)**2)
    second_term = torch.exp(-0.5*((z2-w1+w3)/0.35)**2)
    total = -torch.log(first_term + second_term+1e-10)
    return torch.exp(-total)

def get_density(num):
    if num == 1:
        density = density_1
    elif num == 2:
        density = density_2
    elif num == 3:
        density = density_3
    elif num == 4:
        density = density_4
    return density