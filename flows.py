import torch
import torch.nn as nn

class PlanarTransformation(nn.Module):
    
    def __init__(self, dim):
        super(PlanarTransformation, self).__init__()
        
        # activation functions
        self.h = torch.tanh
        self.h_prime = lambda x: 1-torch.tanh(x)**2
        self.m = lambda x: -1 + torch.log1p(torch.exp(x)) 
        
        # parameters
        self.u = torch.nn.Parameter(torch.randn(dim, 1))
        self.w = torch.nn.Parameter(torch.randn(dim, 1))
        self.b = torch.nn.Parameter(torch.randn(1,))
    
    def forward(self, z):
        # enforce invertibility
        wTu = self.w.T@self.u
        u_hat = self.u + (self.m(wTu)-wTu) * self.w / torch.norm(self.w)**2
        
        # calculate transformation
        z_prime = z + u_hat.T*self.h(z@self.w+self.b)
        
        # calculate log-determinant
        psi = self.h_prime(z@self.w+self.b)*self.w.T
        logdet = torch.log(torch.abs(1 + psi@u_hat) + 1e-8)
        return z_prime, logdet
    
class RadialTransformation(nn.Module):
    
    def __init__(self, dim):
        super(RadialTransformation, self).__init__()
        # parameters
        self.dim = dim
        
        # activation functions
        self.h = torch.tanh
        self.h_prime = lambda x: 1-torch.tanh(x)**2
        self.m = lambda x: torch.log1p(torch.exp(x)) 
        
        # parameters
        self.z_0 = torch.nn.Parameter(torch.randn(1, dim))
        self.pre_alpha = torch.nn.Parameter(torch.randn(1,))
        self.beta = torch.nn.Parameter(torch.randn(1,))
    
    def forward(self, z):
        # enforce invertibility
        alpha = torch.exp(self.pre_alpha)
        beta_hat = -alpha + self.m(self.beta)
        
        # calculate transformation
        diff = z-self.z_0
        r = torch.abs(diff)
        h = 1/(alpha+r)
        z_prime = z + beta_hat*h*(diff)
        
        # calculate log-determinant
        h_prime  = -1/(alpha+r)**2
        logdet = (self.dim - 1)*torch.log1p(beta_hat*h) + torch.log1p(beta_hat*h+beta_hat*h_prime*r)
        
        return z_prime, logdet
    
class RadialFlow(nn.Module):
    
    def __init__(self, dim, K):
        super(RadialFlow, self).__init__()
        
        # create sequence of planar transforms
        self.main = nn.ModuleList([RadialTransformation(dim) for k in range(K)])
        
    def forward(self, z):
        sum_logdet = 0.0
        for transformation in self.main:
            z, logdet = transformation(z) 
            sum_logdet += logdet
            
        return z, sum_logdet
    
class PlanarFlow(nn.Module):
    
    def __init__(self, dim, K):
        super(PlanarFlow, self).__init__()
        
        # create sequence of planar transforms
        self.main = nn.ModuleList([PlanarTransformation(dim) for k in range(K)])
        
    def forward(self, z):
        sum_logdet = 0.0
        for transformation in self.main:
            z, logdet = transformation(z) 
            sum_logdet += logdet
            
        return z, sum_logdet
        