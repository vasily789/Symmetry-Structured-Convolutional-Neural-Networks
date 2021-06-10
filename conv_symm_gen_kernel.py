import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv2d_symm_gen_kernel_ml1m(nn.Module):
    def __init__(self):
        super(Conv2d_symm_gen_kernel_ml1m, self).__init__()

        self._param = nn.Parameter((1/np.sqrt(356))*torch.randn(128,50,1,1), requires_grad=True) 

        self.bias = nn.Parameter(torch.zeros(128), requires_grad=True)
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1

    def forward(self, input):

        #in case we use gpu we need to create the weight matrix there
        device = self._param.device

        weight = torch.zeros((128,100,1,1)).to(device)

        weight[:,0:50,:,:] = self._param[:,:,:,:]
        weight[:,50:100,:,:] = weight[:,0:50,:,:]

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2d_symm_gen_kernel_gowalla(nn.Module):
    def __init__(self):
        super(Conv2d_symm_gen_kernel_gowalla, self).__init__()

        self._param = nn.Parameter((1/np.sqrt(456))*torch.randn(128,100,1,1), requires_grad=True) 

        self.bias = nn.Parameter(torch.zeros(128), requires_grad=True)
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1

    def forward(self, input):

        #in case we use gpu we need to create the weight matrix there
        device = self._param.device

        weight = torch.zeros((128,200,1,1)).to(device)

        weight[:,0:100,:,:] = self._param[:,:,:,:]
        weight[:,100:200,:,:] = weight[:,0:100,:,:]

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
