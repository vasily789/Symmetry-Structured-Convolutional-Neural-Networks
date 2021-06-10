import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv2d_symm_pres_kernel_size1(nn.Module):
    def __init__(self):
        super(Conv2d_symm_pres_kernel_size1, self).__init__()

        self._param = nn.Parameter((1/np.sqrt(768))*torch.randn(256,128,1,1), requires_grad=True)

        self.bias = nn.Parameter(torch.zeros(256), requires_grad=True)
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1

    def forward(self, input):

        #in case we use gpu we need to create the weight matrix there
        device = self._param.device

        weight = torch.zeros((256,128,1,1)).to(device)

        weight[:,:,:,:] = self._param[:,:,:,:]

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2d_symm_pres_kernel_size3_1(nn.Module):
    def __init__(self):
        super(Conv2d_symm_pres_kernel_size3_1, self).__init__()

        self._param = nn.Parameter((1/np.sqrt(512))*torch.randn(128, 128, 2, 3), requires_grad=True)

        self.bias = nn.Parameter(torch.zeros(128), requires_grad=True)
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1

    def forward(self, input):

        #in case we use gpu we need to create the weight matrix there
        device = self._param.device

        weight = torch.zeros((128, 128, 3, 3)).to(device)

        weight[:,:,0,:] = self._param[:,:,0,:] # first row
        weight[:,:,1:3,0] = weight[:,:,0,1:3] # remaining of the first column
        weight[:,:,1,1:3] = self._param[:,:,1,0:2] # remaining of the second row
        weight[:,:,2,1] = weight[:,:,1,2] # remaining of the second column, entry (3,2)
        weight[:,:,2,2] = self._param[:,:,1,2] # remaining entry, entry (3,3)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class Conv2d_symm_pres_kernel_size3_2(nn.Module):
    def __init__(self):
        super(Conv2d_symm_pres_kernel_size3_2, self).__init__()

        self._param = nn.Parameter((1/32)*torch.randn(256, 256, 2, 3), requires_grad=True)

        self.bias = nn.Parameter(torch.zeros(256), requires_grad=True)
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1

    def forward(self, input):

        #in case we use gpu we need to create the weight matrix there
        device = self._param.device

        weight = torch.zeros((256, 256, 3, 3)).to(device)

        weight[:,:,0,:] = self._param[:,:,0,:] # first row
        weight[:,:,1:3,0] = weight[:,:,0,1:3] # remaining of the first column
        weight[:,:,1,1:3] = self._param[:,:,1,0:2] # remaining of the second row
        weight[:,:,2,1] = weight[:,:,1,2] # remaining of the second column, entry (3,2)
        weight[:,:,2,2] = self._param[:,:,1,2] # remaining entry, entry (3,3)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
