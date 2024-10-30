import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

act_dict = {"tanh": nn.Tanh(),"relu": nn.ReLU(),"sigmoid": nn.Sigmoid(),"gelu": nn.GELU()}

init_dict={"xavier_normal": nn.init.xavier_normal_,"xavier_uniform": nn.init.xavier_uniform_,}


class Branch(nn.Module): # rho
    def __init__(self, width):
        super(Branch, self).__init__()
        self.padding = 8
        self.fc0 = nn.Linear(1, width)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3) # batch_size, width, nx,ny,nz
        x = F.pad(x, [0,self.padding, 0,self.padding, 0,self.padding]) # padding for last 3 dimensions (nx,ny,nz)
        x = x.permute(0, 2, 3, 4, 1) # batch_size, nx,ny,nz, width

        return x

class Trunk(nn.Module): # frequency
    def __init__(self, width):
        super(Trunk, self).__init__()
        self.fc0 = nn.Linear(1, width)

    def forward(self, x):

        x = self.fc0(x)

        return x


class BranchTrunk(nn.Module):
    def __init__(self, width):
        super(BranchTrunk, self).__init__()
        self.branch = Branch(width)
        self.trunk = Trunk(width)

    def forward(self, branch_Rho, trunk_Freq):
        
        x1 = self.branch(branch_Rho)
        x2 = self.trunk(trunk_Freq)

        n1 = x1.shape[1]
        n2 = x1.shape[2]
        n3 = x1.shape[3]
        n4 = x1.shape[4]
        x = torch.einsum("bwxyz,cz->bcwxyz", [x1, x2])
        x = x.view(-1, n1, n2, n3, n4)

        return x


class Unet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        if out_conv2.shape != out_deconv2.shape:
            diff_z = out_conv2.shape[2] - out_deconv2.shape[2]
            diff_y = out_conv2.shape[3] - out_deconv2.shape[3]
            diff_x = out_conv2.shape[4] - out_deconv2.shape[4]

            out_deconv2 = F.pad(out_deconv2, [diff_x // 2, diff_x - diff_x // 2,
                                              diff_y // 2, diff_y - diff_y // 2,
                                              diff_z // 2, diff_z - diff_z // 2])
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        if out_conv1.shape != out_deconv1.shape:
            diff_z = out_conv1.shape[2] - out_deconv1.shape[2]
            diff_y = out_conv1.shape[3] - out_deconv1.shape[3]
            diff_x = out_conv1.shape[4] - out_deconv1.shape[4]

            out_deconv1 = F.pad(out_deconv1, [diff_x // 2, diff_x - diff_x // 2,
                                              diff_y // 2, diff_y - diff_y // 2,
                                              diff_z // 2, diff_z - diff_z // 2])

        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        if x.shape != out_deconv0.shape:
            diff_z = x.shape[2] - out_deconv0.shape[2]
            diff_y = x.shape[3] - out_deconv0.shape[3]
            diff_x = x.shape[4] - out_deconv0.shape[4]

            out_deconv0 = F.pad(out_deconv0, [diff_x // 2, diff_x - diff_x // 2,
                                              diff_y // 2, diff_y - diff_y // 2,
                                              diff_z // 2, diff_z - diff_z // 2])

        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):

        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x



class FNO(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, layer_fno, layer_ufno, act_func):
        super(FNO, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.layer_fno = layer_fno
        self.layer_ufno = layer_ufno
        if act_func in act_dict.keys():
            self.activation = act_dict[act_func]
        else:
            raise KeyError("act name not in act_dict")
        
        self.fno = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ufno = nn.ModuleList()

        for _ in range(layer_fno+layer_ufno):
            self.fno.append(SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3))
            self.conv.append(nn.Conv3d(self.width, self.width, 1))

        for _ in range(layer_ufno):
            self.ufno.append(Unet(self.width, self.width, 3, 0 ))
        
    def forward(self, x):

        x = x.permute(0, 4, 1, 2, 3) # batch_size, width, n1,n2,n3

        for i in range(self.layer_fno):
            x1 = self.fno[i](x)
            x2 = self.conv[i](x)
            x  = x1 + x2
            x = self.activation(x)

        for i in range(self.layer_fno, self.layer_fno+self.layer_ufno):
            x1 = self.fno[i](x)
            x2 = self.conv[i](x)
            x3 = self.ufno[i-self.layer_fno](x)
            x  = x1 + x2 + x3
            x = self.activation(x)
        
        x = x.permute(0, 2, 3, 4, 1) # batch_size, n1, n2, width

        return x
    
class HDF(nn.Module):
    def __init__(self, modes1, modes2, modes3,width, nout, layer_sizes, nLoc, init_func, layer_fno=3, layer_ufno=3, act_func="gelu"):
        super(HDF, self).__init__()

        self.width = width
        self.padding = 8
        self.nout = nout
        if act_func in act_dict.keys():
            self.activation = act_dict[act_func]
        else:
            raise KeyError("act name not in act_dict")
        if init_func in init_dict.keys():
            initializer = init_dict[init_func]
        else:
            raise KeyError("init name not in init_dict")
        self.BranchTrunk = BranchTrunk(width)
        self.fno = FNO(modes1, modes2, modes3, width, layer_fno, layer_ufno, act_func)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, nout)
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear( layer_sizes[i - 1], layer_sizes[i]))
            initializer(self.linears[-1].weight)
            nn.init.zeros_(self.linears[-1].bias)

    def forward(self, x, freq):
        # print(x.shape)
        batchSize = x.shape[0]
        nFreq = freq.shape[0]
        x = self.BranchTrunk(x, freq)
        x = self.fno(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x[:, :-self.padding, :-self.padding, :-self.padding, :]
        x = x.contiguous().view(batchSize, nFreq,  -1, self.nout)
        x = x.permute(0, 1, 3, 2)
        for i in range(len(self.linears)-1):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)
        x = x.permute(0, 1, 3, 2)

        return x
