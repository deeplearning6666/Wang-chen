import torch
from torch import nn
import torch.nn.functional as F

##卷积层设置
class ComplexConv2d(nn.Module):
    def __init__(self,input_channels,output_channels,
                 kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True):
        super(ComplexConv2d,self).__init__()
        self.conv_real = nn.Conv2d(input_channels,output_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.conv_imag = nn.Conv2d(input_channels,output_channels,kernel_size,stride,padding,dilation,groups,bias)

    def forward(self,input_real,input_imag):
        assert input_real.shape == input_imag.shape
        return (self.conv_real(input_real) - self.conv_imag(input_imag)),(
            self.conv_imag(input_real) + self.conv_real(input_imag))

##非线性激活层
class ComplexPReLU(nn.Module):
    def __init__(self,slope=0.2):
        super(ComplexPReLU,self).__init__()
        self.slope = slope
    
    def lrelu(self,x):
        outt = torch.max(self.slope * x,x)
        return outt
    
    def forward(self,x_real,x_imag):
        return self.lrelu(x_real),self.lrelu(x_imag)
    
class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU,self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x_real,x_imag):
        return self.relu(x_real),self.relu(x_imag)
    
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x_real, x_imag):
        x_real = self.bn_re(x_real)
        x_imag = self.bn_im(x_imag)
        return x_real, x_imag

##卷积层加归一化层加非线性激活层
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = ComplexBatchNorm2d(out_ch)
        self.relu = ComplexReLU()

    def forward(self, x_real, x_imag):
        x_real, x_imag = self.conv(x_real, x_imag)
        x_real, x_imag = self.relu(x_real, x_imag)
        x_real, x_imag = self.bn(x_real, x_imag)
        return x_real, x_imag
    
class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = ComplexConv2d(ch, ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = ComplexBatchNorm2d(ch)
        self.relu = ComplexReLU()
        self.conv2 = ComplexConv2d(ch, ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = ComplexBatchNorm2d(ch)

    def forward(self, x_real, x_imag):
        res_real, res_imag = x_real, x_imag
        res_real, res_imag = self.bn1(res_real, res_imag)
        res_real, res_imag = self.relu(res_real, res_imag)
        res_real, res_imag = self.conv1(res_real, res_imag)
        # res_real, res_imag = self.bn2(res_real, res_imag)
        # res_real, res_imag = self.conv2(res_real, res_imag)
        return res_real + x_real, res_imag + x_imag

    
class resnet(nn.Module):
    def __init__(self,in_channel=1,out_channel=1,depth=5,ngf=256,sample=False,weight=None,freeze_weight=False,Ny=None,Nx=None):
        super(resnet,self).__init__()

        self.Nx = Nx
        self.Ny = Ny
        self.inchannel = in_channel
        self.sample = sample
        self.weights = nn.ParameterList()
        if sample:
            for _ in range(in_channel):
                param = nn.Parameter(torch.rand(Ny,Nx))
                if freeze_weight:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                self.weights.append(param)
        self.depth = depth
        self.head = ConvBNReLU(in_ch=in_channel,out_ch=ngf,kernel_size=5,stride=1,padding=2)
        self.body1 = ResBlock(ngf,kernel_size=5,stride=1,padding=2)

        if self.depth >=2:
            self.body2 = ResBlock(ngf,kernel_size=5,stride=1,padding=2)
        if self.depth >=3:
            self.body3 = ResBlock(ngf,kernel_size=5,stride=1,padding=2)
        if self.depth >=4:
            self.body4 = ResBlock(ngf,kernel_size=3,stride=1,padding=1)
        if self.depth >=5:
            self.body5 = ResBlock(ngf,kernel_size=3,stride=1,padding=1)
        if self.depth >=6:
            self.body6 = ResBlock(ngf,kernel_size=3,stride=1,padding=1)
        if self.depth >=7:
            self.body7 = ResBlock(ngf,kernel_size=3,stride=1,padding=1)
        if self.depth >=8:
            self.body8 = ResBlock(ngf,kernel_size=3,stride=1,padding=1)
        if self.depth >=9:
            self.body9 = ResBlock(ngf,kernel_size=3,stride=1,padding=1)
        if self.depth >=10:
            self.body10 = ResBlock(ngf,kernel_size=5,stride=1,padding=2)
    
        self.tail = ComplexConv2d(ngf,out_channel,kernel_size=5,stride=1,padding=2)

    def forward(self,x_real,x_imag):
        weights = torch.zeros(self.inchannel,self.Nx,self.Ny,dtype=torch.bfloat16).to(device='cuda:0')
        if self.sample:
            for _ in range(self.inchannel):
                weights[_,:,:] = self.weights[_]
        if self.sample:
            x_real = x_real * weights
            x_imag = x_imag * weights
        
        x_real,x_imag = self.head(x_real,x_imag)
        x_real,x_imag = self.body1(x_real,x_imag)

        if self.depth >= 2:
            x_real,x_imag = self.body2(x_real,x_imag)
        if self.depth >= 3:
            x_real,x_imag = self.body3(x_real,x_imag)
        if self.depth >= 5:
            x_real,x_imag = self.body5(x_real,x_imag)
        if self.depth >= 6:
            x_real,x_imag = self.body6(x_real,x_imag)
        if self.depth >= 7:
            x_real,x_imag = self.body7(x_real,x_imag)
        if self.depth >= 8:
            x_real,x_imag = self.body8(x_real,x_imag)
        if self.depth >= 9:
            x_real,x_imag = self.body9(x_real,x_imag)
        if self.depth >= 10:
            x_real,x_imag = self.body10(x_real,x_imag)
        

        x_real,x_imag = self.tail(x_real,x_imag)


        return x_real,x_imag


if __name__ == '__main__':
    import torch
    net = resnet(depth=1)
    x_real = torch.rand(1,1,128,128)
    x_imag = torch.rand(1,1,128,128)
    x_real,y_imag = net(x_real,x_imag)
    print(net)
