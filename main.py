import os
import tqdm
from TV import TVloss
import torch
import argparse
import numpy as np
from CNN import resnet
from torch.autograd import Variable
from scipy.io import loadmat, savemat
from torchvision.utils import save_image
from torchvision import transforms
from model import sar_model

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder',default='./MMW-v1/data/')
parser.add_argument('--data_name',default='data050_plane')
parser.add_argument('--data_name_1',default='data100_plane')
parser.add_argument('--save_intermediate',default=True,type=bool)
parser.add_argument('--use_init',default=True,type=bool)
parser.add_argument('--f_start',default=348e9,type=float,help='start freq')
parser.add_argument('--f_stop',default=372e9,type=float,help='stop freq')
parser.add_argument('--z_range',default=0.5,type=float,help='the range of target')
parser.add_argument('--Nf',default=2048,type=int,help='freq points of array')
parser.add_argument('--Dx',default=0.4,type=int,help='horizontal points of array')
parser.add_argument('--Dy',default=0.4,type=int,help='vertical points of array')
parser.add_argument('--dx',default=0.002,type=float,help='horizontal scan interval')
parser.add_argument('--dy',default=0.002,type=float,help='vertical scan interval')
parser.add_argument('--z0',default=2,type=float,help='distance between object and array')
parser.add_argument('--Nx',default=200,type=int,help='horizontal resolution of reconstructed images')
parser.add_argument('--Ny',default=200,type=int,help='vertical resolution of reconstructed images')
parser.add_argument('--Nz',default=95,type=int,help='in-depth resolution of reconstructed images')
parser.add_argument('--n_iter',default=1000,type=int,help='total iterations')
parser.add_argument('--device',default='cuda:0',help='default cuda:0')
parser.add_argument('--lr',default=0.1,type=float,help='learning rate')
parser.add_argument('--net_depth',default=8,type=int,help='depth of resbolck')
parser.add_argument('--sparse weight',default=0,type=float)
parser.add_argument('--tv_weight',default=5e-3,type=float)
parser.add_argument('--enable_autocast',default=False,type=bool)
opt = parser.parse_args()

device = torch.device(opt.device)
data = loadmat(opt.data_folder+opt.data_name+'.mat')
data_1 = loadmat(opt.data_folder+opt.data_name_1+'.mat')
S_echo_real = data['real']
S_echo_imag = data['imag']
S_echo_1 = data_1['sig']
S_echo_1 = torch.from_numpy(S_echo_1.astype('float32')).to(device)
S_echo_real = torch.from_numpy(S_echo_real.astype('float32'))
S_echo_imag = torch.from_numpy(S_echo_imag.astype('float32'))
S_echo = torch.complex(real=S_echo_real,imag=S_echo_imag).to(device)


mask = data['mask']
mask = torch.from_numpy(mask.astype('float32')).unsqueeze(0).to(device)
mask = mask[:,:,:,0:opt.Nz].permute(0,3,1,2)
sar_model = sar_model(f_start=opt.f_start,
                      f_stop=opt.f_stop,
                      Dx = opt.Dx,
                      Dy = opt.Dy,
                      Nx = opt.Nx,
                      Ny = opt.Ny,
                      z0 = opt.z0,
                      z_range = opt.z_range,
                      dx = opt.dx,
                      dy = opt.dy,
                      Nz = opt.Nf,
                      dev = torch.device('cuda:0'),
                     )
sar_input = sar_model(S_echo)
input_real = sar_input.real.permute(2,0,1).unsqueeze(0)
input_imag = sar_input.imag.permute(2,0,1).unsqueeze(0)
input_init = torch.ones(opt.Nz,opt.Nx,opt.Ny).unsqueeze(0)
input_real = Variable(input_real,requires_grad=True).to(device)
input_imag = Variable(input_imag,requires_grad=True).to(device)
input_init = Variable(input_init,requires_grad=True).to(device)
sar_measure = sar_model(S_echo_1)
measurement = Variable(sar_measure,requires_grad=True).permute(2,0,1).unsqueeze(0).to(device)

input_depth = input_init.shape[1]
net = resnet(in_channel=input_depth,out_channel=opt.Nz,depth=opt.net_depth,Nx=opt.Nx,Ny=opt.Ny).to(device).bfloat16()

criterion = torch.nn.L1Loss().to(device)
tv1_loss = TVloss(type=1).to(device)

optimizer = torch.optim.Adam(net.parameters(),lr=opt.lr)
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=100,min_lr=5e-6)
scaler = torch.amp.GradScaler()

if not os.path.exists('./Result'):
    os.makedirs('./Result')

pbar = tqdm.tqdm(range(opt.n_iter))
for epoch in range(opt.n_iter):
    optimizer.zero_grad()
    output_real,output_imag = net(input_real.bfloat16(),input_imag.bfloat16())
    output_real_sparse = output_real*mask
    output_imag_sparse = output_imag*mask
    measurement_sparse = measurement*mask

    if opt.enable_autocast:
        with torch.amp.autocast(device_type='cuda'):
            loss_meas = criterion(output_real_sparse,measurement.real) + criterion(output_imag_sparse,measurement.imag)
            loss_tv = tv1_loss(output_real)+ tv1_loss(output_imag)
            loss = loss_meas + opt.tv_weight*loss_tv
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_meas = criterion(output_real_sparse,measurement_sparse.real) + criterion(output_imag_sparse,measurement_sparse.imag)
        loss_tv = (tv1_loss(output_real) + tv1_loss(output_imag))
        loss = loss_meas + opt.tv_weight*loss_tv
        loss.backward()
        optimizer.step()
        schedular.step(loss_meas)

    if(loss_meas <= 1e-3) | (optimizer.param_groups[0]['lr'] <= 5e-6):
        break

    pbar.set_description("iteration:{},lr:{},fidelity loss:{:2f},tv loss:{:2f}"
                         .format(epoch,optimizer.param_groups[0]['lr'],loss_meas.item(),loss_tv.item()))
    pbar.update(1)
    output_real = output_real.type(torch.float32)
    output_imag = output_imag.type(torch.float32)

    with torch.no_grad():
        if(opt.save_intermediate) and (epoch>0) and (epoch % 100 ==0):
            sar_output = torch.complex(real=output_real,imag=output_imag).squeeze(0).permute(2,1,0).unsqueeze(0).unsqueeze(0)
            #sar_output = sar_model(sar_output).unsqueeze(0).unsqueeze(0)
            abs_output = torch.abs(sar_output)                                                      
            front_view = torch.max(abs_output,dim=4).values
            front_view = front_view/torch.max(front_view)
            save_image(front_view,"./MMW-v1/Result/{}_iteration{}.jpg".format(opt.data_name,epoch))
        if epoch == opt.n_iter:
            sar_output = torch.complex(real=output_real,imag=output_imag).squeeze(0).permute(2,1,0).unsqueeze(0).unsqueeze(0)
            #sar_output = sar_model(sar_output).unsqueeze(0).unsqueeze(0)
            abs_output = torch.abs(sar_output)
            front_view = torch.max(abs_output,dim=4).values
            front_view = front_view/torch.max(front_view)
            proj_db = 20*torch.log10(front_view + 1e-10)
            proj_db = transforms.functional.resize(proj_db,(800,800))
            proj_db[proj_db < -20] = -20
            proj_db += 20
            proj_db /= 20
            save_image(proj_db[0], "./MMW-v1/Result/{}_to_detection.jpg".format(opt.data_name))
sar_output = torch.complex(real=output_real,imag=output_imag).squeeze(0).permute(2,1,0)
#sar_output = sar_model(sar_output)
abs_output = torch.abs(sar_output)
front_view = abs_output/torch.max(abs_output)
proj_db = 20*torch.log10(front_view)

mat = {'picture':front_view.detach().clone().cpu().numpy()}
savemat('./MMW-v1/Result/' + opt.data_name + '_reconstruction.mat',mat)
print('Reconstruction done!\n')
