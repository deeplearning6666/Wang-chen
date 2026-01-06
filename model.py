import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from FFT import ftx,fty,iftx,ifty,iftz
from interp import linear_interp

class sar_model(torch.nn.Module):
    def __init__(self,
                 f_start = 348e9,
                 f_stop = 372e9,
                 Dx = 0.4,
                 Dy = 0.4,
                 Nx = 200,
                 Ny = 200,
                 z0 = 2,
                 z_range = 0.5,
                 dx = 0.002,
                 dy = 0.002,
                 Nz = 2048,
                 dev = torch.device('cuda:0'),
                 ):
        super(sar_model,self).__init__()
        c = 3e8
        self.c = c
        fc = (f_start + f_stop)/2
        B = abs(f_stop - f_start)
        self.Dx = Dx
        self.Dy = Dy
        self.dx = dx
        self.dy = dy
        self.Nz = Nz
        self.z_range = z_range
        self.f = fc + torch.linspace(-B/2,B/2,Nz)
        self.k = self.f * 2 * torch.pi/c
        self.dev = dev
        self.z0 = z0
        Nx = int(Dx/dx)
        Ny = int(Dy/dy)
        self.Nx = Nx
        self.Ny = Ny
        k = self.k
        f = self.f
        x = torch.linspace(Dx/2,Dx/2,Nx)
        y = torch.linspace(Dy/2,Dy/2,Ny)
        kkx = 2 * torch.pi /dx
        kky = 2 * torch.pi /dy
        kx = torch.linspace(-kkx/2,kkx/2,Nx)[:Nx]
        ky = torch.linspace(-kky/2,kky/2,Ny)[:Ny]

        KX,KY = torch.meshgrid(kx,ky,indexing='ij')
        G = torch.zeros(Ny,Nx,len(f))

        for i in range(len(f)):
            G[:,:,i] = 4 * k[i]**2 - KX**2 - KY**2
        
        idx = G<0
        G[idx] = 1
        kz = torch.sqrt(G)

        kz_min = kz.min()
        kz_max = kz.max()
        kz_interp = torch.linspace(kz_min,kz_max,Nz)
        range_z = 2 * torch.pi / (kz_interp[1] - kz_interp[0])
        z = torch.linspace(-range_z/2,range_z/2,Nz)
        z_1 = torch.linspace(0,range_z,Nz)#按照距离计算距离向
        dz = z[1] - z[0]

        start_num = int(((z0 - z_range/2 - z[0])/dz).item())
        end_num = int(((z0 + z_range/2 - z[0])/dz).item())

        R_0 = z_1[0]/2
        R_1 = torch.sqrt(z_1[-1]**2 + x[-1]**2 + y[-1]**2)/2
        R = torch.linspace(R_0,R_1,Nz)

        zz1 = z_1[start_num]/2
        zz2 = torch.sqrt(z_1[end_num]**2 + x[-1]**2 + y[-1]**2)/2

        cha1 = R - zz1
        cha2 = zz2 - R
        idx1 = torch.where(cha1 > 0)[0]
        idx2 = torch.where(cha2 < 0)[0]

        R_start = idx1[0] if len(idx1) > 0 else 0
        R_end = idx2[0] if len(idx2) > 0 else len(R)
        self.R_start = R_start
        self.R_end = R_end
        self.kz_seg = kz[:,:,R_start:R_end]
        self.kz_interp_seg = kz_interp[R_start:R_end]

    def forward(self,fs):
        compen1 = torch.exp(-1j*2*torch.pi*self.f*self.z0*2/self.c)
        compen1 = compen1.unsqueeze(0).squeeze(0)
        compen = compen1.repeat(200,200,1).to(device='cuda:0')
        compen = compen[:,:,self.R_start:self.R_end]
        fs = fs * compen
        kz_seg = self.kz_seg
        kz_interp_seg = self.kz_interp_seg.to(device='cuda:0')
        S = fty(ftx(fs))
        S = S.permute(2,1,0)
        kz_seg = kz_seg.permute(2,1,0)
        SS = torch.zeros(self.R_end - self.R_start,self.Nx,self.Ny,dtype=S.dtype,device=self.dev)
        for p in tqdm(range(self.Ny),desc='插值进行中'):
            for q in range(self.Nx):
                kz_current = kz_seg[:,q,p].to(device='cuda:0')
                S_current = S[:,q,p].to(device='cuda:0')

                SS[:,q,p] = linear_interp(S_current,kz_current,kz_interp_seg)
        print("插值完成")
        SS = SS.permute(2,1,0)
        SS = iftz(iftx(ifty(SS)))
   
        return SS
