import torch
import torch.nn.functional as F

def ftx(s,n=0):
    if n == 0:
            n = s.size(1)
    
            n0 = s.size(1)

            Npad = n - n0
    if Npad >= 0:
        if n0 % 2 == 0:
                Npad1 = Npad // 2 
                Npad2 = (Npad + 1) // 2
        else:
                Npad1 = (Npad + 1) // 2
                Npad2 = Npad // 2
        
    pad_list = []

    for i in range(s.dim()):
        pad_list.extend([0,0])

    pad_list[0] = Npad1
    pad_list[1] = Npad2

    s = F.pad(s,pad_list,mode='constant',value=0)

    fs = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(s,dim=0),dim=0),dim=0)

    return fs

def fty(s,n=0):
    if n == 0:
        n = s.size(1)
    
    n0 = s.size(1)

    Npad = n - n0
    if Npad >= 0:
        if n0 % 2 == 0:
            Npad1 = Npad // 2 
            Npad2 = (Npad + 1) // 2
        else:
            Npad1 = (Npad + 1) // 2
            Npad2 = Npad2 // 2
        
    pad_list = []

    for i in range(s.dim()):
        pad_list.extend([0,0])

    pad_list[2] = Npad1
    pad_list[3] = Npad2

    s = F.pad(s,pad_list,mode='constant',value=0)

    fs = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(s,dim=1),dim=1),dim=1)

    return fs

def ftz(s,n=0):
    if n == 0:
            n = s.size(1)
    
            n0 = s.size(1)

            Npad = n - n0
    if Npad >= 0:
        if n0 % 2 == 0:
                Npad1 = Npad // 2 
                Npad2 = (Npad + 1) // 2
        else:
                Npad1 = (Npad + 1) // 2
                Npad2 = Npad2 // 2
        
    pad_list = []

    for i in range(s.dim()):
        pad_list.extend([0,0])

    pad_list[4] = Npad1
    pad_list[5] = Npad2

    s = F.pad(s,pad_list,mode='constant',value=0)

    fs = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(s,dim=2),dim=2),dim=2)

    return fs

def iftx(fs,n=0):
    if n == 0:
            n = fs.size(1)
    
            n0 = fs.size(1)

            Npad = n - n0
    if Npad >= 0:
        if n0 % 2 == 0:
                Npad1 = Npad // 2 
                Npad2 = (Npad + 1) // 2
        else:
                Npad1 = (Npad + 1) // 2
                Npad2 = Npad2 // 2
        
    pad_list = []

    for i in range(fs.dim()):
        pad_list.extend([0,0])

    pad_list[0] = Npad1
    pad_list[1] = Npad2

    fs = F.pad(fs,pad_list,mode='constant',value=0)

    s = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fs,dim=0),dim=0),dim=0)
    return s

def ifty(fs,n=0):
    if n == 0:
            n = fs.size(1)
    
            n0 = fs.size(1)

            Npad = n - n0
    if Npad >= 0:
        if n0 % 2 == 0:
                Npad1 = Npad // 2 
                Npad2 = (Npad + 1) // 2
        else:
                Npad1 = (Npad + 1) // 2
                Npad2 = Npad2 // 2
        
    pad_list = []

    for i in range(fs.dim()):
        pad_list.extend([0,0])

    pad_list[2] = Npad1
    pad_list[3] = Npad2

    fs = F.pad(fs,pad_list,mode='constant',value=0)

    s = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fs,dim=1),dim=1),dim=1)
    return s

def iftz(fs,n=0):
    if n == 0:
            n = fs.size(1)
    
            n0 = fs.size(1)

            Npad = n - n0
    if Npad >= 0:
        if n0 % 2 == 0:
                Npad1 = Npad // 2 
                Npad2 = (Npad + 1) // 2
        else:
                Npad1 = (Npad + 1) // 2
                Npad2 = Npad2 // 2
        
    pad_list = []

    for i in range(fs.dim()):
        pad_list.extend([0,0])

    pad_list[4] = Npad1
    pad_list[5] = Npad2

    fs = F.pad(fs,pad_list,mode='constant',value=0)

    s = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(fs,dim=2),dim=2),dim=2)
    return s
