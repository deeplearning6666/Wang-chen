import torch

def linear_interp(s,x,x_interp):
        indices = torch.searchsorted(x,x_interp)
        indices = torch.clamp(indices,1,len(x) -1)

        x_left = x[indices - 1]
        x_right = x[indices]
        y_left = s[indices - 1]
        y_right = s[indices]

        t = (x_interp - x_left) / (x_right - x_left)
        s_new = (1 - t) * y_left + t * y_right

        out_of_bounds = (x_interp < x[0]) | (x_interp > x[-1])
        s_new[out_of_bounds] = torch.tensor(0.0,dtype=s_new.dtype)

        return s_new 
