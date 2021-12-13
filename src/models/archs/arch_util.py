import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.fft as PF
import numpy as np
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

def padlrtb(k):
    pvt=(k.shape[-2]-1)//2
    phl=(k.shape[-1]-1)//2
    if k.shape[-1]%2==0 :
        phr = phl+1
    else:
        phr = phl
    if k.shape[-2]%2==0 :
        pvb = pvt+1
    else:
        pvb = pvt
    return (phl,phr,pvt,pvb)

def cirpad(x,k):
    p2d=padlrtb(k)
    x=F.pad(x,p2d,mode='circular')
    return x

def reppad(x,k):
    p2d=padlrtb(k)
    x=F.pad(x,p2d,mode='replicate')
    return x


def fft_conv(img, kernel):
    p = padlrtb(kernel)
    X = PF.fft2(img)
    K = p2o(kernel, img.shape)
    return PF.ifft2(X*K).real[...,p[0]:-p[1],p[2]:-p[3]]

# from kruse et al. 17
def pad_for_kernel(img, kernel, mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)

# from kruse et al. 17
def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1-i), img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z/np.max(z))
    return np.outer(*v)

def edgetaper_alpha_torch(kernel, img_shape):
    v = []
    for i in range(1,3):
        z = PF.fft(torch.sum(kernel,-i),img_shape[i+1]-1)
        z = PF.ifft(torch.square(torch.abs(z))).real
        z = torch.cat([z, z[:,0:1]], 1)
        m = torch.max(z.view(z.shape[0],-1),dim=1).values.view(-1,1)
        v.append(1 - z/m)
    return torch.einsum('bi,bj->bij',*v)

# from kruse et al. 17
def edgetaper(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha  = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel,'wrap'), kernel, mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img


def edgetaper_torch(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha_torch(kernel, img.shape)
    if kernel.dim()==3:
        kernel = kernel.unsqueeze(1)
        alpha  = alpha.unsqueeze(1)
    for _ in range(n_tapers):
        blurred = fft_conv(cirpad(img, kernel), kernel)
        img = alpha*img + (1.-alpha)*blurred
    return img

def p2o(psf, shape=None):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    if shape is None:
        shape = psf.shape
    otf = torch.zeros(psf.shape[:-2]+shape[-2:]).type_as(psf)
    otf[...,:psf.shape[-2],:psf.shape[-1]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[-2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=-2+axis)
    #otf = torch.rfft(otf, 2, onesided=False)
    otf = PF.fft2(otf,dim=(-2,-1))
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf.imag[torch.abs(otf.imag) < n_ops*2.22e-16] = 0
    return otf

if __name__ == '__main__':
    from scipy.signal import fftconvolve

    i = torch.randn(5,3,5,5)
    k = torch.randn(5,3,3)
    v =[]
    for ind in range(i.shape[0]):
        ni = i[ind,...].permute(1,2,0).numpy()
        nk = k[ind,...].numpy()
        npo = edgetaper(pad_for_kernel(ni,nk,'edge'),nk)
        v.append(npo[np.newaxis,...])
    v = np.concatenate(v,axis=0)
    o = edgetaper_torch(reppad(i,k),k)
    npo = torch.from_numpy(v).permute(0,3,1,2)
    print(torch.nonzero(torch.abs(o-npo)>1e-4))


