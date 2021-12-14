import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as PF
import models.archs.arch_util as util




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


def zeropad(x,k):
    p2d=padlrtb(k)
    x=F.pad(x,p2d,mode='constant')
    return x



def bconv(x,k, padding_mode='circular'):# N,C,H,W -> conv(C,N,H,W) -> N,C,H,W
    b = x.shape[0]
    x = x.permute(1,0,2,3)
    if padding_mode == 'circular':
        x = cirpad(x,k)
    elif padding_mode=='replicate':
        x = reppad(x,k)
    else:
        x = zeropad(x,k)
    return F.conv2d(x,k.unsqueeze(1),groups=b).permute(1,0,2,3)


def fft_deconv(y, k):
    p = padlrtb(k)
    k = torch.rot90(k,2,(-1,-2))
    y = util.edgetaper_torch(reppad(y,k),k)
    Gx = p2o(torch.tensor([[-1,1]]).type_as(y),y.shape)
    Gy = p2o(torch.tensor([[-1],[1]]).type_as(y),y.shape)
    F = p2o(k , y.shape).unsqueeze(1)
    A = torch.abs(F).pow(2) + 0.002*(torch.abs(Gx).pow(2)+torch.abs(Gy).pow(2))
    b = F.conj()*PF.fft2(y)
    return PF.ifft2(b/A).real[...,p[0]:-p[1],p[2]:-p[3]]

def model_init(y,k,s):
    if s == 1:
        return fft_deconv(y,k)
    else :
        pass


class ConvTran2dCir(nn.Module):
    def __init__(self, conv):
        super(ConvTran2dCir, self).__init__()
        self.out_channels = conv.in_channels
        self.in_channels  = conv.out_channels
        self.kernel_size  = conv.kernel_size[::-1]
        self.stride       = conv.stride
        self.bias         = True if conv.bias is not None else False
        pad               = conv.padding#tuple([2*x for x in conv.padding]) if conv.padding_mode =='circular' else 0
        self.ConvT        = nn.ConvTranspose2d(self.in_channels, self.out_channels,
                                            self.kernel_size, self.stride, padding=pad, groups=conv.groups, bias=self.bias)
        self.padding_mode = conv.padding_mode
        self.ConvT.weight = conv.weight
    def forward(self, x):
        return self.ConvT(x)


class Maxout(nn.Module):
    def __init__(self, in_nc = 49, num_max = 4, drop_p=0.5, kernel_size = 3):
        super(Maxout, self).__init__()
        maxout_nc = in_nc*num_max
        pad = kernel_size //2
        self.conv1 = nn.Conv2d(in_nc, maxout_nc, kernel_size=kernel_size, padding=pad, groups=in_nc)
        self.conv2 = nn.Conv2d(in_nc, maxout_nc, kernel_size=kernel_size, padding=pad, groups=in_nc)
        self.ml  = nn.MaxPool3d((num_max,1,1))
        self.dropout = nn.Dropout2d(p=drop_p, inplace=True)

    def forward(self, x, u=None):
        x = self.dropout(x)
        return self.ml(self.conv1(x))-self.ml(self.conv2(x)) if u is None else self.ml(self.conv1(x+u))-self.ml(self.conv2(x+u))



class NLNet(nn.Module):
    def __init__(self, in_nc=3, channel=64):
        super(NLNet, self).__init__()
        self.p = nn.AvgPool2d(3,padding=1,stride=1,count_include_pad=False)
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc*2, channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, in_nc, 3, padding=1, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(torch.cat((x,self.p(x)),dim=1))
        return x


class HyPaNet(nn.Module):
    def __init__(self, in_nc=6, out_nc=147):
        super(HyPaNet, self).__init__()
        channel = out_nc//4
        self.head =nn.Sequential(
                nn.Conv2d(in_nc, out_nc, 3, padding=1, bias=True),
                nn.ReLU(inplace=True))
        self.res = nn.Sequential(
                nn.Conv2d(out_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True)
                )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.head(x)
        x = x + self.res(x)
        return self.sig(x)

class CGUNet(nn.Module):
    def __init__(self, conv_p, conv_d, in_nc, nf_p, nf_d, padding_mode = 'circular'):
        super(CGUNet, self).__init__()
        nf_t = nf_p+nf_d
        self.in_nc        = in_nc
        self.nf_p         = nf_p
        self.nf_d         = nf_d
        self.conv_prior   = conv_p
        self.conv_data    = conv_d

        self.d1            = down(nf_t,nf_t*2)
        self.d2            = down(nf_t*2, nf_t*4)
        self.d3            = down(nf_t*4, nf_t*8)
        self.bot            = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(nf_t*8,nf_t*16,3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(nf_t*16,nf_t*16,3,padding=1),
            nn.ReLU(True)
            )
        self.u3            = up(nf_t*16, nf_t*8)
        self.u2            = up(nf_t*8, nf_t*4)
        self.u1            = up(nf_t*4, nf_t*2)
        self.u0           =  up(nf_t*2, nf_t)


        self.padding_mode = padding_mode


    def forward(self, x, y, v, z, k, mp, md, mn, ui=None, uj=None):

        # f  = calc_ker(self.conv_prior, x.type(), self.nf_p, self.in_nc, 19)
        # g  = calc_ker(self.conv_data, x.type(), self.nf_d, self.in_nc, 19)
        f = self.conv_prior.weight
        g = self.conv_data.weight
        kt = torch.flip(k,[1,2])
        Gy = self.conv_data(y)
        Gy = Gy - z if uj is None else Gy - z + uj
        v  = v      if ui is None else v - ui
        b  = self.Ad(v,Gy,kt, f, g, mp, md, mn)
        b  = b[0] + b[1]
        #CGM
        r  = b - self.A(x, k, kt, f, g, mp, md, mn)
        r1  = torch.cat(self.Ae(r,k, mp, md),dim=1)
        r2 = self.d1(r1)
        r3 = self.d2(r2)
        r4 = self.d3(r3)
        r5 = self.bot(r4)
        r6 = self.u3(r5,r4)
        r7 = self.u2(r6,r3)
        r8 = self.u1(r7,r2)
        r  = self.u0(r8,r1)
        r  = self.Ad(r[:,:self.nf_p,...], r[:,self.nf_p:,...],kt, f, g, mp, md, mn)
        r  = r[0] + r[1]
        x  = x + r
        return x

    def A(self, x, k, kt, f, g, mp, md, mn):
        x_d = bconv(x, k, self.padding_mode)
        x_d = self.conv_data(x_d).mul(torch.square(md))
        x_d = F.conv_transpose2d(x_d, g, padding=g.size(-1)//2)
        x_d = bconv(x_d, kt, 'zeros')
        x_p = self.conv_prior(x).mul(torch.square(mp))
        x_p = F.conv_transpose2d(x_p, f, padding=f.size(-1)//2)

        return mn*x_p + x_d

    def Ae(self, x, k, mp, md):
        x_d = bconv(x, k, self.padding_mode)
        x_d = self.conv_data(x_d)
        x_p = self.conv_prior(x)

        x_p = x_p*mp
        x_d = x_d*md

        return x_p, x_d

    def Ad(self, x_p, x_d, kt, f, g, mp, md, mn):
        x_d = x_d*md
        x_p = x_p*mp
        x_d = F.conv_transpose2d(x_d, g, padding=g.size(-1)//2)
        x_d = bconv(x_d, kt, 'zeros')
        x_p = F.conv_transpose2d(x_p, f, padding=f.size(-1)//2)

        return mn*x_p, x_d

def calc_ker(conv, type, nf, in_nc, rf):
    delta = torch.zeros(1, in_nc, 1, 1).type(type)
    delta[:,-1,...] = 1.
    pad = [rf//2]*4
    delta = F.pad(delta,pad)
    K = torch.zeros(nf, in_nc, rf, rf).type(type)
    for i in range(in_nc):
        delta = torch.roll(delta, 1, 1)
        KC = conv(delta).rot90(2,(-1,-2))
        K[:,i,...] =KC
    return K


class down(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(down, self).__init__()
        self.d      =   nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(in_nc, out_nc, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_nc, out_nc, 3, padding=1),
            nn.ReLU(True)
            )

    def forward(self, x):
        return self.d(x)



class up(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(up, self).__init__()
        self.u     =   nn.ConvTranspose2d(in_nc, out_nc, 3, stride =2, padding=1)
        self.m     =   nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_nc, out_nc, 3, padding=1),
            nn.ReLU(True)
            )


    def forward(self, x, s):
        x = self.u(x, output_size=s.shape)
        x = torch.cat((x,s),dim=1)
        return self.m(x)


class PriorNet(nn.Module):
    def __init__(self, in_nc, nf=49, kernel_size=7, max_nc = 4, drop_p=0.5, padding_mode='circular'):
        super(PriorNet, self).__init__()
        pad = kernel_size//2 #if padding_mode == 'circular' else 0
        self.conv = nn.Conv2d(in_nc, nf, kernel_size, padding=pad, bias=False)
        self.m = Maxout(nf, max_nc, drop_p)

    def forward(self, x, ui=None):
        fx = self.conv(x)
        v  = self.m(fx, ui)
        return v, fx


class DataNet(nn.Module):
    def __init__(self, in_nc, nf=48, kernel_size=7, max_nc = 4, drop_p=0.5, padding_mode='circular'):
        super(DataNet, self).__init__()
        pad = kernel_size//2 #if padding_mode =='circular' else 0
        self.conv = nn.Conv2d(in_nc, nf, kernel_size, padding=pad, bias = False)
        self.m = Maxout(nf, max_nc, drop_p)
        self.padding_mode = padding_mode

    def forward(self, y, x, k, m, uj=None ):
        kx  = bconv(x, k, self.padding_mode )
        r   = y - kx.mul(m)
        gr  = self.conv(r)
        z   = self.m(gr, uj)
        return z, gr


class DSDNet(nn.Module):
    def __init__(self, n_iter=3, in_nc=3, out_nc=3, nf_p=49, nf_d=48, max_nc=4, kernel_size = 7, drop_p=0.5, padding_mode='circular', upscale=1):
        super(DSDNet,self).__init__()
        self.in_nc  = in_nc
        self.n      = n_iter
        self.s      = upscale
        self.hp     = nn.ModuleList([HyPaNet( in_nc*2, nf_p) for _ in range(n_iter)])
        self.hd     = nn.ModuleList([HyPaNet( in_nc*2, nf_d) for _ in range(n_iter)])
        self.hn     = nn.ModuleList([NLNet( in_nc) for _ in range(n_iter)])
        self.p      = nn.ModuleList([PriorNet( in_nc, nf_p, kernel_size, max_nc=max_nc, drop_p=drop_p, padding_mode=padding_mode) for _ in range(n_iter)])
        self.d      = nn.ModuleList([DataNet( in_nc, nf_d, kernel_size, max_nc=max_nc, drop_p=drop_p, padding_mode=padding_mode) for _ in range(n_iter)])
        self.i      = nn.ModuleList([CGUNet(self.p[i].conv, self.d[i].conv, in_nc, nf_p, nf_d, padding_mode=padding_mode) for i in range(n_iter)])

    def forward(self, y, k, m):
        assert k.dim()==3, 'dimension of kernel is not 3 but {}!'.format(k.dim())
        assert m.dim()==4, 'dimension of mask is not 4 but {}!'.format(m.dim())
        x = model_init(y,k,self.s)

        for i in range(self.n):
            w = torch.cat((x,m),dim=1)
            mn     = self.hn[i](x)
            mp     = self.hp[i](w)
            md     = self.hd[i](w)
            v, fx  = self.p[i](x) if i == 0 else self.p[i](x, ui)
            z, gr  = self.d[i](y, x, k, m) if i==0 else self.d[i](y, x, k, m, uj)
            x      = self.i[i](x, y, v, z, k, mp, md, mn) if i==0 else self.i[i](x, y, v, z, k, mp, md, mn, ui, uj)

            ui = fx - v if i == 0 else ui + fx -v
            uj = gr - z if i == 0 else uj + gr -z


        return x

