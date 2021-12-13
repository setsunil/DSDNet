import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as PF
import models.archs.arch_util as util



def fftbconv(x, k):
    assert len(x.shape)==4, 'the dim of x is not 4 but {}!'.format(len(x.shape))
    assert len(k.shape)==3, 'the dim of k is not 3 but {}!'.format(len(k.shape))
    x = reppad(x,k)
    _,c,n,m = x.shape
    _,k1,k2 = k.shape

    pad_x = (0, k2-1, 0, k1-1 )
    pad_k = (0, m-1, 0, n-1 )
    x_p  = F.pad(x, pad_x)
    k_p  = F.pad(k.rot90(k=2,dims=[-1,-2]), pad_k)
    f_x  = x_p.rfft(2, onesided=False)
    f_k  = k_p.rfft(2, onesided=False).unsqueeze(1).expand(-1,c,-1,-1,-1)
    f_I  = torch.irfft(cmul(f_x,f_k),2,onesided=False)
    (hk2d,hk2u,hk1d,hk1u) = padlrtb(k)

    return f_I[...,hk1d*2:-hk1u*2,hk2d*2:-hk2u*2]





def psum(x,nf, dim=0):
    xr, xg, xb = torch.split(x,nf,dim=dim)
    return torch.cat([xr.sum(dim=dim,keepdim=True),xg.sum(dim=dim,keepdim=True),xb.sum(dim=dim,keepdim=True)],dim=dim)


def XtX(x, complex=False):
    r = x.norm(dim=-1)**2
#    y = torch.cat((y.unsqueeze_(-1),torch.zeros(y.shape)),-1)
    return torch.stack([r, torch.zeros_like(r)], -1) if complex==True else r


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(x):
    y = x.clone()
    y[...,1]*=-1
    return y


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)

def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a/y, b/y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


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


def circonv2d(x,w):
    g=x.shape[1]//w.shape[1]
    x = cirpad(x,w)
    return F.conv2d(x,w,groups=g)

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


def padmask(m, k):
    p2d=padlrtb(k)
    return F.pad(m,p2d,mode='constant',value=1)


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


def crop(x, p):
    assert len(p)==4, 'padding size is not 4 but {}!'.format(len(p))
    return x[...,p[2]:-p[3],p[0]:-p[1]]


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


class ADDNN(nn.Module):
    def __init__(self, n_iter=3, in_nc=3, out_nc=3, nf_p=49, nf_d=48, max_nc=4, kernel_size = 7, drop_p=0.5, padding_mode='circular', upscale=1):
        super(ADDNN,self).__init__()
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

# class ADDNN(nn.Module):
#     def __init__(self, n_iter=3, in_nc=3, out_nc=3, nf_p=49, nf_d=48, max_nc=4, kernel_size = 7, drop_p=0.5, padding_mode='circular', upscale=1):
#         super(ADDNN,self).__init__()
#         self.in_nc  = in_nc
#         self.n      = n_iter
#         self.s      = upscale
#         self.hp     = nn.ModuleList([HyPaNet( in_nc*2, nf_p) for _ in range(n_iter)])
#         self.hd     = nn.ModuleList([HyPaNet( in_nc*2, nf_d) for _ in range(n_iter)])
#         self.hn     = nn.ModuleList([NLNet( in_nc) for _ in range(n_iter)])
#         self.p      = PriorNet( in_nc, nf_p, kernel_size, max_nc=max_nc, drop_p=drop_p, padding_mode=padding_mode)
#         self.d      = DataNet( in_nc, nf_d, kernel_size, max_nc=max_nc, drop_p=drop_p, padding_mode=padding_mode)
#         self.i      = CGUNet(self.p.conv, self.d.conv, in_nc, nf_p, nf_d, padding_mode=padding_mode)


#     def forward(self, y, k, m):
#         assert k.dim()==3, 'dimension of kernel is not 3 but {}!'.format(k.dim())
#         assert m.dim()==4, 'dimension of mask is not 4 but {}!'.format(m.dim())
#         x = model_init(y,k,self.s)

#         for i in range(self.n):
#             w = torch.cat((x,m),dim=1)
#             mn     = self.hn[i](x)
#             mp     = self.hp[i](w)
#             md     = self.hd[i](w)
#             v, fx  = self.p(x) if i == 0 else self.p(x, ui)
#             z, gr  = self.d(y, x, k, m) if i==0 else self.d(y, x, k, m, uj)
#             x      = self.i(x, y, v, z, k, mp, md, mn) if i==0 else self.i(x, y, v, z, k, mp, md, mn, ui, uj)

#             ui = fx - v if i == 0 else ui + fx -v
#             uj = gr - z if i == 0 else uj + gr -z
#         return x


#TEST
if __name__ == '__main__':
# MACS
    import arch_util as util
    from torchprofile import profile_macs
    #from thop import profile
    m = ADDNN(nf_p=49,nf_d = 49)
    input = torch.randn(1,3,256,256)
    kernel = torch.randn(1,31,31)
    mask = torch.ones_like(input)
    #macs = profile(m,inputs=(input,kernel,mask))
    macs = profile_macs(m,(input,kernel,mask))
    print(macs)

# #test calc_ker
#     conv = nn.Conv2d(3,3,7,padding=3,bias=False)
#     k=calc_ker(conv, 'torch.FloatTensor',3,3,13)
#     print(conv.weight)
#     print(k)
#     print(k[...,3:-3,3:-3])
#     print(torch.equal(k[...,3:-3,3:-3],conv.weight))
# testing p2o via fft.fft2
    # a = torch.tensor(
    # [[-1,1]])
    # b = p2o(a,torch.Size([5,5]))
    # print(b)

# #testing via image

    # import cv2
    # import scipy.io as scio
    # import torchvision.utils
    # import utils_deblur as udeblur


    # k = scio.loadmat('../Datasets/val_set5/kernel_LR.mat')['s']['kernel'][0][0]
    # y = cv2.imread('../Datasets/val_set5/Set5_LR/baby_k001_m001.png',cv2.IMREAD_UNCHANGED)
    # y = y.astype(np.float32)/255.0
    # H, W, C = y.shape
    # KH, KW = k.shape[-2:]
    # H = H + KH -1
    # W = W + KW -1
    # #y = udeblur.wrap_boundary_liu(y,[H,W,C]).astype(np.float32)
    # y = udeblur.edgetaper(udeblur.pad_for_kernel(y,k,'edge'),k).astype(np.float32)
    # y = torch.from_numpy(y[:,:,[2,1,0]]).permute(2,0,1).unsqueeze(0)

    # k = torch.from_numpy(k).type(torch.float32).unsqueeze(0)
    # m = torch.ones(y.shape[-2:]).unsqueeze(0).unsqueeze(0)

    # torchvision.utils.save_image(y, 'y.png', nrow=1, padding=0,
    #                               normalize=False)



    # n = ADDNN(n_iter=1, in_nc = 3, out_nc = 3, padding_mode='circular')

    # x = n(y,k,m)

#testing fftbconv
    # x=torch.randn(3,1,6,5)
    # m = torch.randn(3,5,5)
    # o=bconv(x,m,'replicate')
    # of = fftbconv(x,m)
    # print((of-o)>1e-5)


#testing bconv
    # x=torch.ones(6,3,6,6)
    # for i in range(6):
    #     x[i,...]*=i
    # for i in range(3):
    #     x[:,i,...]+=i*0.1
    # m = torch.randn(6,3,3)
    # o=bconv(x,m)
    # print(o)
#testing cirpad & ConvTran2dCir
    # x = torch.randn(5,5,10,11)
    # C = nn.Conv2d(5,10,(3,5),padding=(1,2),groups=5)
    # T = ConvTran2dCir(C)
    # O = T(C(x))
    # print(O.shape)
#testing CGD
    # import scipy.io as scio
    # import torchvision.utils

    # def measure(x,k,m):


    #     x  = bconv(x,k)
    #     x  = x.mul(m)

    #     return x


    # M = scio.loadmat('demo_inp.mat')
    # img = torch.from_numpy(M['I']).type(torch.float32).unsqueeze(0).unsqueeze(0)
    # filt = torch.from_numpy(M['filt']).type(torch.float32).unsqueeze(0)


    # Cp= nn.Conv2d(1,2,3,padding=0,padding_mode='zeros',bias=False)
    # Cp.weight.data=torch.tensor(
    #          [[[[0, 0,  0],
    #             [0, 1, -1],
    #             [0, 0,  0]]],

    #           [[[0, 1,  0],
    #             [0, -1,  0],
    #             [0, 0,  0]]]],dtype=torch.float32)
    # Cd = nn.Conv2d(1,1,3,padding=0,padding_mode='zeros',bias=False)
    # Cd.weight.data=torch.tensor(
    #          [[[[0, 0,  0],
    #             [0, 1,  0],
    #             [0, 0,  0]]]]
    #           ,dtype=torch.float32)
    # x = img
    # I = CGDInverseNet(Cp,Cd, 1)
    # m = torch.ones(img.shape[-2:]).unsqueeze(0)
    # # v = torch.zeros_like(x).expand(1,Cp.out_channels,-1,-1)
    # # z = torch.zeros_like(x).expand(1,Cd.out_channels,-1,-1)
    # v = torch.zeros(1,2,256,367)
    # z = torch.zeros(1,1,256,367)
    # y = measure(x,filt,m)
    # x = I(x, x, v, z, filt, m)
    # torchvision.utils.save_image(x , 'x.png', nrow=1, padding=0,
    #                               normalize=False)