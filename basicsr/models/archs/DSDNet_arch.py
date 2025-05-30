from basicsr.models.archs.arch_util import LayerNorm2d
from einops import rearrange
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as PF


def fftbconv(x, k):
    assert len(x.shape) == 4, 'the dim of x is not 4 but {}!'.format(len(x.shape))
    assert len(k.shape) == 3, 'the dim of k is not 3 but {}!'.format(len(k.shape))
    x = reppad(x, k)
    _, c, n, m = x.shape
    _, k1, k2 = k.shape

    pad_x = (0, k2-1, 0, k1 - 1)
    pad_k = (0, m-1, 0, n - 1)
    x_p  = F.pad(x, pad_x)
    k_p  = F.pad(k.rot90(k=2, dims=[-1, -2]), pad_k)
    f_x  = x_p.rfft(2, onesided=False)
    f_k  = k_p.rfft(2, onesided=False).unsqueeze(1).expand(-1, c, -1, -1, -1)
    f_I  = torch.irfft(cmul(f_x, f_k), 2, onesided=False)
    (hk2d, hk2u, hk1d, hk1u) = padlrtb(k)

    return f_I[..., hk1d*2:-hk1u*2, hk2d*2:-hk2u*2]


def psum(x, nf, dim=0):
    xr, xg, xb = torch.split(x, nf, dim=dim)
    return torch.cat([xr.sum(dim=dim, keepdim=True), xg.sum(dim=dim, keepdim=True), xb.sum(dim=dim, keepdim=True)], dim=dim)


def XtX(x, complex=False):
    r = x.norm(dim=-1)**2
#    y = torch.cat((y.unsqueeze_(-1),torch.zeros(y.shape)),-1)
    return torch.stack([r, torch.zeros_like(r)], -1) if complex is True else r


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
    y[..., 1] *= -1
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
    otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[-2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=-2+axis)
    #otf = torch.rfft(otf, 2, onesided=False)
    otf = PF.fft2(otf, dim=(-2, -1))
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf.imag[torch.abs(otf.imag) < n_ops*2.22e-16] = 0
    return otf


def circonv2d(x, w):
    g = x.shape[1]//w.shape[1]
    x = cirpad(x, w)
    return F.conv2d(x, w, groups=g)


def padlrtb(k):
    pvt = (k.shape[-2] - 1) // 2
    phl = (k.shape[-1] - 1) // 2
    if k.shape[-1] % 2 == 0:
        phr = phl + 1
    else:
        phr = phl
    if k.shape[-2] % 2 == 0:
        pvb = pvt + 1
    else:
        pvb = pvt
    return (phl, phr, pvt, pvb)


def padmask(m, k):
    p2d = padlrtb(k)
    return F.pad(m, p2d, mode='constant', value=1)


def cirpad(x, k):
    p2d = padlrtb(k)
    x = F.pad(x, p2d, mode='circular')
    return x


def reppad(x, k):
    p2d = padlrtb(k)
    x = F.pad(x, p2d, mode='replicate')
    return x


def refpad(x, k):
    p2d = padlrtb(k)
    x = F.pad(x, p2d, mode='reflect')
    return x


def zeropad(x, k):
    p2d = padlrtb(k)
    x = F.pad(x, p2d, mode='constant')
    return x


def crop(x, p):
    assert len(p) == 4, 'padding size is not 4 but {}!'.format(len(p))
    return x[..., p[2]:-p[3], p[0]:-p[1]]


def bconv(x, k, padding_mode='circular'):  # N,C,H,W -> conv(C,N,H,W) -> N,C,H,W
    b = x.shape[0]
    x = x.permute(1, 0, 2, 3)
    if padding_mode == 'circular':
        x = cirpad(x, k)
    elif padding_mode == 'replicate':
        x = reppad(x, k)
    elif padding_mode == 'reflect':
        x = refpad(x, k)
    else:
        x = zeropad(x, k)
    return F.conv2d(x, k.unsqueeze(1), groups=b).permute(1, 0, 2, 3)


def fft_conv(img, kernel):
    p = padlrtb(kernel)
    X = PF.fft2(img)
    K = p2o(kernel, img.shape)
    return PF.ifft2(X * K).real[..., p[0]:-p[1], p[2]:-p[3]]


def edgetaper_alpha_torch(kernel, img_shape):
    v = []
    for i in range(1, 3):
        z = PF.fft(torch.sum(kernel, -i), img_shape[i+1]-1)
        z = PF.ifft(torch.square(torch.abs(z))).real
        z = torch.cat([z, z[:, 0:1]], 1)
        v.append(1 - z / torch.amax(z, dim=1, keepdim=True))
    return torch.einsum('bi,bj->bij', *v)


def edgetaper_torch(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha_torch(kernel, img.shape)
    if kernel.dim() == 3:
        kernel = kernel.unsqueeze(1)
        alpha  = alpha.unsqueeze(1)
    for _ in range(n_tapers):
        blurred = fft_conv(cirpad(img, kernel), kernel)
        img = alpha * img + (1. - alpha) * blurred
    return img


def fft_deconv(y, k):
    p = padlrtb(k)
    k = torch.rot90(k, 2, (-1, -2))
    y = edgetaper_torch(reppad(y, k), k)
    Gx = p2o(torch.tensor([[-1, 1]]).type_as(y), y.shape)
    Gy = p2o(torch.tensor([[-1], [1]]).type_as(y), y.shape)
    F = p2o(k, y.shape).unsqueeze(1)
    A = torch.abs(F).pow(2) + 0.002*(torch.abs(Gx).pow(2)+torch.abs(Gy).pow(2))
    b = F.conj()*PF.fft2(y)
    return PF.ifft2(b/A).real[..., p[0]:-p[1], p[2]:-p[3]]


class ConvTran2dCir(nn.Module):
    def __init__(self, conv):
        super(ConvTran2dCir, self).__init__()
        self.out_channels = conv.in_channels
        self.in_channels  = conv.out_channels
        self.kernel_size  = conv.kernel_size[::-1]
        self.stride       = conv.stride
        self.bias         = True if conv.bias is not None else False
        pad               = conv.padding  # tuple([2*x for x in conv.padding]) if conv.padding_mode =='circular' else 0
        self.ConvT        = nn.ConvTranspose2d(self.in_channels, self.out_channels,
                                               self.kernel_size, self.stride, padding=pad, groups=conv.groups, bias=self.bias)
        self.padding_mode = conv.padding_mode
        self.ConvT.weight = conv.weight

    def forward(self, x):
        return self.ConvT(x)


class Maxout(nn.Module):
    def __init__(self, in_nc=49, num_max=4, drop_p=0.5, kernel_size=3):
        super(Maxout, self).__init__()
        maxout_nc = in_nc * num_max
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_nc, maxout_nc, kernel_size=kernel_size, padding=pad, groups=in_nc)
        self.conv2 = nn.Conv2d(in_nc, maxout_nc, kernel_size=kernel_size, padding=pad, groups=in_nc)
        self.ml  = nn.MaxPool3d((num_max, 1, 1))
        self.dropout = nn.Dropout2d(p=drop_p, inplace=True)

    def forward(self, x, u=None):
        x = self.dropout(x)
        return self.ml(self.conv1(x))-self.ml(self.conv2(x)) if u is None else self.ml(self.conv1(x+u))-self.ml(self.conv2(x+u))


class NLNet(nn.Module):
    def __init__(self, in_nc=3, channel=64):
        super(NLNet, self).__init__()
        self.p = nn.AvgPool2d(3, padding=1, stride=1, count_include_pad=False)
        self.mlp = nn.Sequential(nn.Conv2d(in_nc * 2, channel, 3, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, in_nc, 3, padding=1, bias=True),
                                 nn.Softplus())

    def forward(self, x):
        x = self.mlp(torch.cat((x, self.p(x)), dim=1))
        return x


class NLNet_v2(nn.Module):
    def __init__(self, in_nc=3, channel=64):
        super(NLNet_v2, self).__init__()

        channel = in_nc * 2
        self.p = nn.AvgPool2d(3, padding=1, stride=1, count_include_pad=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, bias=True),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True, groups=channel),
            nn.Conv2d(channel, in_nc, 1, bias=True),
        )

    def forward(self, x):
        x = self.p(x)
        x = x + self.mlp(x)
        return x


class HypNet(nn.Module):
    def __init__(self, in_nc=6, out_nc=147):
        super(HypNet, self).__init__()
        channel = out_nc//4
        self.head = nn.Sequential(nn.Conv2d(in_nc, out_nc, 3, padding=1, bias=True),
                                  nn.ReLU(inplace=True))
        self.res = nn.Sequential(nn.Conv2d(out_nc, channel, 1, padding=0, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, out_nc, 1, padding=0, bias=True))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.head(x)
        x = x + self.res(x)
        return self.sig(x)



class CGUNet(nn.Module):
    def __init__(self, conv_p, conv_d, in_nc, nf_p, nf_d, padding_mode='circular'):
        super(CGUNet, self).__init__()
        nf_t = nf_p+nf_d
        self.in_nc        = in_nc
        self.nf_p         = nf_p
        self.nf_d         = nf_d
        self.conv_prior   = conv_p
        self.conv_data    = conv_d

        self.d1            = down(nf_t, nf_t*2)
        self.d2            = down(nf_t*2, nf_t*4)
        self.d3            = down(nf_t*4, nf_t*8)
        self.bot           = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Conv2d(nf_t*8,nf_t*16,3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(nf_t*16,nf_t*16,3,padding=1),
            nn.ReLU(True)
        )
        self.u3            = up(nf_t*16, nf_t*8)
        self.u2            = up(nf_t*8, nf_t*4)
        self.u1            = up(nf_t*4, nf_t*2)
        self.u0            = up(nf_t*2, nf_t)

        self.padding_mode = padding_mode


    def forward(self, x, y, v, z, k, mp, md, mn, ui=None, uj=None):

        f = self.conv_prior.weight
        g = self.conv_data.weight
        kt = torch.flip(k, [1, 2])
        Gy = self.conv_data(y)
        Gy = Gy - z if uj is None else Gy - z + uj
        v  = v if ui is None else v - ui
        b  = self.Ad(v, Gy, kt, f, g, mp, md, mn)
        b  = b[0] + b[1]
        #CGM
        r  = b - self.A(x, k, kt, f, g, mp, md, mn)
        r1  = torch.cat(self.Ae(r, k, mp, md), dim=1)
        r2 = self.d1(r1)
        r3 = self.d2(r2)
        r4 = self.d3(r3)
        r5 = self.bot(r4)
        r6 = self.u3(r5, r4)
        r7 = self.u2(r6, r3)
        r8 = self.u1(r7, r2)
        r  = self.u0(r8, r1)
        r  = self.Ad(r[:, :self.nf_p, ...], r[:, self.nf_p:, ...], kt, f, g, mp, md, mn)
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


class KryNet(nn.Module):
    def __init__(self, in_nc, nf_p, nf_d, padding_mode='circular'):
        super(KryNet, self).__init__()
        nf_t = nf_p + nf_d
        self.in_nc = in_nc
        self.nf_p  = nf_p
        self.nf_d  = nf_d

        self.d1     = down_v2(nf_t, nf_t*2)
        self.d2     = down_v2(nf_t*2, nf_t*4)
        self.d3     = down_v2(nf_t*4, nf_t*8)
        self.bot    = down_v2(nf_t*8, nf_t*16)
        self.u3     = up_v2(nf_t*16, nf_t*8)
        self.u2     = up_v2(nf_t*8, nf_t*4)
        self.u1     = up_v2(nf_t*4, nf_t*2)
        self.u0     = up_v2(nf_t*2, nf_t)

        self.padding_mode = padding_mode

    def forward(self, conv_prior, conv_data, gamma, x, y, v, z, k, mp, md, mn, ui=None, uj=None):

        f = conv_prior.weight
        g = conv_data.weight
        kt = torch.flip(k, [1, 2])
        Gy = conv_data(y)
        Gy = Gy - z if uj is None else Gy - z + uj
        v  = v if ui is None else v - ui
        b  = self.Ad(v, Gy, kt, f, g, mp, md, mn)
        b  = b[0] + b[1]
        #CGM
        r0  = b - self.A(conv_prior, conv_data, x, k, kt, f, g, mp, md, mn)
        r1  = torch.cat(self.Ae(conv_prior, conv_data, r0, k, mp, md), dim=1)
        r2 = self.d1(r1)
        r3 = self.d2(r2)
        r4 = self.d3(r3)
        r5 = self.bot(r4)
        r6 = self.u3(r5, r4)
        r7 = self.u2(r6, r3)
        r8 = self.u1(r7, r2)
        r  = self.u0(r8, r1)
        r  = self.Ad(r[:, :self.nf_p, ...], r[:, self.nf_p:, ...], kt, f, g, mp, md, mn)
        r  = r[0] + r[1]
        x  = x + r + gamma * r0
        return x

    def A(self, conv_p, conv_d, x, k, kt, f, g, mp, md, mn):
        x_d = bconv(x, k, self.padding_mode)
        x_d = conv_d(x_d).mul(torch.square(md))
        x_d = F.conv_transpose2d(x_d, g, padding=g.size(-1)//2)
        x_d = bconv(x_d, kt, 'zeros')
        x_p = conv_p(x).mul(torch.square(mp))
        x_p = F.conv_transpose2d(x_p, f, padding=f.size(-1)//2)

        return mn*x_p + x_d

    def Ae(self, conv_p, conv_d, x, k, mp, md):
        x_d = bconv(x, k, self.padding_mode)
        x_d = conv_d(x_d)
        x_p = conv_p(x)

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
    delta[:, -1, ...] = 1.
    pad = [rf//2]*4
    delta = F.pad(delta, pad)
    K = torch.zeros(nf, in_nc, rf, rf).type(type)
    for i in range(in_nc):
        delta = torch.roll(delta, 1, 1)
        KC = conv(delta).rot90(2, (-1, -2))
        K[:, i, ...] = KC
    return K


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, bias):
        super(FeedForward, self).__init__()
        h_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, h_dim*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(h_dim*2, h_dim*2, kernel_size=3, stride=1, padding=1, groups=h_dim*2, bias=bias)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=h_dim, out_channels=h_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.project_out = nn.Conv2d(h_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = x * self.sca(x)
        x = self.project_out(x)
        return x


class down(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(down, self).__init__()
        self.d = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_nc, out_nc, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_nc, out_nc, 3, padding=1),
            nn.ReLU(True))

    def forward(self, x):
        return self.d(x)


class down_v2(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(down_v2, self).__init__()
        self.d = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1)
        self.m = FeedForward(out_nc, 2.66, False)
        self.n = LayerNorm(out_nc, 'WithBias')

    def forward(self, x):
        x = self.d(x)
        return x + self.m(self.n(x))


class up(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(up, self).__init__()
        self.u = nn.ConvTranspose2d(in_nc, out_nc, 3, stride=2, padding=1)
        self.m = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_nc, out_nc, 3, padding=1),
            nn.ReLU(True))

    def forward(self, x, s):
        x = self.u(x, output_size=s.shape)
        x = torch.cat((x, s), dim=1)
        return self.m(x)


class up_v2(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(up_v2, self).__init__()
        self.u = nn.Sequential(
            nn.Conv2d(in_nc, in_nc*2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2))
        self.r = nn.Conv2d(in_nc, out_nc, 1)
        self.m = FeedForward(out_nc, 2.66, False)
        self.n = LayerNorm(out_nc, 'WithBias')

    def forward(self, x, s):
        x = self.u(x)
        if x.shape[-2:] != s.shape[-2:]:
            x = x[..., :s.shape[-2], :s.shape[-1]]
        x = self.r(torch.cat([x, s], dim=1))
        x = x + self.m(self.n(x))
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PriorNet(nn.Module):
    def __init__(self, in_nc, nf=49, kernel_size=7, max_nc=4, drop_p=0.5, padding_mode='circular'):
        super(PriorNet, self).__init__()
        pad = kernel_size // 2  # if padding_mode == 'circular' else 0
        self.conv = nn.Conv2d(in_nc, nf, kernel_size, padding=pad, bias=False)
        self.m = Maxout(nf, max_nc, drop_p)

    def forward(self, x, ui=None):
        fx = self.conv(x)
        v  = self.m(fx, ui)
        return v, fx


class DataNet(nn.Module):
    def __init__(self, in_nc, nf=48, kernel_size=7, max_nc=4, drop_p=0.5, padding_mode='circular'):
        super(DataNet, self).__init__()
        pad = kernel_size // 2  # if padding_mode =='circular' else 0
        self.conv = nn.Conv2d(in_nc, nf, kernel_size, padding=pad, bias=False)
        self.m = Maxout(nf, max_nc, drop_p)
        self.padding_mode = padding_mode

    def forward(self, y, x, k, uj=None):
        kx  = bconv(x, k, self.padding_mode)
        r   = y - kx
        gr  = self.conv(r)
        z   = self.m(gr, uj)
        return z, gr


class DSDNet(nn.Module):
    def __init__(self, n_iter=3, in_nc=3, out_nc=3, nf_p=49, nf_d=48, max_nc=4, kernel_size=7, dropout_rate=0.5, padding_mode='circular', task='db'):
        super(DSDNet, self).__init__()
        self.in_nc  = in_nc
        self.n      = n_iter
        self.hp     = nn.ModuleList([HypNet(in_nc*2, nf_p) for _ in range(n_iter)])
        self.hd     = nn.ModuleList([HypNet(in_nc*2, nf_d) for _ in range(n_iter)])
        self.hn     = nn.ModuleList([NLNet(in_nc) for _ in range(n_iter)])
        self.p      = nn.ModuleList([PriorNet(in_nc, nf_p, kernel_size, max_nc=max_nc, drop_p=dropout_rate, padding_mode=padding_mode) for _ in range(n_iter)])
        self.d      = nn.ModuleList([DataNet(in_nc, nf_d, kernel_size, max_nc=max_nc, drop_p=dropout_rate, padding_mode=padding_mode) for _ in range(n_iter)])
        self.i      = nn.ModuleList([CGUNet(self.p[i].conv, self.d[i].conv, in_nc, nf_p, nf_d, padding_mode=padding_mode) for i in range(n_iter)])

    def forward(self, y, k):
        x = fft_deconv(y, k)

        for i in range(self.n):
            w = torch.cat((x, torch.ones_like(x)), dim=1)
            mn     = self.hn[i](x)
            mp     = self.hp[i](w)
            md     = self.hd[i](w)
            v, fx  = self.p[i](x) if i == 0 else self.p[i](x, ui)
            z, gr  = self.d[i](y, x, k) if i == 0 else self.d[i](y, x, k, uj)
            x      = self.i[i](x, y, v, z, k, mp, md, mn) if i == 0 else self.i[i](x, y, v, z, k, mp, md, mn, ui, uj)
            ui = fx - v if i == 0 else ui + fx - v
            uj = gr - z if i == 0 else uj + gr - z

        return x


class DSDNet_plus(nn.Module):
    def __init__(self, n_iter=3, in_nc=3, out_nc=3, nf_p=49, nf_d=48, max_nc=4, kernel_size=7, dropout_rate=0.5, padding_mode='circular', task='db'):
        super(DSDNet_plus, self).__init__()
        self.in_nc  = in_nc
        self.n      = n_iter
        self.mp     = nn.Parameter(torch.ones(n_iter))
        self.md     = nn.Parameter(torch.ones(n_iter))
        self.hn     = nn.ModuleList([NLNet_v2(in_nc) for _ in range(n_iter)])
        self.p      = nn.ModuleList([PriorNet(in_nc, nf_p, kernel_size, max_nc=max_nc, drop_p=dropout_rate, padding_mode=padding_mode) for _ in range(n_iter)])
        self.d      = nn.ModuleList([DataNet(in_nc, nf_d, kernel_size, max_nc=max_nc, drop_p=dropout_rate, padding_mode=padding_mode) for _ in range(n_iter)])
        self.gamma  = nn.Parameter(torch.ones(n_iter, 1))
        self.i      = KryNet(in_nc, nf_p, nf_d, padding_mode=padding_mode)

    def forward(self, y, k):
        x = fft_deconv(y, k)
        output = []
        for i in range(self.n):
            mp, md, mn = torch.sigmoid(self.mp[i]), torch.sigmoid(self.md[i]), self.hn[i](x)
            v, fx  = self.p[i](x) if i == 0 else self.p[i](x, ui)
            z, gr  = self.d[i](y, x, k) if i == 0 else self.d[i](y, x, k, uj)
            x      = self.i(self.p[i].conv, self.d[i].conv, self.gamma[i], x, y, v, z, k, mp, md, mn) if i == 0 else self.i(self.p[i].conv, self.d[i].conv, self.gamma[i], x, y, v, z, k, mp, md, mn, ui, uj)
            ui = fx - v if i == 0 else ui + fx - v
            uj = gr - z if i == 0 else uj + gr - z

            output.append(x)

        return output
