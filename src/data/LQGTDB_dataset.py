import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import data.util as util


def cirpad(x,k):
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
    p2d=(phl,phr,pvt,pvb)
    x=F.pad(x,p2d,mode='circular')
    return x


def reppad(x,k):
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
    p2d=(phl,phr,pvt,pvb)
    x=F.pad(x,p2d,mode='replicate')
    return x


def bconv(x,k):# N,C,H,W -> conv(C,N,H,W) -> N,C,H,W
    b = x.shape[0]
    x = x.permute(1,0,2,3)
    x = reppad(x,k)
    return F.conv2d(x,torch.from_numpy(k).unsqueeze(0).unsqueeze(0).float(),groups=b).permute(1,0,2,3)


def measure(x,k,m):

    x  = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
    x  = bconv(x,k)
    if m.shape[-1]==3 and len(m.shape)==3:
        m = np.transpose(m[:,:,[2,1,0]],(2,0,1))#permute for matching BGR of CV2 format
    x  = x.mul(torch.from_numpy(m).float())

    return x.squeeze(0).permute(1,2,0).numpy()


class LQGTDBDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(LQGTDBDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']

        self.GT_env = None
        self.LQ_env = None
        self.KR_env = None

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        self.paths_KR, self.sizes_KR = util.get_image_paths(self.data_type, opt['dataroot_KM'])

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]


    def __getitem__(self, index):

        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]

        img_GT = util.read_img(self.GT_env, GT_path)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        if self.opt['color']:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LQ image, kernel and mask
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            KR_path = self.paths_KR[index]
            img_LQ = util.read_img(self.LQ_env, LQ_path)
            kernel = util.read_ker(self.KR_env, KR_path)
            mask   = np.ones_like(img_LQ)


        if self.opt['color']:  # change color space if necessary
            img_LQ = util.channel_convert(img_LQ.shape[2], self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        kernel = torch.from_numpy(np.ascontiguousarray(kernel)).float()
        if len(mask.shape)==2:
            mask = np.expand_dims(mask,axis=2)
        mask   = torch.from_numpy(np.ascontiguousarray(np.transpose(mask,   (2, 0, 1)))).float()
        if mask.shape[0] != img_GT.shape[0]:
            mask = torch.cat([mask]*3, dim=0)

        if LQ_path is None:
            LQ_path = GT_path

        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path, 'kernel': kernel, 'mask': mask}

    def __len__(self):
            return len(self.paths_GT)

