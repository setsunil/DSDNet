# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch import from_numpy
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    triple_paths_from_folder)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_kl, measure


class NonBlindDeblurringDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(NonBlindDeblurringDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.kl_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_kl']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            # self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.kl_folder]
            # self.io_backend_opt['client_keys'] = ['lq', 'gt', 'kl']
            # self.paths = triple_paths_from_folder(
            #     [self.lq_folder, self.gt_folder, self.kl_folder], ['lq', 'gt', 'kl'])
            pass
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            # self.paths = triple_paths_from_folder(
            #     [self.lq_folder, self.gt_folder, self.kl_folder], ['lq', 'gt', 'kl'],
            #     self.opt['meta_info_file'], self.filename_tmpl)
            pass
        else:
            self.paths = triple_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.kl_folder], ['lq', 'gt', 'kl'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        kl_path = self.paths[index]['kl_path']
        img_bytes = self.file_client.get(kl_path, 'kl')
        try:
            img_kl = imfrombytes(img_bytes, float32=True)
            img_kl /= (np.sum(img_kl) / 3)
        except:
            raise Exception("kl path {} not working".format(kl_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            kl_size = self.opt['kl_size']
            img_kl = padding_kl(img_kl, kl_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # img_lq = measure(img_gt, img_kl)
            # flip, rotation
            img_gt, img_lq, img_kl = augment([img_gt, img_lq, img_kl], self.opt['use_flip'],
                                             self.opt['use_rot'])

            noise = np.random.randn(*img_lq.shape)*0.05*np.random.rand(1)
            img_lq += noise[..., ::-1]

            noise = from_numpy(np.ascontiguousarray(np.transpose(noise, (2, 0, 1)))).float()

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_kl = img2tensor([img_gt, img_lq, img_kl],
                                            bgr2rgb=True,
                                            float32=True)

        img_kl = img_kl[0, ...]
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        if self.opt['phase'] == 'train':
            return {
                'lq': img_lq,
                'gt': img_gt,
                'kl': img_kl,
                'noise': noise,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'kl_path': kl_path
            }
        else:
            return {
                'lq': img_lq,
                'gt': img_gt,
                'kl': img_kl,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'kl_path': kl_path
            }

    def __len__(self):
        return len(self.paths)
