import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import random
import torch
from lib.config import cfg
import cv2
scale_ratio = 1.3
Output = [256,256]

class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg

    def extract_BBOX(self, mask) -> list:
        locs = []
        loc = np.where(mask > 0)
        x1 = np.min(loc[1])
        x2 = np.max(loc[1])
        y1 = np.min(loc[0])
        y2 = np.max(loc[0])
        locs.append([x1, y1, x2, y2])
        # cv2.imshow('mat',mat)
        # cv2.waitKey(0)
        return locs

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        inp = Image.open(path)

        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)

        if inp.width > 256 and inp.height > 256:
            bbox = self.extract_BBOX(mask)[0]
            inp_cv = cv2.cvtColor(np.asarray(inp),cv2.COLOR_RGB2BGR)
            inp_cv_sub,mask,kpt_2d, trans_input, trans_input_inv = pvnet_data_utils.corp_img(inp_cv,mask,kpt_2d,
                                                                       bbox, scale_ratio, Output)
            inp = Image.fromarray(cv2.cvtColor(inp_cv_sub,cv2.COLOR_BGR2RGB))
            do_crop =True
            # for i in range(kpt_2d.shape[0]):
            #     cv2.circle(inp_cv_sub, tuple(kpt_2d[i].astype('int')), radius=1, color=(0, 255, 0), thickness=2)
            # cv2.imshow('sub',inp_cv_sub)
            # cv2.waitKey(0)
        else:
            do_crop = False
            trans_input = np.eye(4, 4)
            trans_input_inv = np.eye(4, 4)

        return inp,mask,kpt_2d, do_crop, trans_input, trans_input_inv

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img,mask,kpt_2d, do_crop, trans_input, trans_input_inv = self.read_data(img_id)
        if self.split == 'train':
            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)
        else:
            inp = img

        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'meta': {},
               'do_crop': do_crop, 'trans_input': trans_input, 'trans_input_inv': trans_input_inv,'kpt_2d':kpt_2d}
        # visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, True)

        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((9, 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            img, mask, hcoords = rotate_instance(img, mask, hcoords, self.cfg.train.rotate_min, self.cfg.train.rotate_max)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         self.cfg.train.overlap_ratio,
                                                         self.cfg.train.resize_ratio_min,
                                                         self.cfg.train.resize_ratio_max)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask
