import cv2
from matplotlib import image
from lib.config import cfg, args, reInit
import numpy as np
import os

from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_network
import tqdm
import torch
from torchvision import transforms
from lib.visualizers import make_visualizer
from lib.datasets.transforms import make_transforms

from PIL import Image
from lib.utils import data_utils


def corp_img(img,box,scale_ratio,output):
    input_h, input_w = output
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    scale = max(box[2] - box[0], box[3] - box[1]) * scale_ratio
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    img = img.astype(np.uint8).copy()
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    trans_input_inv = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h],inv=1)
    return inp,trans_input_inv


img = cv2.imread('/home/kuka/mmdetection-master/data/coco_rgb/val2017/rgb_image_00006.jpg')
rows,cols = img.shape[:2]

box=[665.5543, 473.70355, 995.3987, 828.8725, 1.0] #x1,y1,x2,y2,score
#box=[761.9117, 468.0333, 1095.9054, 868.8202, 1.0]
#box=[715.982, 645.658, 1103.6398, 1010.1474, 0.99999774]
scale_ratio = 1.3
output = [256,256]

inp,trans_input_inv=corp_img(img,box,scale_ratio,output)


# p1_img = data_utils.affine_transform(p1_inp,trans_input_inv)


network = make_network(cfg).cuda()
load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
network.eval()

#data_loader = make_data_loader(cfg, is_train=False)
visualizer = make_visualizer(cfg)

img_cv_BGR = inp
#img_cv_RGB = cv2.cvtColor(img_cv_BGR,cv2.COLOR_BGR2RGB)#network input is RGB
img_cv_RGB =img_cv_BGR[:,:,(0,1,2)] #here, this dataset input is BGR?


trans = make_transforms(cfg, is_train=False)
n_img_cv_RGB = trans(img_cv_RGB)[0]

t_img_cv_RGB = torch.from_numpy(n_img_cv_RGB)

t_img_cv_RGB = t_img_cv_RGB.view(1,*t_img_cv_RGB.size())
print(t_img_cv_RGB.size())
print(t_img_cv_RGB.dtype)

output = network(t_img_cv_RGB.cuda())

batch={'inp':img_cv_BGR,'cv_img':img_cv_BGR}
image_inp,_ = visualizer.visualize_cv_inp(output,batch)

batch={'inp':img_cv_BGR,'cv_img':img}
image_org,_=visualizer.visualize_cv_uncrop(output,batch,trans_input_inv)


cv2.imshow('img',image_org)

cv2.imshow('image_inp',image_inp)
#cv2.imshow('inp',img_cv_BGR)
cv2.waitKey(0)


