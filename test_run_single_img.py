import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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

def check_input_size(img):
    return


network = make_network(cfg).cuda()
load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
network.eval()

#data_loader = make_data_loader(cfg, is_train=False)
visualizer = make_visualizer(cfg)

img_PTL_RGB = Image.open('/home/kuka/3D_Detection/clean-pvnet-master/data/custom/test_data/rgb/rgb-8.jpg').convert('RGB')
img_cv_BGR = cv2.imread('/home/kuka/3D_Detection/clean-pvnet-master/data/custom/test_data/rgb/rgb-8.jpg',cv2.IMREAD_COLOR)
img_cv_RGB = cv2.cvtColor(img_cv_BGR,cv2.COLOR_BGR2RGB)#network input is RGB

trans = make_transforms(cfg, is_train=False)
n_img_PTL_RGB = trans(img_PTL_RGB)[0]
n_img_cv_RGB = trans(img_cv_RGB)[0]

t_img_PTL_RGB = torch.from_numpy(n_img_PTL_RGB)
t_img_cv_RGB = torch.from_numpy(n_img_cv_RGB)
print(t_img_PTL_RGB.size())

t_img_PTL_RGB = t_img_PTL_RGB.view(1,*t_img_PTL_RGB.size())
t_img_cv_RGB = t_img_cv_RGB.view(1,*t_img_cv_RGB.size())
print(t_img_PTL_RGB.size())
print(t_img_PTL_RGB.dtype)

output = network(t_img_cv_RGB.cuda())
batch={'inp':img_cv_BGR,'cv_img':img_cv_BGR}

# output = network(t_img_PTL_RGB.cuda())
# batch={'inp':t_img_PTL_RGB}

image,_ = visualizer.visualize_cv_inp(output,batch)

cv2.imshow('mask',image)
cv2.imshow('img',img_cv_BGR)
cv2.waitKey(0)


