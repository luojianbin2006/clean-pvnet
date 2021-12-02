from lib.config import cfg, args, reInit
import matplotlib.pyplot as plt
import cv2
import numpy as np
from lib.utils import data_utils
import open3d as o3d

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

img1=img.transpose(2,0,1)
cv2.imshow('img',img)
cv2.imshow('img1',img1)
cv2.waitKey(0)


box=[665.5543, 473.70355, 995.3987, 828.8725, 1.0] #x1,y1,x2,y2,score
scale_ratio = 1.3
output = [256,256]

inp,trans_input_inv=corp_img(img,box,scale_ratio,output)

p1_inp = [100,50]
cv2.circle(inp,p1_inp,3,(0,0,255),thickness=2)

p1_img = data_utils.affine_transform(p1_inp,trans_input_inv)
cv2.circle(img,p1_img.astype('int'),3,(0,0,255),thickness=2)

plt.figure(figsize=(8,8))
plt.subplot(221)
plt.imshow(img[:,:,::-1])

plt.subplot(222)
plt.imshow(inp[:,:,::-1])

plt.subplots_adjust(top=0.8, bottom=0.08, left=0.10, right=0.95, hspace=0,wspace=0.35)
plt.show()
