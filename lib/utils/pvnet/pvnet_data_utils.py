import pycocotools.mask as mask_utils
import numpy as np
from plyfile import PlyData
from PIL import Image
from lib.utils import data_utils
import cv2
import copy

def binary_mask_to_polygon(binary_mask, tolerance=0):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) <= 4:
            continue
        contour = np.flip(contour, axis=1)
        contour = np.round(np.maximum(contour, 0)).astype(np.int32)
        polygons.append(contour)
    return polygons


def coco_poly_to_mask(poly, h, w):
    rles = mask_utils.frPyObjects(poly, h, w)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def compute_vertex(mask, kpt_2d):
    h, w = mask.shape
    m = kpt_2d.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]

    vertex = kpt_2d[None] - xy[:, None]
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm < 1e-3] += 1e-3
    vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    vertex_out = np.reshape(vertex_out, [h, w, m * 2])

    return vertex_out


def get_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    model = np.stack([x, y, z], axis=-1)
    return model


def read_linemod_mask(path, ann_type, cls_idx):
    if ann_type == 'real':
        mask = np.array(Image.open(path))
        if len(mask.shape) == 3:
            return (mask[..., 0] != 0).astype(np.uint8)
        else:
            return (mask != 0).astype(np.uint8)
    elif ann_type == 'fuse':
        return (np.asarray(Image.open(path)) == cls_idx).astype(np.uint8)
    elif ann_type == 'render':
        return (np.asarray(Image.open(path))).astype(np.uint8)


def read_tless_mask(ann_type, path):
    if ann_type == 'real':
        return (np.asarray(Image.open(path))).astype(np.uint8)
    elif ann_type == 'render':
        depth = np.asarray(Image.open(path))
        return (depth != 65535).astype(np.uint8)

def corp_img(img,mask,kpt_2d,box,scale_ratio,output_):
    input_h, input_w = output_
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    scale = max(box[2] - box[0], box[3] - box[1]) * scale_ratio
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    img = img.astype(np.uint8).copy()
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    trans_input_inv = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h],inv=1)

    mask_ = cv2.warpAffine(mask, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    kpt_2d_=copy.deepcopy(kpt_2d)
    for i, p in enumerate(kpt_2d):
        kpt_2d_[i] = data_utils.affine_transform(p, trans_input)
    return inp, mask_, kpt_2d_, trans_input, trans_input_inv