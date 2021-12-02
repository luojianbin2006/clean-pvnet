from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
import cv2
from lib.utils import data_utils


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = output['mask'][0].detach().cpu().numpy()
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        print('kpt_2d')
        print(kpt_2d)
       


        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        fig = plt.figure()
        ax, ax2 = fig.subplots(1, 2, sharey=True)

        ax.imshow(inp)
        ax2.imshow(mask)
        for i in range(kpt_2d.shape[0]):
            print(kpt_2d[i])
            circle = Circle(xy=kpt_2d[i],    # 圆心坐标
               radius=2,    # 半径
               fc='white',    # facecolor
               ec='cornflowerblue',    # 浅蓝色，矢车菊蓝
               
              )
            ax.add_patch(p=circle)

        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.savefig('test.jpg')
        plt.close(0)


    def visualize_(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = output['mask'][0].detach().cpu().numpy()
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        print('kpt_2d')
        print(kpt_2d)
       
        
        fps_3d= [[-0.06156599894165993, -0.09982690215110779, 0.02286509983241558], [0.05790970101952553, -0.09977620095014572, 0.028202500194311142], [-0.07792600244283676, 0.0530925989151001, -0.01851690001785755], [0.06838379800319672, 0.0593469999730587, -0.015219800174236298], [0.07918979972600937, -0.03550710156559944, -0.030209600925445557], [-0.079647496342659, -0.04248090088367462, -0.03711619973182678], [-0.005793889984488487, 0.07201319932937622, -0.03725780174136162], [-0.0012174600269645452, -0.07993759959936142, 0.005319789983332157]]
        center_3d=[-4.194999999999893e-05, -0.013754249999999996, -0.004038749999999999]
        
        kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)
        K = np.array([[1540.43164062, 0.0, 128.0], [0.0, 1540.43164062, 128.0], [0.0, 0.0, 1.0]])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d= [[-0.0805148, -0.100964, -0.0383363], [-0.0805148, -0.100964, 0.0302588], [-0.0805148, 0.0734555, -0.0383363], [-0.0805148, 0.0734555, 0.0302588], [0.0804309, -0.100964, -0.0383363], [0.0804309, -0.100964, 0.0302588], [0.0804309, 0.0734555, -0.0383363], [0.0804309, 0.0734555, 0.0302588]]
        corner_3d = np.array(corner_3d)

        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        fig = plt.figure()
        ax, ax2 = fig.subplots(1, 2, sharey=True)

        ax.imshow(inp)
        ax2.imshow(mask)
        for i in range(kpt_2d.shape[0]):
            print(kpt_2d[i])
            circle = Circle(xy=kpt_2d[i],    # 圆心坐标
               radius=2,    # 半径
               fc='white',    # facecolor
               ec='cornflowerblue',    # 浅蓝色，矢车菊蓝
               
              )
            ax.add_patch(p=circle)

        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()


    def visualize_cv_inp(self, output, batch):
        inp = batch['inp']
        img = batch['cv_img']
        mask = output['mask'][0].detach().cpu().numpy().astype('uint8')
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        print('kpt_2d')
        print(kpt_2d)
       
        
        fps_3d= [[-0.06156599894165993, -0.09982690215110779, 0.02286509983241558], [0.05790970101952553, -0.09977620095014572, 0.028202500194311142], [-0.07792600244283676, 0.0530925989151001, -0.01851690001785755], [0.06838379800319672, 0.0593469999730587, -0.015219800174236298], [0.07918979972600937, -0.03550710156559944, -0.030209600925445557], [-0.079647496342659, -0.04248090088367462, -0.03711619973182678], [-0.005793889984488487, 0.07201319932937622, -0.03725780174136162], [-0.0012174600269645452, -0.07993759959936142, 0.005319789983332157]]
        center_3d=[-4.194999999999893e-05, -0.013754249999999996, -0.004038749999999999]
        
        kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)
        K = np.array([[2749.593, 0.0, 956.16], [0.0, 2749.804, 620.37], [0.0, 0.0, 1.0]])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d= [[-0.0805148, -0.100964, -0.0383363], [-0.0805148, -0.100964, 0.0302588], [-0.0805148, 0.0734555, -0.0383363], [-0.0805148, 0.0734555, 0.0302588], [0.0804309, -0.100964, -0.0383363], [0.0804309, -0.100964, 0.0302588], [0.0804309, 0.0734555, -0.0383363], [0.0804309, 0.0734555, 0.0302588]]
        corner_3d = np.array(corner_3d)

        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        redImg = np.zeros(img.shape, img.dtype)
        redImg[:,:] = (0, 0, 255)
        redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
        image=cv2.addWeighted(redMask, 0.25, img, 1, 0)

        for i in range(kpt_2d.shape[0]):
            cv2.circle(image, tuple(kpt_2d[i].astype('int')), radius=1, color=(0,255,0), thickness=2)

        dd= corner_2d_pred[[0, 1, 3, 2, 0]][np.newaxis,:,:]
        dd = dd[np.newaxis,:,:]
        cv2.polylines(image,corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]][np.newaxis,:,:].astype('int'),False,(255,0,0),1)
        cv2.polylines(image,corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]][np.newaxis,:,:].astype('int'),False,(255,0,0),1)

        return image,pose_pred


    def visualize_cv_uncrop(self, output, batch, trans_input_inv):
        img2 = batch['cv_img_for_mask']
        img = batch['cv_img']
        mask = output['mask'][0].detach().cpu().numpy().astype('uint8')
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        # print('kpt_2d')
        # print(kpt_2d)
       
        for i,p in enumerate(kpt_2d):
            kpt_2d[i]=data_utils.affine_transform(p,trans_input_inv)

        # print('trans kpt_2d')
        # print(kpt_2d)

        rows,cols = img.shape[:2]
        mask = cv2.warpAffine(mask, trans_input_inv, (cols, rows), flags=cv2.INTER_LINEAR)
        
        fps_3d= [[-0.06156599894165993, -0.09982690215110779, 0.02286509983241558], [0.05790970101952553, -0.09977620095014572, 0.028202500194311142], [-0.07792600244283676, 0.0530925989151001, -0.01851690001785755], [0.06838379800319672, 0.0593469999730587, -0.015219800174236298], [0.07918979972600937, -0.03550710156559944, -0.030209600925445557], [-0.079647496342659, -0.04248090088367462, -0.03711619973182678], [-0.005793889984488487, 0.07201319932937622, -0.03725780174136162], [-0.0012174600269645452, -0.07993759959936142, 0.005319789983332157]]
        center_3d=[-4.194999999999893e-05, -0.013754249999999996, -0.004038749999999999]
        
        kpt_3d = np.concatenate([fps_3d, [center_3d]], axis=0)
        K = np.array([[2749.593, 0.0, 956.16], [0.0, 2749.804, 620.37], [0.0, 0.0, 1.0]])

        #dist_coeffs=np.float32([[0,0,0,0,0]])
        dist_coeffs=np.array([[-0.05874,0.1484786,0.0002441,-0.000231656,0]])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K,dist_coeffs=dist_coeffs)
        R,T = pvnet_pose_utils.pnp_(kpt_3d, kpt_2d, K,dist_coeffs=dist_coeffs)

        corner_3d_= [[-0.0805148, -0.100964, -0.0383363], [-0.0805148, -0.100964, 0.0302588], [-0.0805148, 0.0734555, -0.0383363], [-0.0805148, 0.0734555, 0.0302588], [0.0804309, -0.100964, -0.0383363], [0.0804309, -0.100964, 0.0302588], [0.0804309, 0.0734555, -0.0383363], [0.0804309, 0.0734555, 0.0302588]]
        corner_3d_ = np.array(corner_3d_)


        corner_3d= [[0.078420698643,-0.034628000110,-0.029868800193],
        [0.079929597676,-0.053123898804,0.005412660073],
        [0.064694501460,-0.088080398738,0.010334799998],
        [0.059085100889,-0.098564900458,0.027516100556],
        [0.046896900982,-0.098787203431,0.029116300866],
        [0.038839198649,-0.094261400402,0.018769800663],
        [0.030524000525,-0.093829900026,0.017493899912],
        [0.022062599659,-0.098404198885,0.027310300618],
        [0.011879899539,-0.098980598152,0.028196500614],
        [0.005476119928,-0.095635399222,0.018456799909],
        [-0.007199539803,-0.095423996449,0.016361799091],
        [-0.012683399953,-0.098568797112,0.023970000446],
        [-0.024108100682,-0.099474802613,0.027328999713],
        [-0.030662899837,-0.095261998475,0.017186099663],
        [-0.044685799628,-0.094698198140,0.016522999853],
        [-0.050576798618,-0.099273897707,0.025798099115],
        [-0.063089199364,-0.098139598966,0.021770000458],
        [-0.064178101718,-0.091373898089,0.008467189968],
        [-0.078469298780,-0.064617298543,0.002527480014],
        [-0.080035403371,-0.043585199863,-0.034129299223],
        [-0.053292699158,-0.040756501257,-0.036202799529],
        [-0.037297401577,-0.057909101248,0.002260569949],
        [-0.024015599862,-0.058269601315,0.003150600009],
        [-0.046066001058,-0.004279349931,-0.005773659796],
        [-0.059589400887,0.012852899730,-0.027029899880],
        [-0.067024998367,0.053112801164,-0.015437800437],
        [-0.060664501041,0.046521499753,0.003837249940],
        [-0.062400501221,0.060547500849,0.007967679761],
        [-0.067451097071,0.064310103655,-0.005374010187],
        [-0.051149800420,0.066208600998,-0.009070119821],
        [-0.048850398511,0.071037203074,-0.031050799415],
        [-0.033409301192,0.073167100549,-0.036616999656],
        [-0.028345799074,0.069124698639,-0.025319699198],
        [-0.020022800192,0.068368397653,-0.025284200907],
        [-0.014958499931,0.072803497314,-0.035449501127],
        [-0.001263399958,0.072244800627,-0.033449601382],
        [0.001147820032,0.068460799754,-0.025926100090],
        [0.012957000174,0.069729700685,-0.023782800883],
        [0.019608199596,0.070458203554,-0.034021098167],
        [0.033946998417,0.070717900991,-0.031334601343],
        [0.036039099097,0.064418703318,-0.003271440044],
        [0.054294098169,0.063878901303,-0.001410300029],
        [0.052008498460,0.059416998178,0.008812669665],
        [0.052641600370,0.052822500467,0.009370059706],
        [0.058143299073,0.057466100901,-0.014679100364],
        [0.046932499856,0.008936639875,-0.024671800435],
        [0.040109399706,-0.001067630015,-0.005080199800],
        [0.019149400294,-0.056210499257,0.004101779778],
        [0.032329101115,-0.054195899516,0.004025040194],
        [0.044905498624,-0.036349598318,-0.031136600301]]
        corner_3d = np.array(corner_3d)


        corner_2d_pred_ = pvnet_pose_utils.project_(corner_3d_,R,T,K, dist_coeffs)
        corner_2d_pred = pvnet_pose_utils.project_(corner_3d, R,T,K, dist_coeffs)

        redImg = np.zeros(img.shape, img.dtype)
        redImg[:,:] = (0, 255, 0)

        mask_ = np.zeros(img.shape, img.dtype)
        cv2.fillPoly(mask_,corner_2d_pred[np.newaxis,:,:].astype('int'),(255,255,255))
        mask_=cv2.cvtColor(mask_,cv2.COLOR_RGB2GRAY)

        redMask = cv2.bitwise_and(redImg, redImg, mask=mask_)
        image=cv2.addWeighted(redMask, 0.25, img, 1, 0)

        for i in range(kpt_2d.shape[0]):
            cv2.circle(image, kpt_2d[i].astype('int'), radius=1, color=(0,255,0), thickness=2)

        cv2.polylines(image,corner_2d_pred_[[0, 1, 3, 2, 0, 4, 6, 2]][np.newaxis,:,:].astype('int'),False,(255,0,0),1)
        cv2.polylines(image,corner_2d_pred_[[5, 4, 6, 7, 5, 1, 3, 7]][np.newaxis,:,:].astype('int'),False,(255,0,0),1)

        #mask
        yImg = np.zeros(img.shape, img.dtype)
        yImg[:,:] = (0, 255, 255)

        mask_ = np.zeros(img.shape, img.dtype)
        cv2.fillPoly(mask_,corner_2d_pred[np.newaxis,:,:].astype('int'),(255,255,255))
        mask_=cv2.cvtColor(mask_,cv2.COLOR_RGB2GRAY)

        yMask = cv2.bitwise_and(yImg, yImg, mask=mask)
        image_mask=cv2.addWeighted(yMask, 0.45, img2, 1, 0)
        for i in range(kpt_2d.shape[0]):
            cv2.circle(image_mask, kpt_2d[i].astype('int'), radius=1, color=(0,255,0), thickness=2)


        return image,image_mask,np.concatenate((pose_pred,np.array([[0,0,0,1]]))) 






