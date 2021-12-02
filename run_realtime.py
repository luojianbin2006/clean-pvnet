from typing import Union
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

import open3d as o3d
import copy

import ros_numpy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2 

import message_filters
from sensor_msgs.msg import Image as ImageMsg
import time
import rospy
from std_msgs.msg import String,Header,Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
IMAGE_WIDTH=1920
IMAGE_HEIGHT=1200
scale_ratio = 1.3
Output = [256,256]


def corp_img(img,box,scale_ratio,output_):
    input_h, input_w = output_
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    scale = max(box[2] - box[0], box[3] - box[1]) * scale_ratio
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    img = img.astype(np.uint8).copy()
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    trans_input_inv = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h],inv=1)
    return inp,trans_input_inv


def extract_masks(subcriber_mask:ImageMsg)->Union[list,list]:
    mats=[]
    locs=[]
    num=(int)(subcriber_mask.data.__len__()/subcriber_mask.height/subcriber_mask.width)
    for i in range(num):
        mat=np.frombuffer(subcriber_mask.data,dtype=np.uint8,count=subcriber_mask.height*subcriber_mask.width,
        offset=subcriber_mask.height*subcriber_mask.width*i).reshape(subcriber_mask.height,subcriber_mask.width)
        mats.append(mat)
        loc=np.where(mat>0)
        x1=np.min(loc[1])
        x2=np.max(loc[1])
        y1=np.min(loc[0])
        y2=np.max(loc[0])
        locs.append([x1,y1,x2,y2])
        # cv2.imshow('mat',mat)
        # cv2.waitKey(0)
    return mats,locs

def extract_ROI_cloud(org_cloud:np.array,mask:np.array):
    h,w =mask.shape
    index = mask.reshape(h*w)
    index=index>0
    cloud = org_cloud[index,:]

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud)
    print('downbefore:%d'% np.asarray(pcd1.points).__len__())
    pcd1=pcd1.voxel_down_sample(0.005)

    pcd1,ind=pcd1.remove_statistical_outlier(nb_neighbors=400,std_ratio=0.3)
    print('downafter:%d'% np.asarray(pcd1.points).__len__())

    return pcd1

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def multi_callback(subcriber_image, subcriber_mask,subcriber_pointcloud):
    global coco_demo
    rospy.loginfo(rospy.get_caller_id() + " I heard ")  

    # cloud = pc2.read_points(subcriber_pointcloud,skip_nans=False,field_names=("x","y","z"))
    # print (type(cloud))
    # points_ = np.array(list(cloud))
    points = ros_numpy.numpify(subcriber_pointcloud)
    h,w=points.shape
    np_points = np.zeros((h*w,3),dtype=np.float32)
    x=points['x']
    np_points[:,0]=np.resize(points['x'],h*w)
    np_points[:,1]=np.resize(points['y'],h*w)
    np_points[:,2]=np.resize(points['z'],h*w)

    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(subcriber_image,"bgr8")
    cv_image_show = np.copy(cv_image)
    cv_image_show2 = np.copy(cv_image)
    header_re = subcriber_image.header

    masks,locs=extract_masks(subcriber_mask)

    for i in range(masks.__len__()):
        pcd1=extract_ROI_cloud(np_points,masks[i])
  

        t_s=time.time()  
        inp,trans_input_inv=corp_img(cv_image,locs[i],scale_ratio,Output)
        mask_inp,_=corp_img(masks[i],locs[i],scale_ratio,Output)
        img_cv_BGR = inp
        #img_cv_RGB = cv2.cvtColor(img_cv_BGR,cv2.COLOR_BGR2RGB)#network input is RGB
        img_cv_RGB =img_cv_BGR[:,:,(0,1,2)] #here, this dataset input is BGR?

        trans = make_transforms(cfg, is_train=False)
        n_img_cv_RGB = trans(img_cv_RGB)[0]
        t_img_cv_RGB = torch.from_numpy(n_img_cv_RGB)
        t_img_cv_RGB = t_img_cv_RGB.view(1,*t_img_cv_RGB.size())

        t_mask_inp = torch.from_numpy(mask_inp.astype('int64'))
        t_mask_inp = t_mask_inp.view(1,*t_mask_inp.size())

        #net_output = network(t_img_cv_RGB.cuda())
        net_output = network(t_img_cv_RGB.cuda(),t_mask_inp.cuda())

        # batch={'inp':img_cv_BGR,'cv_img':img_cv_BGR}
        # image_inp,_ = visualizer.visualize_cv_inp(net_output,batch)

        batch={'cv_img_for_mask':cv_image_show2,'cv_img':cv_image_show}
        image_org,img_mask,pose=visualizer.visualize_cv_uncrop(net_output,batch,trans_input_inv)
        cv_image_show=image_org
        cv_image_show2=img_mask

        # cv2.imshow('inp',image_inp)
        # # cv2.imshow('mask',masks[i]*255)
        # cv2.imshow('img',image_org)
        # cv2.waitKey(1000)
        t_e=time.time()
        print('match time:'+str(t_e-t_s))

        publish_image_mask(cv_image_show2,header_re)
        publish_image_result(cv_image_show, header_re)

        M1 = np.array([[1., 0., 0., 0.],
                      [0., -1., 0., 0.],
                      [0., 0., -1., 0.],
                      [0., 0., 0., 1.]])
        
        model_cloud_=copy.deepcopy(model_cloud)
        model_cloud_.transform(pose)

        pose1=np.dot(M1,pose)

        pcd2_1 =copy.deepcopy(model_cloud)
        d1=pcd2_1.transform(pose1)



        #ICP
        threshold = 0.5
        trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                            [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                            [0,0,1,0],   # 这个矩阵为初始变换
                            [0,0,0,1]])
        source=pcd2_1      
        target=pcd1              
        reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 50))

        source_temp = copy.deepcopy(source)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(reg_p2p.transformation)
        cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0,0,0])
        o3d.visualization.draw_geometries([pcd1,source_temp,pcd2_1,model_cloud_,model_cloud,cam_axis],window_name='pcd')


def publish_image_result(imgdata, header):
    image_temp=ImageMsg()
    #header = Header(stamp=rospy.Time.now())
    #header.frame_id = 'rgb_camera_link'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='bgr8'
    image_temp.data=np.array(imgdata).tostring()
    #print(imgdata)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=1920
    image_publish_rgb_result.publish(image_temp)

def publish_image_mask(imgdata, header):
    image_temp=ImageMsg()
    #header = Header(stamp=rospy.Time.now())
    #header.frame_id = 'rgb_camera_link'
    image_temp.height=IMAGE_HEIGHT
    image_temp.width=IMAGE_WIDTH
    image_temp.encoding='bgr8'
    image_temp.data=np.array(imgdata).tostring()
    #print(imgdata)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=1920
    image_publish_mask.publish(image_temp)


def publish_inp_mask(imgdata, header):
    image_temp=ImageMsg()
    #header = Header(stamp=rospy.Time.now())
    #header.frame_id = 'rgb_camera_link'
    image_temp.height=Output[1]
    image_temp.width=Output[0]
    image_temp.encoding='bgr8'
    image_temp.data=np.array(imgdata).tostring()
    #print(imgdata)
    #image_temp.is_bigendian=True
    image_temp.header=header
    image_temp.step=Output[1]
    image_publish_mask.publish(image_temp)


model_cloud = o3d.io.read_point_cloud('/home/kuka/3D_Detection/data_pvnet/custom/model.ply')
print('downbefore:%d'% np.asarray(model_cloud.points).__len__())
model_cloud=model_cloud.voxel_down_sample(0.005)
print('downafter:%d'% np.asarray(model_cloud.points).__len__())
#o3d.visualization.draw_geometries([model_cloud],window_name='model_cloud')

network = make_network(cfg).cuda()
load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
network.eval()

#data_loader = make_data_loader(cfg, is_train=False)
visualizer = make_visualizer(cfg)

rospy.init_node("rgb_pose", anonymous=True)
image_publish_mask=rospy.Publisher('/rgb_pose/img_mask',ImageMsg,queue_size=10)

image_publish_rgb_result=rospy.Publisher('/rgb_pose/img_result',ImageMsg,queue_size=10)



subcriber_image = message_filters.Subscriber('/mechmind/color_image', ImageMsg, queue_size=10)
subcriber_mask  = message_filters.Subscriber('/image_mask_kuka', ImageMsg, queue_size=10)
#obj1_x1,bj1_y1,bj1_x2,bj1_y2,obj2_x1,bj2_y1,bj2_x2,bj2_y2....
subcriber_locations  = message_filters.Subscriber('/image_mask_location', Float32MultiArray, queue_size=10)
subcriber_pointcloud  = message_filters.Subscriber('/mechmind/point_cloud', PointCloud2, queue_size=10)
sync = message_filters.ApproximateTimeSynchronizer([subcriber_image, subcriber_mask,subcriber_pointcloud],10,0.1,allow_headerless=True)
sync.registerCallback(multi_callback)

rospy.loginfo(rospy.get_caller_id() + " start rospy.spin...")
rospy.spin()



