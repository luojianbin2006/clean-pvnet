import os
import random
import shutil

data_root = '/home/kuka/3D_Detection/pvnet_data/custom'


def sample_wirte_txt():
    trainval_percent = 1  #
    train_percent = 0.8

    istrain = True
    if istrain:
        pose_dir = os.path.join(data_root, 'train_data/pose')
        rgb_dir = os.path.join(data_root, 'train_data/rgb')
        mask_dir = os.path.join(data_root, 'train_data/mask')
    else:
        pose_dir = os.path.join(data_root, 'test_data/pose')
        rgb_dir = os.path.join(data_root, 'test_data/rgb')
        mask_dir = os.path.join(data_root, 'test_data/mask')

    rgb_names = os.listdir(rgb_dir)
    pose_names = os.listdir(pose_dir)
    mask_names = os.listdir(mask_dir)

    num = len(rgb_names)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(list, tr)

    ftrainval = open(os.path.join(data_root, 'classify_data/trainval.txt'), 'w')
    ftest = open(os.path.join(data_root, 'classify_data/test.txt'), 'w')
    ftrain = open(os.path.join(data_root, 'classify_data/train.txt'), 'w')
    fval = open(os.path.join(data_root, 'classify_data/val.txt'), 'w')

    for i in list:
        name_num = rgb_names[i].split('-')[1].split('.')[0] + '\n'
        if i in trainval:
            ftrainval.write(name_num)
            if i in train:
                ftrain.write(name_num)
            else:
                fval.write(name_num)
        else:
            ftest.write(name_num)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


def classify_file():

    src_pose_dir = os.path.join(data_root, 'org_data/pose')
    src_rgb_dir = os.path.join(data_root, 'org_data/rgb')
    src_mask_dir = os.path.join(data_root, 'org_data/mask')

    dis_pose_dir_train = os.path.join(data_root, 'classify_data/train_data/pose')
    dis_rgb_dir_train = os.path.join(data_root, 'classify_data/train_data/rgb')
    dis_mask_dir_train = os.path.join(data_root, 'classify_data/train_data/mask')

    dis_pose_dir_test = os.path.join(data_root, 'classify_data/test_data/pose')
    dis_rgb_dir_test = os.path.join(data_root, 'classify_data/test_data/rgb')
    dis_mask_dir_test = os.path.join(data_root, 'classify_data/test_data/mask')

    num_train=0
    num_val=0
    with  open(os.path.join(data_root, 'classify_data/train.txt'), 'r') as f:
         for line in f.readlines():
            num = line.strip('\n')
            rgb_file = 'rgb-%s.jpg' %num
            mask_file ='sem-%s.png' %num
            pose_file = 'pose-%s.npy' %num

            dis_rgb_file = 'rgb-%s_.jpg' %num
            dis_mask_file ='sem-%s_.png' %num
            dis_pose_file = 'pose-%s_.npy' %num

            shutil.copy(src_rgb_dir + '/' + rgb_file, dis_rgb_dir_train + '/' + dis_rgb_file)
            shutil.copy(src_mask_dir + '/' + mask_file, dis_mask_dir_train + '/' + dis_mask_file)
            shutil.copy(src_pose_dir + '/' + pose_file, dis_pose_dir_train + '/' + dis_pose_file)
            num_train = num_train+1
            print('num_train:%d, num%s\n'%(num_train,num))

    with  open(os.path.join(data_root, 'classify_data/val.txt'), 'r') as f:
         for line in f.readlines():
            num = line.strip('\n')
            rgb_file = 'rgb-%s.jpg' %num
            mask_file ='sem-%s.png' %num
            pose_file = 'pose-%s.npy' %num

            dis_rgb_file = 'rgb-%s_.jpg' %num
            dis_mask_file ='sem-%s_.png' %num
            dis_pose_file = 'pose-%s_.npy' %num

            shutil.copy(src_rgb_dir + '/' + rgb_file, dis_rgb_dir_test + '/' + dis_rgb_file)
            shutil.copy(src_mask_dir + '/' + mask_file, dis_mask_dir_test + '/' + dis_mask_file)
            shutil.copy(src_pose_dir + '/' + pose_file, dis_pose_dir_test + '/' + dis_pose_file)
            num_val = num_val+1
            print('num_val:%d, num%s\n' % (num_val, num))

    print('num_train:%d, num_val:%d, total:%d\n'%(num_train,num_val,num_train+num_val))







classify_file()