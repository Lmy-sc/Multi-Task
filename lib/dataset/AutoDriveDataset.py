import os
from cProfile import label
from operator import index

import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from fontTools.ttLib.tables.C_P_A_L_ import table_C_P_A_L_
from numpy.ma.core import indices
from requests.packages import target
from torch.utils.data import Dataset
from torch.utils.tensorboard.summary import image

from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout
import albumentations as A
from collections import OrderedDict

from torch.nn.utils.rnn import pad_sequence

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        #路径
        img_root1 = Path(cfg.DATASET.DATAROOT1)
        img_root2 = Path(cfg.DATASET.DATAROOT2)
        img_root3 = Path(cfg.DATASET.DATAROOT3)

        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        #是否是训练？test==val
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET

        #路径拼接
        self.img_root1 = img_root1 / indicator
        self.img_root2 = img_root2 / indicator
        self.img_root3 = img_root3 / indicator

        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator

        # self.label_list = self.label_root.iterdir()
        # self.mask_list = self.mask_root.iterdir()
        self.det_list = list(self.label_root.glob("*.txt")) if self.label_root.exists() else []
        self.seg_list = list(self.mask_root.glob("*.bmp")) if self.mask_root.exists() else []
        self.lane_list = list(self.lane_root.glob("*.png")) if self.lane_root.exists() else []

        # albumentation data arguments
        self.albumentations_transform = A.Compose([

            A.OneOf([
                A.MotionBlur(p=0.1),
                A.MedianBlur(p=0.1),
                A.Blur(p=0.1),
            ], p=0.2),

            A.GaussNoise(p=0.02),
            A.CLAHE(p=0.02),
            A.RandomBrightnessContrast(p=0.02),
            A.RandomGamma(p=0.02),
            A.ImageCompression(quality_lower=75, p=0.02),

            A.OneOf([
                A.RandomSnow(p=0.1),  # 加雪花
                A.RandomRain(p=0.1),  # 加雨滴
                A.RandomFog(p=0.1),  # 加雾
                A.RandomSunFlare(p=0.1),  # 加阳光
                A.RandomShadow(p=0.1),  # 加阴影
            ], p=0.2),

            A.OneOf([
                A.ToGray(p=0.1),
                A.ToSepia(p=0.1),
            ], p=0.2),

            ],
            
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
            additional_targets={'mask0': 'mask'})

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.mosaic_border = [-192, -320]

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)

        self.mosaic_rate = cfg.mosaic_rate
        self.mixup_rate = cfg.mixup_rate
        # self.mosaic_rate = 1
        # self.mixup_rate = 1
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    #4张图像拼接增强
    def load_mosaic(self, idx ,task):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        w_mosaic, h_mosaic = 640, 384

        yc = int(random.uniform(-self.mosaic_border[0], 2 * h_mosaic + self.mosaic_border[0])) # 192,3x192
        xc = int(random.uniform(-self.mosaic_border[1], 2 * w_mosaic + self.mosaic_border[1])) # 320,3x320
        #task = self.db[0][idx]["tape"]
        # indices =[]
        # if task== 'detect':
        #     indices=self.db[1]
        # elif task == 'seg':
        #     indices=self.db[2]
        # elif task == 'depth':
        #     indices = self.db[3]

        indices = range(len(self.db))
        if len(indices) < 3:
            print(f"[Warning] Task '{task}' has too few samples ({len(indices)}) for mosaic augmentation")

        # leng = range (len(indices))
        indices = [idx] + random.choices(indices, k=3)  # 3 additional iWmage indices
                        
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            # img, labels, seg_label, (h0,w0), (h, w), path = self.load_image(index), h=384, w = 640
            img, labels, seg_label, lane_label, (h0, w0), (h,w), path  = self.load_image(index,task)
                        
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((h_mosaic * 2, w_mosaic * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

                seg4 = np.full((h_mosaic * 2, w_mosaic * 2 ), 0, dtype=np.uint8)  # base image with 4 tiles

                lane4 = np.full((h_mosaic * 2, w_mosaic * 2), 0, dtype=np.uint8)  # base image with 4 tiles
                # 大图中左上角、右下角的坐标
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # 小图中左上角、右下角的坐标
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_mosaic * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_mosaic * 2), min(h_mosaic * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            if task == 'seg':
                seg4[y1a:y2a, x1a:x2a] = seg_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            if task == 'depth':
                lane4[y1a:y2a, x1a:x2a] = lane_label[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

            padw = x1a - x1b
            padh = y1a - y1b
            if task == 'detect':
                if len(labels):
                    labels[:, 1] += padw
                    labels[:, 2] += padh
                    labels[:, 3] += padw
                    labels[:, 4] += padh

                    labels4.append(labels)

        # Concat/clip labels
        if labels4:
            labels4 = np.concatenate(labels4, 0)

            new = labels4.copy()
            new[:, 1:] = np.clip(new[:, 1:], 0, 2*w_mosaic)
            new[:, 2:5:2] = np.clip(new[:, 2:5:2], 0, 2*h_mosaic)

            # filter candidates
            i = box_candidates(box1=labels4[:,1:5].T, box2=new[:,1:5].T)
            labels4 = labels4[i]
            labels4[:] = new[i]

        return img4, labels4, seg4, lane4, (h0, w0), (h, w), path

    # mixup：图像混合增强
    def mixup(self, im, labels, seg_label, depth_label, im2, labels2, seg_label2, depth_label2 ,task):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        labels_depth =None
        seg_label_final =None
        depth_label_final =None
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)

        def to_one_hot(mask, num_classes=8):
            # mask: H x W, values in [0, num_classes-1]
            return np.eye(num_classes)[mask]  # → H x W x C

        if task == 'detect':
            labels_depth = np.concatenate((labels, labels2), 0)

        elif task == 'seg':
            seg1 = to_one_hot(seg_label)  # H x W x C
            seg2 = to_one_hot(seg_label2)  # H x W x C
            seg_label_final = seg1 * r + seg2 * (1 - r)  # soft label
            seg_label_final = np.argmax(seg_label_final, axis=-1).astype(np.uint8)  # H x W

        elif task == 'depth':
            depth_label_final = (depth_label > 0) & (depth_label > 0)
            depth_label_final = np.where(depth_label_final, depth_label * r + depth_label * (1 - r), 0)

        #seg_label |= seg_label2
        #depth_label |= depth_label2
        return im, labels_depth, seg_label_final, depth_label_final

    def load_image(self, idx,task):
        # data = self.db[idx]
        # img1 = cv2.imread(data["image1"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # img2 = cv2.imread(data["image2"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # img3 = cv2.imread(data["image3"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        #
        # if img1 is None or img2 is None or img3 is None:
        #     raise FileNotFoundError(f"❌ 图像读取失败：{data['image']}")
        #
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        #
        # # 根据配置文件判断任务类型
        # # 加载目标检测标签 (如果存在)
        # det_label = None
        # if "label" in data and data["label"] is not None:
        #     det_label = data["label"]
        #     det_label = np.array(det_label)  # 确保标签是数组格式
        #
        # # 加载分割标签 (如果存在)
        # seg_label = None
        # if "mask" in data and data["mask"] is not None:
        #     seg_label = cv2.imread(data["mask"])  # 读取彩色分割标签
        #
        # # 加载车道线标签 (如果存在)
        # lane_label = None
        # if "lane" in data and data["lane"] is not None:
        #     lane_label = cv2.imread(data["lane"], 0)  # 读取车道线标签
        #
        # # 图像大小和目标大小调整
        # resized_shape = self.inputsize
        # if isinstance(resized_shape, list) or isinstance(resized_shape, tuple):
        #     resized_shape = max(resized_shape)  # 取最大边，确保是 int
        #
        # h0, w0 = img.shape[:2]
        # max_h, max_w = 480, 640
        # # 再计算 resize 比例
        # r = resized_shape / max(max_h, max_w)
        #
        # # 计算 resize 后的新尺寸
        # new_h, new_w = int(max_h * r), int(max_w * r)
        # interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #
        # # resize 三张图
        # img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        # if seg_label is not None:
        #     seg_label = cv2.resize(seg_label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # if lane_label is not None:
        #     lane_label = cv2.resize(lane_label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # # if isinstance(resized_shape, list):
        # #     resized_shape = max(resized_shape)
        # # h0, w0 = img.shape[:2]  # 原始高宽
        # # r = resized_shape / max(h0, w0)  # 计算缩放比例
        # #
        # # if r != False:  # 如果缩放比例不是1，进行调整
        # #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        # #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        # #     print("11111111111111111")
        # #     print(img.shape[:2])
        # #     if seg_label is not None:
        # #         seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        # #
        # #     if lane_label is not None:
        # #         lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        # #
        #
        # h, w = img.shape[:2]  # 更新后的高度和宽度
        #
        # # 目标检测标签：将 YOLO 格式的 [cx, cy, w, h] 转为 [xmin, ymin, xmax, ymax]
        # labels = []
        # if det_label is not None and det_label.size > 0:
        #     labels = det_label.copy()
        #     labels[:, 1] = (det_label[:, 1] - det_label[:, 3] / 2) * w
        #     labels[:, 2] = (det_label[:, 2] - det_label[:, 4] / 2) * h
        #     labels[:, 3] = (det_label[:, 1] + det_label[:, 3] / 2) * w
        #     labels[:, 4] = (det_label[:, 2] + det_label[:, 4] / 2) * h
        #
        # if seg_label is not None:
        #     seg_label = torch.tensor(seg_label)
        #     # seg_label=self.Tensor(seg_label)
        #     r = (seg_label[:, :, 0] > 127).long()
        #     g = (seg_label[:, :, 1] > 127).long()
        #     b = (seg_label[:, :, 2] > 127).long()
        #     seg_label = (r << 2) + (g << 1) + b
        #     seg_label = seg_label.cpu()
        #
        #     # 步骤 B: 使用 .numpy() 方法将其转换为 NumPy 数组
        #     seg_label = seg_label.numpy()
        #
        #     # 步骤 C: （可选但通常推荐）转换数据类型
        #     # 你的位运算 (r << 2) + (g << 1) + b 会产生 0 到 7 之间的整数。
        #     # .long() 使 r, g, b 成为 torch.int64 类型，所以 final_seg_label_tensor 也是 torch.int64。
        #     # 转换后的 seg_label_numpy 将是 np.int64 类型。
        #     # 对于图像掩码或标签图，通常使用 np.uint8 类型更合适，因为值范围小。
        #     if seg_label.dtype != np.uint8:
        #         seg_label = seg_label.astype(np.uint8)

        data = self.db[idx]
        if self.is_train:
            if task == 'detect':
                data = {
                    'image': data['image1'],  # 保留 image2，改成统一的 key
                    'label': data['label'],
                    'mask': None,
                    'lane': None,
                    'tape': 'detect'
                }
            elif task == 'seg':
                data = {
                    'image': data['image2'],  # 保留 image2，改成统一的 key
                    'label': None,
                    'mask': data['mask'],
                    'lane': None,
                    'tape': 'seg'
                }
            elif task == 'depth':
                data = {
                    'image': data['image3'],  # 保留 image2，改成统一的 key
                    'label': None,
                    'mask': None,
                    'lane': data['lane'],
                    'tape': 'depth'
                }



        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if img is None:
            raise FileNotFoundError(f"❌ 图像读取失败：{data['image']}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 根据配置文件判断任务类型
        # 加载目标检测标签 (如果存在)
        det_label = None
        if "label" in data and data["label"] is not None:

            det_label = data["label"]
            det_label = np.array(det_label)  # 确保标签是数组格式

        # 加载分割标签 (如果存在)
        seg_label = None
        if "mask" in data and data["mask"] is not None:
            seg_label = cv2.imread(data["mask"])  # 读取彩色分割标签


        # 加载车道线标签 (如果存在)
        lane_label = None
        if "lane" in data and data["lane"] is not None:
            lane_label = cv2.imread(data["lane"],0)  # 读取车道线标签


        # 图像大小和目标大小调整
        resized_shape = self.inputsize
        if isinstance(resized_shape, list) or isinstance(resized_shape, tuple):
            resized_shape = max(resized_shape)  # 取最大边，确保是 int

        h0, w0 = img.shape[:2]
        max_h, max_w = 480,640
        # 再计算 resize 比例
        r = resized_shape / max(max_h, max_w)

        # 计算 resize 后的新尺寸
        new_h, new_w = int(max_h * r), int(max_w * r)
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR

        # resize 三张图
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        if seg_label is not None:
            seg_label = cv2.resize(seg_label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if lane_label is not None:
            lane_label = cv2.resize(lane_label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # if isinstance(resized_shape, list):
        #     resized_shape = max(resized_shape)
        # h0, w0 = img.shape[:2]  # 原始高宽
        # r = resized_shape / max(h0, w0)  # 计算缩放比例
        #
        # if r != False:  # 如果缩放比例不是1，进行调整
        #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        #     print("11111111111111111")
        #     print(img.shape[:2])
        #     if seg_label is not None:
        #         seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        #
        #     if lane_label is not None:
        #         lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        #

        h, w = img.shape[:2]  # 更新后的高度和宽度


        # 目标检测标签：将 YOLO 格式的 [cx, cy, w, h] 转为 [xmin, ymin, xmax, ymax]
        labels = []
        if det_label is not None and det_label.size > 0:
            labels = det_label.copy()
            labels[:, 1] = (det_label[:, 1] - det_label[:, 3] / 2) * w
            labels[:, 2] = (det_label[:, 2] - det_label[:, 4] / 2) * h
            labels[:, 3] = (det_label[:, 1] + det_label[:, 3] / 2) * w
            labels[:, 4] = (det_label[:, 2] + det_label[:, 4] / 2) * h

        if seg_label is not None:
            seg_label = torch.tensor(seg_label)
            # seg_label=self.Tensor(seg_label)
            r = (seg_label[:, :, 0] > 127).long()
            g = (seg_label[:, :, 1] > 127).long()
            b = (seg_label[:, :, 2] > 127).long()
            seg_label = (r << 2) + (g << 1) + b
            seg_label = seg_label.cpu()

            # 步骤 B: 使用 .numpy() 方法将其转换为 NumPy 数组
            seg_label = seg_label.numpy()

            # 步骤 C: （可选但通常推荐）转换数据类型
            # 你的位运算 (r << 2) + (g << 1) + b 会产生 0 到 7 之间的整数。
            # .long() 使 r, g, b 成为 torch.int64 类型，所以 final_seg_label_tensor 也是 torch.int64。
            # 转换后的 seg_label_numpy 将是 np.int64 类型。
            # 对于图像掩码或标签图，通常使用 np.uint8 类型更合适，因为值范围小。
            if seg_label.dtype != np.uint8:
                seg_label = seg_label.astype(np.uint8)


        # 返回图像和标签
        return img, labels, seg_label, lane_label, (h0, w0), (h, w), data['image']



    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """

        data1 = self.db[idx]

        if self.is_train:
            image_all = []
            target_all = []
            path_all = []
            shape_all = []
            task_all = []
            for i in range (3):
                if i ==0:
                    data = {
                        'image': data1['image1'],  # 保留 image2，改成统一的 key
                        'label': data1['label'],
                        'mask': None,
                        'lane': None,
                        'tape': 'detect'
                    }
                elif i == 1:
                    data = {
                        'image': data1['image2'],  # 保留 image2，改成统一的 key
                        'label': None,
                        'mask': data1['mask'],
                        'lane': None,
                        'tape': 'seg'
                    }
                elif i == 2:
                    data = {
                        'image': data1['image3'],  # 保留 image2，改成统一的 key
                        'label': None,
                        'mask': None,
                        'lane': data1['lane'],
                        'tape': 'depth'
                    }

                task = data["tape"]
            # indices = []
            # if task == 'detect':
            #     indices = self.db[1]
            # elif task == 'seg':
            #     indices = self.db[2]
            # elif task == 'depth':
            #     indices = self.db[3]
                indices = range(len(self.db))
                mosaic_this = False
                if random.random() < self.mosaic_rate:
                    mosaic_this = True
                    #  this doubles training time with inherent stuttering in tqdm, prob cpu or io bottleneck, does prefetch_generator work with ddp? (no improvement)
                    #  updated, mosaic is inherently slow, maybe cache the images in RAM? maybe it was IO bottleneck of reading 4 images everytime? time it
                    img, labels, seg_label, lane_label, (h0, w0), (h, w), path = self.load_mosaic(idx,task)


                    # mixup is double mosaic, really slow
                    if random.random() < self.mixup_rate and task == 'detect':
                        # img2, labels2, seg_label2, lane_label2, (_, _), (_, _), _ = self.load_mosaic(random.randint(0, len(self.db) - 1))
                        img2, labels2, seg_label2, lane_label2, (_, _), (_, _), _ = self.load_mosaic(random.choice(indices),task)
                        img, labels, seg_label, lane_label = self.mixup(img, labels, seg_label, lane_label, img2, labels2, seg_label2, lane_label2,task)

                else:
                    img, labels, seg_label, lane_label, (h0, w0), (h,w), path  = self.load_image(idx)



                try:
                    # labels = None
                    # seg_label = None
                    # depth_label = None
                    if task == 'detect':
                        new = self.albumentations_transform(image=img,
                                                            bboxes=labels[:, 1:] if len(labels) else labels,
                                                            class_labels=labels[:, 0] if len(labels) else labels)
                        labels = np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]) if len(
                            labels) else labels
                        img = new['image']
                    elif task == 'seg':
                        new = self.albumentations_transform(image=img, mask=seg_label)

                        seg_label = new['mask']
                        img = new['image']
                    elif task == 'depth':
                        new = self.albumentations_transform(image=img, depth=lane_label)

                        lane_label = new['depth']
                        img = new['image']
                except ValueError:  # bbox have width or height == 0
                    pass



                combination = (img, seg_label, lane_label)
                # combination = [img]  # 必须用 list 方便过滤
                # if seg_label is not None:
                #     combination.append(seg_label)
                # if lane_label is not None:
                #     combination.append(lane_label)

                (img, seg_label, lane_label), labels = random_perspective(
                    combination=combination,
                    targets=labels,
                    degrees=self.cfg.DATASET.ROT_FACTOR,
                    translate=self.cfg.DATASET.TRANSLATE,
                    scale=self.cfg.DATASET.SCALE_FACTOR,
                    shear=self.cfg.DATASET.SHEAR,
                    border=self.mosaic_border if mosaic_this else (0, 0)
                )



                # (img, seg_label, lane_label), labels = random_perspective(
                #     combination=combination,
                #     targets=labels,
                #     degrees=self.cfg.DATASET.ROT_FACTOR,
                #     translate=self.cfg.DATASET.TRANSLATE,
                #     scale=self.cfg.DATASET.SCALE_FACTOR,
                #     shear=self.cfg.DATASET.SHEAR,
                #     border=self.mosaic_border if mosaic_this else (0, 0)
                # )

                # img = combination[0]
                # seg_label = combination[1] if len(combination) > 1 else None
                # lane_label = combination[2] if len(combination) > 2 else None
                #img1 = self.transform(np.ascontiguousarray(img))


                augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)

                #random left-right flip
                if random.random() < 0.5:
                    img = np.fliplr(img)

                    if data['tape']=='detect':
                        rows, cols, channels = img.shape
                        x1 = labels[:, 1].copy()
                        x2 = labels[:, 3].copy()
                        x_tmp = x1.copy()
                        labels[:, 1] = cols - x2
                        labels[:, 3] = cols - x_tmp

                    if data['tape']=='seg':
                        seg_label = np.fliplr(seg_label)
                    if data['tape'] == 'depth':
                        lane_label = np.fliplr(lane_label)

                # random up-down flip
                if random.random() < 0.5:
                    img = np.flipud(img)

                    if data['tape']=='detect':
                        rows, cols, channels = img.shape
                        y1 = labels[:, 2].copy()
                        y2 = labels[:, 4].copy()
                        y_tmp = y1.copy()
                        labels[:, 2] = rows - y2
                        labels[:, 4] = rows - y_tmp
                    if data['tape'] == 'seg':
                        seg_label = np.flipud(seg_label)
                    if data['tape'] == 'depth':
                        lane_label = np.flipud(lane_label)

                (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), 640, auto=True,
                                                                     scaleup=self.is_train)
                shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

                labels_out = None
                if data['tape'] == 'detect':
                    if len(labels):
                        # update labels after letterbox
                        labels[:, 1] = ratio[0] * labels[:, 1] + pad[0]
                        labels[:, 2] = ratio[1] * labels[:, 2] + pad[1]
                        labels[:, 3] = ratio[0] * labels[:, 3] + pad[0]
                        labels[:, 4] = ratio[1] * labels[:, 4] + pad[1]

                        # convert xyxy to ( cx, cy, w, h )
                        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                    labels_out = torch.zeros((len(labels), 5))
                    if len(labels):
                        labels_out[:, :] = torch.from_numpy(labels)

                img = np.ascontiguousarray(img)

                if data['tape'] == 'depth':
                    if isinstance(lane_label, np.ndarray):
                        lane_label = self.Tensor(lane_label)
                    # 如果深度图是 [1, H, W]，可以保持，或者 squeeze 掉 1 个通道变成 [H, W]
                    if lane_label.ndim == 2:
                        lane_label = lane_label.unsqueeze(0)

                    # lane_label 就是单通道深度图
                target = [labels_out, seg_label, lane_label]

                def tensor_to_numpy_img(tensor_img):
                    """
                    支持将图像（Tensor或ndarray）转为可视化格式的uint8 HWC图像。
                    如果是float，会反标准化和放缩；如果是uint8则直接返回。
                    """
                    if isinstance(tensor_img, torch.Tensor):
                        img = tensor_img.clone().detach().cpu().numpy()
                        if img.ndim == 3 and img.shape[0] == 3:
                            img = img.transpose(1, 2, 0)  # C,H,W → H,W,C
                        elif img.ndim == 3 and img.shape[2] == 3:
                            pass  # 已是 HWC
                        else:
                            raise ValueError("图像维度异常，无法转换为OpenCV格式")

                        # 反标准化（如果做过 ImageNet 标准化）
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img = img * std + mean
                        img = (img * 255.0).clip(0, 255).astype(np.uint8)

                    elif isinstance(tensor_img, np.ndarray):
                        if tensor_img.dtype != np.uint8:
                            img = (tensor_img * 255.0).clip(0, 255).astype(np.uint8)
                        else:
                            img = tensor_img
                    else:
                        raise TypeError("不支持的图像类型")

                    return np.ascontiguousarray(img)

                img_vis = tensor_to_numpy_img(img)
                save_dir = 'D:/Multi-task/Git/Multi-Task/inference/image'
                os.makedirs(save_dir, exist_ok=True)
                base_name = Path(path).stem
                # if labels_out is not None and len(labels_out):
                #     for label in labels_out:
                #         cls_id, cx, cy, w, h = label.tolist()
                #         x1 = int(cx - w / 2)
                #         y1 = int(cy - h / 2)
                #         x2 = int(cx + w / 2)
                #         y2 = int(cy + h / 2)
                #         cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #         cv2.putText(img_vis, f'{int(cls_id)}', (x1, y1 - 5),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                #
                # # 保存图像
                #
                #
                # cv2.imwrite(os.path.join(save_dir, f'{base_name}_detect.jpg'), img_vis)

                # 分割图可视化（8类 uint8）
                # if seg_label is not None and task == 'seg':
                #     if isinstance(seg_label, torch.Tensor):
                #         seg_label = seg_label.squeeze().cpu().numpy().astype(np.uint8)
                #
                #     def decode_seg_label_to_rgb(seg_label):
                #         """
                #         将 0~7 的整数 label 转回 3位二进制 RGB 可视化图像。
                #         输入：
                #             seg_label: H x W 的 np.uint8 / int 类型
                #         输出：
                #             rgb_img: H x W x 3 的 np.uint8 图像
                #         """
                #         # 将 label 还原为3位二进制
                #         seg_label = seg_label.astype(np.uint8)
                #         r = ((seg_label >> 2) & 1) * 255
                #         g = ((seg_label >> 1) & 1) * 255
                #         b = (seg_label & 1) * 255
                #
                #         rgb_img = np.stack([r, g, b], axis=-1).astype(np.uint8)  # HWC 格式
                #         return rgb_img
                #
                #     seg_vis = decode_seg_label_to_rgb(seg_label)
                #     cv2.imwrite(os.path.join(save_dir, f'{base_name}_seg.jpg'), seg_vis)

                # 深度图可视化（灰度）
                # if lane_label is not None and task == 'depth':
                #     if isinstance(lane_label, torch.Tensor):
                #         depth_map = lane_label.squeeze().cpu().numpy()  # H x W, float32
                #
                #     # 深度图是 [0, 1]，直接映射到 [0, 255]，转换为 uint8 灰度图
                #     depth_gray = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
                #
                #     # 保存为灰度图
                #     cv2.imwrite(os.path.join(save_dir, f'{base_name}_depth.png'), depth_gray)
                # if lane_label is not None and task == 'depth':
                #     if isinstance(lane_label, torch.Tensor):
                #         depth_map = lane_label.squeeze().cpu().numpy()  # H x W, float32
                #
                #     # 深度图：0-1 → 0-255 灰度图
                #     depth_gray = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
                #
                #     # 转换原图（tensor）为 numpy 格式（uint8, HWC）
                #     orig_img = tensor_to_numpy_img(img)  # 你之前写好的函数
                #
                #     # 若原图为彩色 (H,W,3)，灰度图为 (H,W)，需要扩展通道
                #     if len(depth_gray.shape) == 2:
                #         depth_vis = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
                #     else:
                #         depth_vis = depth_gray  # 已经是 BGR 图
                #
                #     # 若大小不一致（可能经过resize），统一为一样高
                #     if orig_img.shape[:2] != depth_vis.shape[:2]:
                #         depth_vis = cv2.resize(depth_vis, (orig_img.shape[1], orig_img.shape[0]))
                #
                #     # 横向拼图
                #     combined = np.hstack((orig_img, depth_vis))
                #
                #     # 保存
                #     cv2.imwrite(os.path.join(save_dir, f'{base_name}_depth_compare.png'), combined)

                img = self.transform(img)
                task = data['tape']

                image_all.append(img)
                path_all.append(path)
                if i == 0:
                    target_all.append(target[i])
                elif i == 1:
                    target_all.append(target[i])
                else:
                    target_all.append(target[i])
                shape_all.append(shapes)
                task_all.append(task)
                # 1

            return image_all, target_all, path_all, shape_all, task_all

        else:
            data = data1
            task = data['tape']
            img, labels, seg_label, lane_label, (h0, w0), (h,w), path = self.load_image(idx,task)

            (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), 640, auto=True, scaleup=self.is_train)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels_out = None
            if data['tape']== 'detect':
                if len(labels):
                    # update labels after letterbox
                    labels[:, 1] = ratio[0] * labels[:, 1] + pad[0]
                    labels[:, 2] = ratio[1] * labels[:, 2] + pad[1]
                    labels[:, 3] = ratio[0] * labels[:, 3] + pad[0]
                    labels[:, 4] = ratio[1] * labels[:, 4] + pad[1]

                    # convert xyxy to ( cx, cy, w, h )
                    labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                labels_out = torch.zeros((len(labels), 5))
                if len(labels):
                    labels_out[:, :] = torch.from_numpy(labels)

            img = np.ascontiguousarray(img)

            if data['tape'] == 'depth':
                if isinstance(lane_label, np.ndarray):
                    lane_label = self.Tensor(lane_label)
                # 如果深度图是 [1, H, W]，可以保持，或者 squeeze 掉 1 个通道变成 [H, W]
                if lane_label.ndim == 2:
                    lane_label = lane_label.unsqueeze(0)

                # lane_label 就是单通道深度图
            target = [labels_out, seg_label, lane_label]

            def tensor_to_numpy_img(tensor_img):
                """
                支持将图像（Tensor或ndarray）转为可视化格式的uint8 HWC图像。
                如果是float，会反标准化和放缩；如果是uint8则直接返回。
                """
                if isinstance(tensor_img, torch.Tensor):
                    img = tensor_img.clone().detach().cpu().numpy()
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)  # C,H,W → H,W,C
                    elif img.ndim == 3 and img.shape[2] == 3:
                        pass  # 已是 HWC
                    else:
                        raise ValueError("图像维度异常，无法转换为OpenCV格式")

                    # 反标准化（如果做过 ImageNet 标准化）
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)

                elif isinstance(tensor_img, np.ndarray):
                    if tensor_img.dtype != np.uint8:
                        img = (tensor_img * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img = tensor_img
                else:
                    raise TypeError("不支持的图像类型")

                return np.ascontiguousarray(img)



            img_vis = tensor_to_numpy_img(img)
            save_dir = 'D:/Multi-task/Git/Multi-Task/inference/image'
            os.makedirs(save_dir, exist_ok=True)
            base_name = Path(path).stem
            # if labels_out is not None and len(labels_out):
            #     for label in labels_out:
            #         cls_id, cx, cy, w, h = label.tolist()
            #         x1 = int(cx - w / 2)
            #         y1 = int(cy - h / 2)
            #         x2 = int(cx + w / 2)
            #         y2 = int(cy + h / 2)
            #         cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         cv2.putText(img_vis, f'{int(cls_id)}', (x1, y1 - 5),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #
            # # 保存图像
            #
            #
            # cv2.imwrite(os.path.join(save_dir, f'{base_name}_detect.jpg'), img_vis)

            # 分割图可视化（8类 uint8）
            # if seg_label is not None and task == 'seg':
            #     if isinstance(seg_label, torch.Tensor):
            #         seg_label = seg_label.squeeze().cpu().numpy().astype(np.uint8)
            #
            #     def decode_seg_label_to_rgb(seg_label):
            #         """
            #         将 0~7 的整数 label 转回 3位二进制 RGB 可视化图像。
            #         输入：
            #             seg_label: H x W 的 np.uint8 / int 类型
            #         输出：
            #             rgb_img: H x W x 3 的 np.uint8 图像
            #         """
            #         # 将 label 还原为3位二进制
            #         seg_label = seg_label.astype(np.uint8)
            #         r = ((seg_label >> 2) & 1) * 255
            #         g = ((seg_label >> 1) & 1) * 255
            #         b = (seg_label & 1) * 255
            #
            #         rgb_img = np.stack([r, g, b], axis=-1).astype(np.uint8)  # HWC 格式
            #         return rgb_img
            #
            #     seg_vis = decode_seg_label_to_rgb(seg_label)
            #     cv2.imwrite(os.path.join(save_dir, f'{base_name}_seg.jpg'), seg_vis)

            # 深度图可视化（灰度）
            # if lane_label is not None and task == 'depth':
            #     if isinstance(lane_label, torch.Tensor):
            #         depth_map = lane_label.squeeze().cpu().numpy()  # H x W, float32
            #
            #     # 深度图是 [0, 1]，直接映射到 [0, 255]，转换为 uint8 灰度图
            #     depth_gray = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
            #
            #     # 保存为灰度图
            #     cv2.imwrite(os.path.join(save_dir, f'{base_name}_depth.png'), depth_gray)
            # if lane_label is not None and task == 'depth':
            #     if isinstance(lane_label, torch.Tensor):
            #         depth_map = lane_label.squeeze().cpu().numpy()  # H x W, float32
            #
            #     # 深度图：0-1 → 0-255 灰度图
            #     depth_gray = (np.clip(depth_map, 0, 1) * 255).astype(np.uint8)
            #
            #     # 转换原图（tensor）为 numpy 格式（uint8, HWC）
            #     orig_img = tensor_to_numpy_img(img)  # 你之前写好的函数
            #
            #     # 若原图为彩色 (H,W,3)，灰度图为 (H,W)，需要扩展通道
            #     if len(depth_gray.shape) == 2:
            #         depth_vis = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
            #     else:
            #         depth_vis = depth_gray  # 已经是 BGR 图
            #
            #     # 若大小不一致（可能经过resize），统一为一样高
            #     if orig_img.shape[:2] != depth_vis.shape[:2]:
            #         depth_vis = cv2.resize(depth_vis, (orig_img.shape[1], orig_img.shape[0]))
            #
            #     # 横向拼图
            #     combined = np.hstack((orig_img, depth_vis))
            #
            #     # 保存
            #     cv2.imwrite(os.path.join(save_dir, f'{base_name}_depth_compare.png'), combined)

            img = self.transform(img)
            task =data['tape']
            return img, target, path, shapes,task

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected


    # def collate_fn(batch):
    #     img, label, paths, shapes= zip(*batch)
    #     label_det, label_seg, label_lane = [], [], []
    #     for i, l in enumerate(label):
    #         l_det, l_seg, l_lane = l
    #         # l_det[:, 0] = i  # add target image index for build_targets()
    #         label_det.append(l_det)
    #         label_seg.append(l_seg)
    #         label_lane.append(l_lane)
    #
    #     label_det = pad_sequence(label_det, batch_first = True, padding_value = 0)
    #
    #     return torch.stack(img, 0), [label_det, torch.stack(label_seg, 0), torch.stack(label_lane, 0)], paths, shapes

    from torch.nn.utils.rnn import pad_sequence
    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes,task = zip(*batch)
        print("coffate_fn 批次打包",task[0])

        label_det, label_seg, label_depth = [], [], []
        has_det, has_seg, has_lane = False, False, False

        for i, l in enumerate(label):
            l_det, l_seg, l_depth = l

            if l_det is not None:
                label_det.append(l_det)
                has_det = True
            else:
                label_det.append(None)  # 占位

            if l_seg is not None:
                label_seg.append(l_seg)
                has_seg = True
            else:
                # 填充全0张图（C, H, W），你可以自定义 H, W，比如 160x320
                #label_seg.append(torch.zeros((2, 160, 320)))  # 假设2类语义分割
                label_seg.append(None)

            if l_depth is not None:
                label_depth.append(l_depth)
                has_lane = True
            else:
                #label_lane.append(torch.zeros((2, 160, 320)))
                label_depth.append(None)# 假设2类车道线分割

            # if not label_det and not label_seg and not label_depth:
            #     print("读取错误===============", l_det, l_seg, l_depth)
        if isinstance(img, tuple) and isinstance(img[0], list) and isinstance(img[0][0], torch.Tensor):
            img1, img2, img3 = zip(*img)
            #img = img.permute(1, 0, 2, 3, 4)
            # img1 = img[0]  # shape: (4, 3, 380, 640)
            # img2 = img[1]  # shape: (4, 3, 380, 640)
            # img3 = img[2]  # shape: (4, 3, 380, 640)
            img1 = torch.stack(img1, 0)
            img2 = torch.stack(img2, 0)
            img3 = torch.stack(img3, 0)
            img = [img1,img2,img3]
            label_det = pad_sequence(label_det, batch_first=True, padding_value=0)
            label_seg = [torch.from_numpy(arr).long() for arr in label_seg]
            label_seg = torch.stack(label_seg, 0)
            label_depth = torch.stack(label_depth, 0)

        else:
            task = task[0]
            img = torch.stack(img,0)
            task = task[0] if isinstance(task, list) else task
            if task == 'detect':
                label_det = pad_sequence(label_det, batch_first=True, padding_value=0)
            else:
                label_det = None

            if  task == 'seg' :
                label_seg = [torch.from_numpy(arr).long() for arr in label_seg]

                label_seg = torch.stack(label_seg, 0)
            else:
                label_seg = None

            if  task == 'depth' :

                label_depth = torch.stack(label_depth, 0)
            else:
                label_depth = None
            if (label_det is None or label_det.numel() == 0) and \
                    (label_seg is None or label_seg.numel() == 0) and \
                    (label_depth is None or label_depth.numel() == 0):
                print("读取错误===============", label_det, label_seg, label_depth)

        #return img, [label_det, label_seg, label_lane], paths, shapes ,task

        return img, [label_det, label_seg, label_depth], paths, shapes, task



