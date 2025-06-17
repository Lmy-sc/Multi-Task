import os

import numpy as np
import json
from lib.config import cfg
from lib.config import update_config
from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single
from tqdm import tqdm

single_cls = True       # just detect vehicle

class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg
        self.is_train = is_train

    # def _get_db(self):
    #     print('building database...')
    #     print('bdd进入')
    #     gt_db = []
    #     height, width = self.shapes
    #     det_db, seg_db, depth_db = [], [], []

        # ===================== 1. detection dataset =====================
        # if hasattr(self, "det_list"):
        #     mask_path=None
        #     lane_path=None
        #
        #     coco_json_path =self.det_list[0]
        #     print("detlist=========",self.det_list)
        #     print(coco_json_path)
        #
        #     with open(coco_json_path, 'r') as f:
        #         coco = json.load(f)
        #
        #     # COCO格式索引构建
        #     images = {img['id']: img for img in coco['images']}
        #     ann_by_image = {}
        #     for ann in coco['annotations']:
        #         img_id = ann['image_id']
        #         if img_id not in ann_by_image:
        #             ann_by_image[img_id] = []
        #         ann_by_image[img_id].append(ann)
        #
        #     # 类别映射表（如果没有提供就默认COCO的原始映射）
        #     if id_dict is None:
        #         id_map = {cat['id']: cat['id'] for cat in coco['categories']}
        #
        #     # 构建rec列表
        #     rec_list = []
        #
        #     for img_id, img_info in tqdm(images.items(), desc="Parsing COCO annotations"):
        #         file_name = img_info['file_name']
        #         width = img_info['width']
        #         height = img_info['height']
        #         image_path = os.path.join(self.img_root1, file_name)
        #
        #         anns = ann_by_image.get(img_id, [])
        #         gt = np.zeros((len(anns), 5))
        #
        #         for idx, ann in enumerate(anns):
        #             coco_cat_id = ann['category_id']
        #             cls_id = id_map.get(coco_cat_id, None)
        #             if cls_id is None:
        #                 continue
        #             if single_cls:
        #                 cls_id = 0
        #             x, y, w, h = ann['bbox']
        #             cx = (x + x + w) / 2 / width
        #             cy = (y + y + h) / 2 / height
        #             bw = w / width
        #             bh = h / height
        #             gt[idx][0] = cls_id
        #             gt[idx][1:] = [cx, cy, bw, bh]
        #
        #     # for label_path in tqdm(self.det_list, desc="Loading detection"):
        #     #     print(label_path)
        #     #     image_path = str(label_path).replace(str(self.label_root), str(self.img_root1)).replace(".json", ".jpg")
        #     #     mask_path = None
        #     #     lane_path = None
        #     #
        #     #     with open(label_path, 'r') as f:
        #     #         label = json.load(f)
        #     #
        #     #     data = label['frames'][0]['objects']
        #     #     data = self.filter_data(data)
        #     #     gt = np.zeros((len(data), 5))
        #     #     for idx, obj in enumerate(data):
        #     #         category = obj['category']
        #     #         if category == "traffic light":
        #     #             color = obj['attributes']['trafficLightColor']
        #     #             category = "tl_" + color
        #     #         if category in id_dict.keys():
        #     #             x1 = float(obj['box2d']['x1'])
        #     #             y1 = float(obj['box2d']['y1'])
        #     #             x2 = float(obj['box2d']['x2'])
        #     #             y2 = float(obj['box2d']['y2'])
        #     #             cls_id = id_dict[category]
        #     #             if single_cls:
        #     #                 cls_id = 0
        #     #             gt[idx][0] = cls_id
        #     #             box = convert((width, height), (x1, x2, y1, y2))
        #     #             gt[idx][1:] = list(box)
        #
        #
        #             rec = [{
        #                 'image': image_path,
        #                 'label': gt,
        #                 'mask': mask_path,
        #                 'lane': lane_path,
        #                 'tape': 'detect'
        #             }]
        #             det_db += rec
        #         #gt_db += rec
        #
        # # ===================== 2. segmentation dataset =====================
        # if hasattr(self, "seg_list"):
        #     for mask_path in tqdm(self.seg_list, desc="Loading segmentation"):
        #         image_path = str(mask_path).replace(str(self.mask_root), str(self.img_root2)).replace(".bmp", ".jpg")
        #         rec = [{
        #             'image': image_path,
        #             'label': None,
        #             'mask': str(mask_path),
        #             'lane': None,
        #             'tape': 'seg'
        #         }]
        #         seg_db+= rec
        #
        # # ===================== 3. lane line dataset =====================
        # if hasattr(self, "lane_list"):
        #     for lane_path in tqdm(self.lane_list, desc="Loading lane line"):
        #         image_path = str(lane_path).replace(str(self.lane_root), str(self.img_root3)).replace(".png", ".jpg")
        #         rec = [{
        #             'image': image_path,
        #             'label': None,
        #             'mask': None,
        #             'lane': str(lane_path),
        #             'tape': 'depth'
        #         }]
        #         depth_db+= rec
        #
        #
        # batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        # max_len = max(len(det_db), len(seg_db), len(depth_db))
        #
        #
        #
        # for i in range(0, max_len, batch_size):
        #     if i  < len(det_db):
        #         gt_db += det_db[i:i + batch_size]
        #     if i  < len(seg_db):
        #         gt_db += seg_db[i:i + batch_size]
        #     if i  < len(depth_db):
        #         gt_db += depth_db[i:i + batch_size]
        #
        # print('database build finish')
        # return gt_db
    def _get_db(self):
        print('building database...')
        gt_db = []
        height, width = self.shapes
        det_db, seg_db, depth_db = [], [], []
        det_num,seg_num,depth_num =[],[],[]

        # ===================== 1. detection dataset =====================
        if hasattr(self, "det_list"):

            for label_path in tqdm(self.det_list, desc="Loading detection"):
                print(label_path)
                image_path = str(label_path).replace(str(self.label_root), str(self.img_root1)).replace(".txt", ".jpg")
                mask_path = None
                lane_path = None

                with open(label_path, 'r') as f:
                    lines = f.readlines()

                gt = np.zeros((len(lines), 5))  # 每行为一个目标，共5列
                for idx, line in enumerate(lines):
                    items = line.strip().split()
                    cls_id = int(items[0])
                    cx = float(items[1])
                    cy = float(items[2])
                    w = float(items[3])
                    h = float(items[4])

                    gt[idx][0] = cls_id
                    gt[idx][1:] = [cx, cy, w, h]

                rec = [{
                        'image': image_path,
                        'label': gt,
                        'mask': mask_path,
                        'lane': lane_path,
                        'tape': 'detect'
                }]
                det_db += rec
                #gt_db += rec

#        ===================== 2. segmentation dataset =====================
        if hasattr(self, "seg_list"):
            for mask_path in tqdm(self.seg_list, desc="Loading segmentation"):
                image_path = str(mask_path).replace(str(self.mask_root), str(self.img_root2)).replace(".bmp", ".jpg")
                rec = [{
                    'image': image_path,
                    'label': None,
                    'mask': str(mask_path),
                    'lane': None,
                    'tape': 'seg'
                }]
                seg_db+= rec

        # ===================== 3. lane line dataset =====================
        if hasattr(self, "lane_list"):
            for lane_path in tqdm(self.lane_list, desc="Loading lane line"):
                image_path = str(lane_path).replace(str(self.lane_root), str(self.img_root3)).replace(".png", ".png")
                rec = [{
                    'image': image_path,
                    'label': None,
                    'mask': None,
                    'lane': str(lane_path),
                    'tape': 'depth'
                }]
                depth_db+= rec


        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU

        # depth_db = self.align_db(depth_db, batch_size)
        # len1 = int(max(len(det_db), len(seg_db), len(depth_db)))
        # for i in tqdm(range(0, len1, batch_size), desc="Building gt_db"):
        #     if i < len(depth_db):
        #         gt_db += depth_db[i:i + batch_size]


        # det_db = self.align_db(det_db, batch_size)
        # seg_db = self.align_db(seg_db, batch_size)
        # depth_db = self.align_db(depth_db, batch_size)
        #
        # #max_len = max(len(det_db), len(seg_db), len(depth_db))
        #
        # #self.cfg.Train.Batchnum=min(len(det_db), len(seg_db), len(depth_db))
        # len1 = min(len(det_db), len(seg_db), len(depth_db))

        max_len = max(len(det_db), len(seg_db), len(depth_db))
        if self.is_train is True :
            def repeat_to_align(db, max_len):
                if len(db) == 0:
                    return db
                repeat_factor = (max_len + len(db) - 1) // len(db)  # 向上取整
                return (db * repeat_factor)[:max_len]  # 重复并截断
            det_db = repeat_to_align(det_db, max_len)
            seg_db = repeat_to_align(seg_db, max_len)
            depth_db = repeat_to_align(depth_db, max_len)

        det_db = self.align_db(det_db, batch_size)
        seg_db = self.align_db(seg_db, batch_size)
        depth_db = self.align_db(depth_db, batch_size)

        max_len = max(len(det_db), len(seg_db), len(depth_db))

        if self.is_train:
            for i in tqdm(range(0, max_len), desc="Building gt_db"):
                image_path1 = det_db[i]['image']
                label       = det_db[i]['label']
                image_path2 = seg_db[i]['image']
                seg         = seg_db[i]['mask']
                image_path3 = depth_db[i]['image']
                depth       = depth_db[i]['lane']
                rec = [{
                        'image1': image_path1,
                        'image2': image_path2,
                        'image3': image_path3,
                        'label': label,
                        'mask': seg,
                        'lane': depth,
                        'tape': None
                     }]
                gt_db += rec
        else :
            # for i in range(0, len1, batch_size):
            #     if i  < len(det_db):
            #         gt_db += det_db[i:i + batch_size]
            #     if i  < len(seg_db):
            #         gt_db += seg_db[i:i + batch_size]
            #     if i  < len(depth_db):
            #         gt_db += depth_db[i:i + batch_size]


        # for i in tqdm(range(0, len1, batch_size), desc="Building gt_db"):
        #     if i < len(det_db):
        #         gt_db += det_db[i:i + batch_size]
        #         det_num.append(i)
        #     if i < len(seg_db):
        #         gt_db += seg_db[i:i + batch_size]
        #         seg_num.append(i)
        #     if i < len(depth_db):
        #         gt_db += depth_db[i:i + batch_size]
        #         depth_num.append(i)
            for i in tqdm(range(0, max_len, batch_size), desc="Building gt_db"):
                if i < len(det_db):
                    batch = det_db[i:i + batch_size]
                    det_num.extend(range(len(gt_db), len(gt_db) + len(batch)))
                    gt_db += batch

                if i < len(seg_db):
                    batch = seg_db[i:i + batch_size]
                    seg_num.extend(range(len(gt_db), len(gt_db) + len(batch)))
                    gt_db += batch

                if i < len(depth_db):
                    batch = depth_db[i:i + batch_size]
                    depth_num.extend(range(len(gt_db), len(gt_db) + len(batch)))
                    gt_db += batch

        # for i in tqdm(range(0, max_len, batch_size), desc="Building gt_db"):
        #     det_batch = det_db[i:i + batch_size]
        #     seg_batch = seg_db[i:i + batch_size]
        #     depth_batch = depth_db[i:i + batch_size]
        #
        #     det_num.extend(range(len(gt_db), len(gt_db) + len(det_batch)))
        #     gt_db += det_batch
        #
        #     seg_num.extend(range(len(gt_db), len(gt_db) + len(seg_batch)))
        #     gt_db += seg_batch
        #
        #     depth_num.extend(range(len(gt_db), len(gt_db) + len(depth_batch)))
        #     gt_db += depth_batch

        # for i in range(0, len1, batch_size):
        #     if i  < len(det_db):
        #         gt_db += det_db[i:i + batch_size]
        #     if i  < len(seg_db):
        #         gt_db += seg_db[i:i + batch_size]
        #     if i  < len(depth_db):
        #         gt_db += depth_db[i:i + batch_size]

        print('database build finish')
        # return gt_db,det_db,seg_db,depth_db
        #return gt_db,det_num,seg_num,depth_num

        return gt_db


    def align_db(self,db, batch_size):
        n = len(db)
        aligned_n = (n // batch_size) * batch_size  # 向下取整
        return db[:aligned_n]

    # def filter_data(self, data):
    #     remain = []
    #     for obj in data:
    #         if 'box2d' in obj.keys():  # obj.has_key('box2d'):
    #             if single_cls:
    #                 if obj['category'] in id_dict_single.keys():
    #                     remain.append(obj)
    #             else:
    #                 remain.append(obj)
    #     return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
