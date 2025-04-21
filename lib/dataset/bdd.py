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

    def _get_db(self):
        print('building database...')
        print('bdd进入')
        gt_db = []
        height, width = self.shapes
        det_db, seg_db, depth_db = [], [], []

        # ===================== 1. detection dataset =====================
        if hasattr(self, "det_list"):

            for label_path in tqdm(self.det_list, desc="Loading detection"):
                print(label_path)
                image_path = str(label_path).replace(str(self.label_root), str(self.img_root1)).replace(".json", ".jpg")
                mask_path = None
                lane_path = None

                with open(label_path, 'r') as f:
                    label = json.load(f)

                data = label['frames'][0]['objects']
                data = self.filter_data(data)
                gt = np.zeros((len(data), 5))
                for idx, obj in enumerate(data):
                    category = obj['category']
                    if category == "traffic light":
                        color = obj['attributes']['trafficLightColor']
                        category = "tl_" + color
                    if category in id_dict.keys():
                        x1 = float(obj['box2d']['x1'])
                        y1 = float(obj['box2d']['y1'])
                        x2 = float(obj['box2d']['x2'])
                        y2 = float(obj['box2d']['y2'])
                        cls_id = id_dict[category]
                        if single_cls:
                            cls_id = 0
                        gt[idx][0] = cls_id
                        box = convert((width, height), (x1, x2, y1, y2))
                        gt[idx][1:] = list(box)

                rec = [{
                    'image': image_path,
                    'label': gt,
                    'mask': mask_path,
                    'lane': lane_path,
                    'tape': 'detect'
                }]
                det_db += rec
                #gt_db += rec

        # ===================== 2. segmentation dataset =====================
        if hasattr(self, "seg_list"):
            for mask_path in tqdm(self.seg_list, desc="Loading segmentation"):
                image_path = str(mask_path).replace(str(self.mask_root), str(self.img_root2)).replace(".png", ".jpg")
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
                image_path = str(lane_path).replace(str(self.lane_root), str(self.img_root3)).replace(".png", ".jpg")
                rec = [{
                    'image': image_path,
                    'label': None,
                    'mask': None,
                    'lane': str(lane_path),
                    'tape': 'depth'
                }]
                depth_db+= rec


        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        max_len = max(len(det_db), len(seg_db), len(depth_db))



        for i in range(0, max_len, batch_size):
            if i  < len(det_db):
                gt_db += det_db[i:i + batch_size]
            if i  < len(seg_db):
                gt_db += seg_db[i:i + batch_size]
            if i  < len(depth_db):
                gt_db += depth_db[i:i + batch_size]

        print('database build finish')
        return gt_db

    # def _get_db(self):
    #     """
    #     get database from the annotation file
    #
    #     Inputs:
    #
    #     Returns:
    #     gt_db: (list)database   [a,b,c,...]
    #             a: (dictionary){'image':, 'information':, ......}
    #     image: image path
    #     mask: path of the segmetation label
    #     label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
    #     """
    #     print('building database...')
    #     gt_db = []
    #     height, width = self.shapes
    #     # for mask in tqdm(list(self.mask_list)[0:200] if self.is_train==True else list(self.mask_list)[0:30]):
    #     # for mask in tqdm(list(self.mask_list)[0:20000] if self.is_train==True else list(self.mask_list)[0:3000]):
    #
    #
    #     for mask in tqdm(list(self.mask_list)):
    #         mask_path = str(mask)
    #         label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
    #         image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
    #         lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
    #         print(mask_path,label_path,image_path,lane_path )
    #         with open(label_path, 'r') as f:
    #             label = json.load(f)
    #
    #         #获取第一个 frame 的所有 objects（因为每个 json 通常只标注一个 frame）。
    #         data = label['frames'][0]['objects']
    #         data = self.filter_data(data)
    #         gt = np.zeros((len(data), 5))
    #         for idx, obj in enumerate(data):
    #
    #             category = obj['category']
    #             if category == "traffic light":
    #                 color = obj['attributes']['trafficLightColor']
    #                 category = "tl_" + color
    #             if category in id_dict.keys():
    #                 x1 = float(obj['box2d']['x1'])
    #                 y1 = float(obj['box2d']['y1'])
    #                 x2 = float(obj['box2d']['x2'])
    #                 y2 = float(obj['box2d']['y2'])
    #                 cls_id = id_dict[category]
    #                 if single_cls:
    #                      cls_id=0
    #                 gt[idx][0] = cls_id
    #                 box = convert((width, height), (x1, x2, y1, y2))
    #                 gt[idx][1:] = list(box)
    #
    #
    #         rec = [{
    #             'image': image_path,
    #             'label': gt,
    #             'mask': mask_path,
    #             'lane': lane_path
    #         }]
    #
    #         gt_db += rec
    #     print('database build finish')
    #     return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
