import torch
from torch import tensor
import torch.nn as nn

from torch.nn import Conv2d

import sys,os
import math
import sys
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
# from lib.models.common import Conv, SPPF, Focus, Concat, Detect, MergeBlock, C3
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.common import Conv, seg_head, PSA_p, MergeBlock, FPN_C2C12C131619, ResidualBlockSequence_DEPTH
from lib.models.common import Concat, FPN_C2, FPN_C3, FPN_C4, ELANNet, ELANBlock_Head, PaFPNELAN, IDetect, RepConv,ResidualBlockSequence_DET,ResidualBlock
# from lib.models.YOLOX_Head_scales import YOLOXHead
from lib.models.YOLOX_Head_scales_noshare import YOLOXHead
from bts.bts import bts


# 修改
# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
###### prediction head index
# [2, 16, 28],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx no_use c2
[3, 18, 21],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx use_c2

###### Backbone
[ -1, ELANNet, [True]],   #0

###### PaFPNELAN
[ -1, PaFPNELAN, []],   #1

[ -1, ResidualBlockSequence_DET ,[[128,256,512]]], #2


###### Detect Head
[ -1, YOLOXHead,  [4]], #3 #Detection head

# ###### 渐进式上采样
[ 1, FPN_C3, []],   #4
[ 1, FPN_C4, []],   #5

# segmentation head
[ -1, ResidualBlock ,[512,512]], #6
[ -1, Conv, [512, 256, 3, 1]],   #7
[ -1, Upsample, [None, 2, 'bilinear']],  #8
[ -1, ELANBlock_Head, [256, 128]], #9
[ -1, Conv, [128, 64, 3, 1]],  #10
[ -1, Upsample, [None, 2, 'bilinear']], #11
[ -1, Conv, [64, 32, 3, 1]],  #12
[ -1, Upsample, [None, 2, 'bilinear']], #13
[ -1, Conv, [32, 16, 3, 1]],  #14
[ -1, ELANBlock_Head, [16, 16]], #15
[ -1, Upsample, [None, 2, 'bilinear']],  #16
[ -1, Conv, [16, 8, 3, 1]], #17
[ -1, seg_head, ['softmax']],  #18 segmentation head

# # no use C2
###########
# [ 3, Conv, [256, 128, 3, 1]],    #17
# [ -1, Upsample, [None, 2, 'bilinear']],
# [ -1, ELANBlock_Head, [128, 64]],
# [ -1, PSA_p, [64, 64]],
# [ -1, Conv, [64, 32, 3, 1]],
# [ -1, Upsample, [None, 2, 'bilinear']],
# [ -1, Conv, [32, 16, 3, 1]],
# [ -1, ELANBlock_Head, [16, 8]],
# [ -1, PSA_p, [8, 8]],
# [ -1, Upsample, [None, 2, 'bilinear']],
# [ -1, Conv, [8, 2, 3, 1]],
# [ -1, seg_head, ['sigmoid']],  #28 segmentation head
###########

# use C2
###########
[1 ,FPN_C2C12C131619,[]], #19
[ -1, ResidualBlockSequence_DEPTH ,[[64,256,256,256,512]]],
[-1 , bts , [[64,256,256,256,512]] ]    #21
# [ 1, FPN_C2, []],  #17
# [ -1, Conv, [256, 128, 3, 1]],    #18
# # sum c2 and p3
# [ 3, Conv, [256, 128, 3, 1]],
# [ -1, Upsample, [None, 2, 'bilinear']],
# [ [-1, 18], MergeBlock, ["add"]],    #C2 and P3
# [ -1, ELANBlock_Head, [128, 64]],
# [ -1, PSA_p, [64, 64]],
# [ -1, Conv, [64, 32, 3, 1]],
# [ -1, Upsample, [None, 2, 'bilinear']],
# [ -1, Conv, [32, 16, 3, 1]],
# [ -1, ELANBlock_Head, [16, 8]],
# [ -1, PSA_p, [8, 8]],
# [ -1, Upsample, [None, 2, 'bilinear']],
# [ -1, Conv, [8, 2, 3, 1]], #
# [ -1, seg_head, ['sigmoid']],  #31 segmentation head

# # use C2
# ###########
# [ 1, FPN_C2, []],  #17
# [ -1, Conv, [256, 128, 3, 1]],    #18
# [ -1, Conv, [128, 64, 3, 1]],    #19
# [ 3, Conv, [256, 128, 3, 1]],  
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [128, 64, 3, 1]],  
# [ [-1, 19], MergeBlock, ["cat"]],    #concat C2 and P3

# [ -1, ELANBlock_Head, [128, 64]], 
# [ -1, PSA_p, [64, 64]], 
# [ -1, Conv, [64, 32, 3, 1]], 
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [32, 16, 3, 1]], 
# [ -1, ELANBlock_Head, [16, 8]], 
# [ -1, PSA_p, [8, 8]], 
# [ -1, Upsample, [None, 2, 'bilinear']], 
# [ -1, Conv, [8, 2, 3, 1]], #
# [ -1, seg_head, ['sigmoid']],  #33 segmentation head
# ###########

]

# 修改
class MCnet(nn.Module):
    # block_cfg = YOLOP-list
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        # 27 
        self.det_out_idx = block_cfg[0][0]


        # 63 67
        #self.seg_out_idx = block_cfg[0][1:]
        self.seg_out_idx = block_cfg[0][1]

        self.depth_out_idx = block_cfg[0][2]

        self.det_idx = block_cfg[0][0]

        # Build model
        # e.g. [ -1, Focus, [3, 32, 3]],   #0
        # i从0开始编号，from_ = -1，block = Focus，args = [3, 32, 3]
        # 注意，block是类，不是str
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            #print(i,(from_, block, args))
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is YOLOXHead:
                # detector_index  # 27
                self.detector_index = i

            # *args,参数解码 [3, 32, 3] -> 3, 32, 3
            # 构建一系列模块，实例化block
            print(i)
            block_ = block(*args)

            # 模块索引，模块输入来源索引          
            block_.index, block_.from_ = i, from_

            # 向layers_list中添加block_
            layers.append(block_)

            # [ 6, 4, 14, 10, 23, 17, 20, 23, 25, 26, 26, 25, 23, 20, 17, 2, 37, 45, 51, 55, 57, 58, 59, 59 ]
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
            #print("save",save)
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        #print(self.names)

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, YOLOXHead):
            s = 512  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s),task=['detect'])

            self.stride = Detector.strides
            Detector.initialize_biases(1e-2)

        initialize_weights(self)


    def forward(self, x, task):
        # print(task)
        # if len(task) >1 :
        #     task = task[0]
        if torch.all(x == 0):
            print("intput =========0,forward")

        cache = []
        #out = []
        det_out = None
        seg_out = None
        depth_out = None

        # 定义各分支的范围
        backbone_neck_idx = [0, 1]
        det_idx = [2,3]
        da_seg_idx = list(range(4, 19))
        ll_seg_idx = list(range(19, 22))

        # 决定本次 forward 要执行哪些层
        task = task[0] if isinstance(task, list)  else task
        print("1111111111111",task)

        if task == 'detect':
            run_idx = backbone_neck_idx + det_idx
        elif task == 'seg':
            run_idx = backbone_neck_idx + da_seg_idx
        elif task == 'depth':
            run_idx = backbone_neck_idx + ll_seg_idx
        else:
            raise ValueError(f"Unknown task: {task}")
        print(task)
        for i, block in enumerate(self.model):
            if i not in run_idx:
                cache.append(None)
                continue

            # 构造输入
            if block.from_ != -1:
                x_in = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in
                                                                                block.from_]
            else:
                x_in = x


            x = block(x_in)

            # 保存输出结果
            if i == 3 and task == 'detect':
                det_out = x
            if i == 18 and task == 'seg':
                seg_out = x
            if i == 21 and task == 'depth':
                depth_out = x

            cache.append(x if block.index in self.save else None)

        #out.insert(0, det_out)
        return [det_out,seg_out,depth_out]

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            # elif isinstance(m, RepConv_OREPA):
            #     #print(f" switch_to_deploy")
            #     m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, IDetect):
                m.fuse()
                m.forward = m.fuseforward
        # self.info()
        return self


# class FullModel(nn.Module):
#     def __init__(self, unit_cfg, multitask_model):
#         super().__init__()
#         self.unit_module = unit_module()  # 你的 UnitModule 配置
#         self.multi_task_model = multitask_model    # 原有多任务模型
#
#     def forward(self, x, targets=None):
#         # 前向走 UnitModule（启用训练分支）
#         x = self.unit_module(x, training=True)[0]  # 输出去噪图像
#
#         # 前向走多任务主模型
#         if self.training:
#             return self.multi_task_model(x, targets)
#         else:
#             return self.multi_task_model(x)


def get_net(cfg, **kwargs):
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

if __name__ == "__main__":
#    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    # model_out,SAD_out = model(input_)
    # detects, dring_area_seg, lane_line_seg = model_out
    # Da_fmap, LL_fmap = SAD_out
    detects, dring_area_seg, lane_line_seg = model(input_,task=['seg'])
    #Da_fmap, LL_fmap = SAD_out
    # for det in detects:
    #     print(det.shape)
    print(dring_area_seg.shape)
    # print(lane_line_seg.shape)
 
