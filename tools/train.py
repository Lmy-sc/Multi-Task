import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor
from unitmodule.models.data_preprocessors import unit_module
from mmengine.config import Config, DictAction
import argparse
import logging
import os
import os.path as osp
import sys
import os
from mmengine.registry import MODELS

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank
    #print(rank)
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU* len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
    print("load model to device")
    model = get_net(cfg).to(device)

    cfg1 = Config.fromfile('D:\Multi-task\Git\Multi-Task\lib\config//unitmodule//unitmodule.py')
    unit_cfg = cfg1.model
    model1 = MODELS.build(unit_cfg).to(device)
    print("unitmodule=======",model1)

    criterion = get_loss(cfg, device, model)
    params = list(model1.parameters()) + list(model.parameters())
    #params = list(model.parameters())
    optimizer = get_optimizer(cfg, params)

    #optimizer = get_optimizer(cfg, model)

    # load checkpoint model
    best_perf = 0.0
    best_model = False
    last_epoch = -1
    #加载完整预训练模型 只加载检测分支权重 从中断点自动恢复训练（AUTO_RESUME）

    #是为了冻结不同的头部子网络
    Encoder_para_idx = [str(i) for i in range(0, 2)]
    Det_Head_para_idx = [str(i) for i in range(2, 3)]
    Da_Head_para_idx = [str(i) for i in range(3, 17)]
    Ll_Head_para_idx = [str(i) for i in range(17, 32)]

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        #加载完整模型（包括三任务分支）
        def init_weights(m):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        model1.apply(init_weights)

        # ---------- 加载 model 的部分预训练权重 ----------
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)

            begin_epoch = 1
            last_epoch = checkpoint.get('epoch', 0)

            # 加载 model 的部分参数
            state_dict = checkpoint['state_dict']
            for k, v in state_dict.items():
                print(f'{k}: {v.shape}')
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['state_dict'], strict=False
            )
            if missing_keys:
                logger.warning(f"=> model: Missing keys not loaded: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"=> model: Unexpected keys ignored: {unexpected_keys}")

            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    logger.info("=> optimizer state loaded")
                except Exception as e:
                    logger.warning(f"=> failed to load optimizer state: {e}")

            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint.get('epoch', 'unknown')))
        else:
            logger.warning(f"=> pretrained model not found: {cfg.MODEL.PRETRAINED}")
            begin_epoch = 0
            last_epoch = 0
        # if os.path.exists(cfg.MODEL.PRETRAINED):
        #     logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED1))
        #     logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
        #     checkpoint1 = torch.load(cfg.MODEL.PRETRAINED1)
        #     checkpoint = torch.load(cfg.MODEL.PRETRAINED)
        #
        #     #begin_epoch = checkpoint['epoch']
        #     begin_epoch = 60
        #     # best_perf = checkpoint['perf']
        #     last_epoch = checkpoint['epoch']
        #     model1.load_state_dict(checkpoint1['state_dict'])
        #     model.load_state_dict(checkpoint['state_dict'])
        #
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     logger.info("=> loaded checkpoint '{}' (epoch {})".format(
        #         cfg.MODEL.PRETRAINED, checkpoint['epoch']))
        #     #cfg.NEED_AUTOANCHOR = False     #disable autoanchor

        #仅加载检测分支参数（state_dict 中挑选第 0~24 层）
        if os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))
            det_idx_range = [str(i) for i in range(0,25)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))

        #是否开启 AUTO_RESUME 且存在 checkpoint.pth：
        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            # checkpoint_file = "/mnt/disk1/zhanjiao/workspace/YOLOP-main-test/runs/BddDataset/_2022-05-29-01-00/epoch-120.pth"
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer = get_optimizer(cfg, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            #cfg.NEED_AUTOANCHOR = False     #disable autoanchor
        # model = model.to(device)

        #是否需要只训练某一分支
        # if cfg.TRAIN.ENC_DET_ONLY or cfg.TRAIN.DET_ONLY:    # Only train encoder and detection branchs
        #     logger.info('freeze two Seg heads...')
        #     for k, v in model.named_parameters():
        #         v.requires_grad = True  # train all layers
        #         if k.split(".")[1] in Da_Head_para_idx + Ll_Head_para_idx:
        #             print('freezing %s' % k)
        #             v.requires_grad = False

        # if cfg.TRAIN.LANE_ONLY: 
        #     logger.info('freeze encoder and Det head and Da_Seg heads...')
        #     # print(model.named_parameters)
        #     for k, v in model.named_parameters():
        #         v.requires_grad = True  # train all layers
        #         if k.split(".")[1] in Encoder_para_idx + Da_Head_para_idx + Det_Head_para_idx:
        #             print('freezing %s' % k)
        #             v.requires_grad = False

        # if cfg.TRAIN.DRIVABLE_ONLY:
        #     logger.info('freeze encoder and Det head and Ll_Seg heads...')
        #     # print(model.named_parameters)
        #     for k, v in model.named_parameters():
        #         v.requires_grad = True  # train all layers
        #         if k.split(".")[1] in Encoder_para_idx + Ll_Head_para_idx + Det_Head_para_idx:
        #             print('freezing %s' % k)
        #             v.requires_grad = False
        
    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
        # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # # DDP mode
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

    # assign model params
    model.gr = 1.0
    model.nc = 4
    model.names = ['0', '1', '2', '3']
    # print('bulid model finished')

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None

    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
        # prefetch_factor = 4
        # persistent_workers = True
    )
    num_batch = len(train_loader)

    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
        print('load data finished')
    
    # if rank in [-1, 0]:
    #     if cfg.NEED_AUTOANCHOR:
    #         logger.info("begin check anchors")
    #         run_anchor(logger,train_dataset, model=model, thr=cfg.TRAIN.ANCHOR_THRESHOLD, imgsz=min(cfg.MODEL.IMAGE_SIZE))
    #     else:
    #         logger.info("anchors loaded successfully")
    #         det = model.module.model[model.module.detector_index] if is_parallel(model) \
    #             else model.model[model.detector_index]
    #         logger.info(str(det.anchors))
    
    # training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    # scaler = torch.cuda.amp.GradScaler()
    print('=> start training...')
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        if rank != -1:
            #打乱
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        # for i, (input, target, paths, shapes, task) in enumerate(train_loader):
        #     # 假设你的数据集返回的是 (image, target)，这里打印targets的索引或者样本编号
        #     # 具体看你的dataset输出什么，这里假设target里有index或你用image tensor的id代替
        #     print(f"  Batch {i}: first sample id (or target) = {paths}")
        train(cfg, train_loader, model, model1,criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, rank)

        lr_scheduler.step()

        # evaluate on validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH or epoch in list(range(181,200)) or epoch in [162, 165, 167, 170, 172, 175, 178]) and rank in [-1, 0]:
            # print('validate')
            da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
                epoch,cfg, valid_loader, valid_dataset, model,model1, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, rank
            )
            fi = fitness(np.array(detect_results).reshape(1, -1))  #目标检测评价指标

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          epoch,  loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1])
            logger.info(msg)

        # save checkpoint model and best model
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH or epoch in list(range(181,200)) or epoch in [162, 165, 167, 170, 172, 175, 178]) and rank in [-1, 0]:
        # if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))

            savepath1 = os.path.join(final_output_dir, f'epoch-{epoch}_model1.pth')
            logger.info('=> saving checkpoint of model1 to {}'.format(savepath1))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME + '_model1',
                model=model1,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}_model1.pth'
            )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME + '_model1',
                model=model1,
                optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint_model1.pth'
            )

            save_checkpoint(
                    epoch=epoch,
                    name=cfg.MODEL.NAME,
                    model=model,
                    # 'best_state_dict': model.module.state_dict(),
                    # 'perf': perf_indicator,
                    optimizer=optimizer,
                    output_dir=final_output_dir,
                    filename=f'epoch-{epoch}.pth'
                )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint.pth'
            )

    # save final mod

if __name__ == '__main__':
    main()