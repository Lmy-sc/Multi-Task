import time
from lib.core.evaluate import ConfusionMatrix, SegmentationMetric, DepthMetric
from lib.core.general import non_max_suppression, check_img_size, scale_coords, xyxy2xywh, xywh2xyxy, box_iou, \
    coco80_to_coco91_class, plot_images, ap_per_class, output_to_target, save_segmentation_visualizations, \
    save_depth_visualizations, save_raw_visualizations, flatten_grads, pcgrad, write_grads_to_model, \
    UncertaintyWeighting
from lib.utils.utils import time_synchronized
from lib.utils import plot_img_and_mask, plot_one_box, show_seg_result
import torch
from threading import Thread
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json
import random
import cv2
import os
import math
from torch.cuda import amp
from tqdm import tqdm
from pathlib import Path
import pdb

from unitmodule.models.data_preprocessors import unit_module

grad_snapshots = {}


def save_grad_snapshot(model, task_id):
    # 保存当前模型的梯度副本（注意此时是缩放后的）
    grad = []
    for p in model.parameters():
        if p.grad is not None:
            grad.append(p.grad.detach().clone())
        else:
            grad.append(None)
    grad_snapshots[task_id] = grad


def train(cfg, train_loader, model, model1, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, rank=-1):
    """
    train for one epoch

    Inputs:
    - config: configurations
    - train_loader: loder for data
    - model:
    - criterion: (function) calculate all the loss, return total_loss, head_losses
    - writer_dict:
    outputs(2,)
    output[0] len:3, [1,3,32,32,85], [1,3,16,16,85], [1,3,8,8,85]
    output[1] len:1, [2,256,256]
    output[2] len:1, [2,256,256]
    target(2,)
    target[0] [1,n,5]
    target[1] [2,256,256]
    target[2] [2,256,256]
    Returns:
    None

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()

    depth_loss = []
    det_loss = []
    seg_loss = []
    task_grads = {}
    task_loss1 = {}

    accumulate_steps = 0
    num_tasks = 3
    uncertainty_weighting = UncertaintyWeighting(num_tasks)
    uncertainty_weighting_det = UncertaintyWeighting(2)
    uncertainty_weighting_seg = UncertaintyWeighting(2)
    uncertainty_weighting_depth = UncertaintyWeighting(2)
    for i, (input, target, paths, shapes, task) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
        # for i, (input, target, paths, shapes,task) in enumerate(train_loader):

        # print("trainloader",paths)
        # print(input, target, paths, shapes,task)
        intermediate = time.time()
        # print('tims:{}'.format(intermediate-start))
        num_iter = i + num_batch * (epoch - 1)

        if num_iter < num_warmup:
            # warm up
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
            xi = [0, num_warmup]
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(num_iter, xi,
                                    [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])

        data_time.update(time.time() - start)
        # if not cfg.DEBUG:
        #     input = input.to(device, non_blocking=True)
        #     assign_target = []
        #     for tgt in target:
        #         print(tgt)
        #         assign_target.append(tgt.to(device))
        #     target = assign_target
        if not cfg.DEBUG:
            assign_input= []
            for ipt in input :
                ipt = ipt.to(device, non_blocking=True)
                assign_input.append(ipt)
            input = assign_input
            assign_target = []

            # print(target)
            for tgt in target:
                # print(type(tgt), tgt.shape if tgt is not None else "None")
                if isinstance(tgt, torch.Tensor):
                    assign_target.append(tgt.to(device))
                else:
                    assign_target.append(None)

            target = assign_target

        with amp.autocast(enabled=device.type != 'cpu'):
            # input1, losses_unit_module = model1(input)
            # weights = [4.5, 0.01, 2, 0, 0.1]  # 自定义的加权系数
            #
            # total_loss_enhance = 0
            # for i, (name, loss_val) in enumerate(losses_unit_module.items()):
            #     weight = weights[i]
            #     if i == 3:
            #         continue  # 把这个模块中所有子损失加起来
            #     total_loss_enhance += weight * loss_val
            # total_loss_enhance *= 0.4
            # print(f"total_loss_enhance=={total_loss_enhance}")
            #task1 = task[0] if isinstance(task, list) else task
            # outputs = model(input1, task)
            # total_loss, head_losses = criterion(outputs, target, shapes, model, input)
            # # total_loss = total_loss + sum(losses_unit_module.values())
            #
            # task_loss = [total_loss_enhance, total_loss]
            # if task1 == "detect":
            #     task_loss = uncertainty_weighting_det(task_loss)
            # elif task1 == "seg":
            #     task_loss = uncertainty_weighting_seg(task_loss)
            # elif task1 == "depth":
            #     task_loss = uncertainty_weighting_depth(task_loss)
            # print(f"task_loss=={task_loss}")

            outputs_det = model(input[0], task[0][0])
            outputs_seg = model(input[1], task[0][1])
            outputs_depth= model(input[2], task[0][2])
            total_loss_det, head_losses_det = criterion(outputs_det, target, shapes[0], model, input[0],task[0][0])
            total_loss_seg, head_losses_seg = criterion(outputs_seg, target, shapes[1], model, input[1],task[0][1])
            total_loss_depth, head_losses_depth = criterion(outputs_depth, target, shapes[2], model, input[2],task[0][2])
            print(f"det_loss,{total_loss_det}", f"seg_Loss,{total_loss_seg}", f"depth_loss,{total_loss_depth}")

            total_loss = (total_loss_det +total_loss_seg +total_loss_depth)/3

        # freeze_branch(model, task)
        # if i%3== 0 :
        # compute gradient and do update step
        # optimizer.zero_grad()
        # ith torch.autograd.detect_anomaly():

        optimizer.zero_grad()

        task_losses = [total_loss_det, total_loss_seg, total_loss_depth]
        task_names = ['det', 'seg', 'depth']
        task_grads = {}
        freeze_branch(model,epoch)
        # 逐个任务执行 backward
        for i in range(3):
            loss = task_losses[i]
            task = task_names[i]

            # backward with AMP
            scaler.scale(loss).backward()
            task_grads[task] = flatten_grads(model)
            optimizer.zero_grad()  # 清梯度，以防残留（注意顺序）

        # 获取 AMP scale 并反缩放
        scale = scaler.get_scale()
        inv = 1.0 / scale
        for k in task_grads:
            task_grads[k] = task_grads[k] * inv

        # 执行 PCGrad
        if torch.rand(1).item() > 0.3:
            pcgrad_grads = pcgrad(task_grads)
            print("=== PCGrad 后梯度大小 ===")
            for i, grad in enumerate(pcgrad_grads):
                print(f"[{task_names[i]}] grad norm after PCGrad: {grad.norm().item():.6f}")
        else:
            pcgrad_grads = list(task_grads.values())
            print("未执行 PCGrad，直接使用原始梯度。")

        # Uncertainty Weighting（例如用均值或 learned log-sigma 权重）
        resize_loss = uncertainty_weighting(pcgrad_grads)

        print("=== Uncertainty Weighted 后梯度大小 ===")
        print(f"[UW] grad norm after weighting: {resize_loss.norm().item():.6f}")

        # 再放大 scale（AMP）
        resize_loss = resize_loss * scale
        print(f"after scale [UW] grad norm: {resize_loss.norm().item():.6f}")

        # 写入梯度
        write_grads_to_model(model, resize_loss)

        # optimizer step
        scaler.step(optimizer)
        scaler.update()
            # scaler.scale(task_loss).backward()
            # task_grads[accumulate_steps-1] = flatten_grads(model)
            # optimizer.zero_grad()
            #
            # # # if accumulate_steps == num_tasks :
            # scale = scaler.get_scale()
            # inv = 1.0 / scale
            # task_grads[accumulate_steps-1] = task_grads[accumulate_steps-1] * inv
            #
            # if accumulate_steps == num_tasks:
            #     # scaler.unscale_(optimizer)
            #     # ----- PCGrad 投影 -----
            #     if torch.rand(1).item() > 0.3:  # 70%的概率执行PCGrad
            #         # ----- PCGrad 投影 -----
            #         pcgrad_grads = pcgrad(task_grads)  # task_grads: dict {task_id: flattened_grad}
            #         print("=== PCGrad 后梯度大小 ===")
            #         for i, grad in enumerate(pcgrad_grads):
            #             print(
            #                 f"[{'det' if i == 0 else 'seg' if i == 1 else 'depth'}] grad norm after  PCGrad: {grad.norm().item():.6f}")
            #     else:
            #         # 30%的几率不执行 PCGrad
            #         pcgrad_grads = list(task_grads.values())  # task_grads: dict {task_id: flattened_grad}
            #         print("未执行 PCGrad，直接使用原始梯度。")
            #
            #     resize_loss = uncertainty_weighting(pcgrad_grads)
            #     print("=== Uncertainty Weighted 后梯度大小 ===")
            #     print(f"[UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
            #     resize_loss = resize_loss * scale
            #     print(f"after scale [UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
            #     # 将梯度写入 model（注意 unflatten）
            #     write_grads_to_model(model, resize_loss)
            #     # write_grads_to_model(model, pcgrad_grads)
            #
            #     # 优化器 step
            #     scaler.step(optimizer)
            #     scaler.update()
            #
            #     # 重置梯度累积计数器
            #     task_grads.clear()
            #     task_loss1.clear()
            #
            #     optimizer.zero_grad()
            #     accumulate_steps = 0
            #
            # scaler.scale(total_loss).backward()

        if rank in [-1, 0]:
            # measure accuracy and record loss
            losses.update(total_loss.item(), input[0].size(0))

            # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
            #                                  target.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - start)
            end = time.time()
            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time, loss=losses)

                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def freeze_branch(model, epoch):

    if epoch < 30 :
        for i, m in enumerate(model.model):
            if i not in [2,6,20]:
                for param in m.parameters():
                    param.requires_grad = True
            else:
                for param in m.parameters():
                    param.requires_grad = False

    else:
        for i, m in enumerate(model.model):
            for param in m.parameters():
                param.requires_grad = True


def validate(epoch, config, val_loader, val_dataset, model, model1, criterion, output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    validata

    Inputs:
    - config: configurations
    - train_loader: loder for data
    - model:
    - criterion: (function) calculate all the loss, return
    - writer_dict:

    Return:
    None
    """
    # setting
    max_stride = 32
    weights = None

    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # print(save_dir)
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]  # imgsz is multiple of max_stride
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    is_coco = False  # is coco dataset
    save_conf = False  # save auto-label confidences
    verbose = False
    save_hybrid = False
    log_imgs, wandb = min(16, 100), None

    device = torch.device('cuda:0')
    nc = 4
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=model.nc)  # detector confusion matrix

    da_metric = SegmentationMetric(config.num_seg_class)  # segment confusion matrix

    # ll_metric = SegmentationMetric(2)  # segment confusion matrix
    depth_metric = DepthMetric(valid_threshold=0.1)

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.

    losses = AverageMeter()

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    # ll_acc_seg = AverageMeter()
    # ll_IoU_seg = AverageMeter()
    # ll_mIoU_seg = AverageMeter()

    depth_rmse_eval = AverageMeter()
    depth_sqrel_eval = AverageMeter()
    depth_mae_eval  = AverageMeter()


    T_inf = AverageMeter()
    T_nms = AverageMeter()

    # switch to eval mode
    model1.eval()
    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    depth_metric.reset()
    for batch_i, (img, target, paths, shapes, task) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Valid"):
        device = torch.device('cuda:0')  # 指定使用第一个 GPU
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            # 使用列表推导将每个张量移动到指定的设备
            for idx, tensor in enumerate(target):
                if tensor is not None:
                    target[idx] = tensor.to(device)
            assign_target = []

            # print(target)
            for tgt in target:
                print(type(tgt), tgt.shape if tgt is not None else "None")
                if isinstance(tgt, torch.Tensor):
                    assign_target.append(tgt.to(device))
                else:
                    assign_target.append(None)

            # img = img.to(device, non_blocking=True)
            # assign_target = []
            # for tgt in target:
            #     assign_target.append(tgt.to(device))
            # target = assign_target
            nb, _, height, width = img.shape  # batch size, channel, height, width

        with torch.no_grad():
            task = task[0] if (isinstance(task, list) or isinstance(task, tuple)) else task
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]

            t = time_synchronized()

            img = img.to(device)

            # input1, loss = model1(img)
            # if config.TEST.PLOTS and batch_i <50 :
            #     f = save_dir + '/' + f'pic_test_batch{batch_i}_labels.jpg'
            #     save_raw_visualizations(img, input1,f)
            #
            # det_out, da_seg_out, ll_seg_out = model(input1, task)
            det_out, da_seg_out, ll_seg_out = model(img, task)

            t_inf = time_synchronized() - t

            if batch_i > 0:
                T_inf.update(t_inf / img.size(0), img.size(0))

            train_out = None
            inf_out = None


            if task == "detect":
                inf_out, train_out = det_out

            if task == "seg":
                # driving area segment evaluation
                # print(da_seg_out.size(),target[1].size())
                _, da_predict = torch.max(da_seg_out, 1)
                da_gt = target[1]
                # _,da_gt=torch.max(target[1], 1)
                da_predict = da_predict[:, pad_h:height - pad_h, pad_w:width - pad_w]
                da_gt = da_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]

                da_metric.reset()
                da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
                da_acc = da_metric.pixelAccuracy()

                da_IoU = da_metric.IntersectionOverUnion()
                da_mIoU = da_metric.meanIntersectionOverUnion()

                da_acc_seg.update(da_acc, img.size(0))
                da_IoU_seg.update(da_IoU, img.size(0))
                da_mIoU_seg.update(da_mIoU, img.size(0))

                if config.TEST.PLOTS and batch_i <50 :
                    f = save_dir + '/' + f'seg_test_batch{batch_i}_labels.jpg'
                    save_segmentation_visualizations(img, da_seg_out, target[1], f)

            if task == "depth":
                # lane line segment evaluation
                depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, depth_est = ll_seg_out
                print(depth_est.size(), target[2].size())
                # _, ll_predict = torch.max(depth_est, 1)
                # _, ll_gt = torch.max(target[2], 1)
                ll_predict = torch.squeeze(depth_est,dim=1)
                ll_gt = torch.squeeze(target[2],dim=1)
                # ll_gt = target[2]
                ll_predict = ll_predict[:, pad_h:height - pad_h, pad_w:width - pad_w]
                ll_gt = ll_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]


                depth_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
                depth_rmse = depth_metric.get_rmse()
                depth_mae = depth_metric.get_mae()
                depth_sqrel = depth_metric.get_sqrel()

                depth_rmse_eval.update( depth_rmse, img.size(0))
                depth_mae_eval.update( depth_mae, img.size(0))
                depth_sqrel_eval.update( depth_sqrel, img.size(0))

                # ll_metric.reset()
                # ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
                # ll_acc = ll_metric.lineAccuracy()
                # ll_IoU = ll_metric.IntersectionOverUnion()
                # ll_mIoU = ll_metric.meanIntersectionOverUnion()
                #
                # ll_acc_seg.update(ll_acc, img.size(0))
                # ll_IoU_seg.update(ll_IoU, img.size(0))
                # ll_mIoU_seg.update(ll_mIoU, img.size(0))
                if config.TEST.PLOTS and batch_i <50 :
                    f = save_dir + '/' + f'depth_test_batch{batch_i}_labels.jpg'
                    save_depth_visualizations(img, depth_est, target[2], f)

            total_loss, head_losses = criterion((train_out,da_seg_out,ll_seg_out), target, shapes,model, img ,task)   #Compute loss
            losses.update(total_loss.item(), img.size(0))
            #losses.update(0.1, img.size(0))

            if task == "detect":
                # NMS
                t = time_synchronized()
                # target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = []  # for autolabelling
                output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRESHOLD,
                                             iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
                # output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
                # output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRES, iou_thres=config.TEST.NMS_IOU_THRES)
                t_nms = time_synchronized() - t
                if batch_i > 0:
                    T_nms.update(t_nms / img.size(0), img.size(0))

        # Statistics per image
        # output([xyxy,conf,cls])
        # target[0] ([img_id,cls,xyxy])
        if task == 'detect':
            nlabel = (target[0].sum(dim=2) > 0).sum(dim=1)  # number of objects # [batch, num_gt ]
            for si, pred in enumerate(output):
                # gt per image
                nl = int(nlabel[si])
                labels = target[0][si, :nl, 0:5]  # [ num_gt_per_image, [cx, cy, w, h] ]
                # gt_classes = target[0][si, :nl, 0]

                # labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image
                # nl = num_gt
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path = Path(paths[si])
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Append to text file
                if config.TEST.SAVE_TXT:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    for *xyxy, conf, cls in predn.tolist():
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # W&B logging
                if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

                # Append to pycocotools JSON dictionary
                if config.TEST.SAVE_JSON:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(predn[:, :4])  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                      'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    print(shapes[si][0])
                    print(shapes[si][1])
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                    if config.TEST.PLOTS:
                        confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            # n*m  n:pred  m:label
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            if config.TEST.PLOTS and batch_i < 55:
                f = save_dir + '/' + f'det_test_batch{batch_i}_labels.jpg'  # labels
                plot_images(img, target[0], paths, f, names)
                # Thread(target=plot_images, args=(img, target[0], paths, f, names), daemon=True).start()
                f = save_dir + '/' + f'det_test_batch{batch_i}_pred.jpg'  # predictions
                plot_images(img, output_to_target(output), paths, f, names)
                # Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()
                # 可视化原图、GT和预测框
                # mean = [0.485, 0.456, 0.406]
                # std = [0.229, 0.224, 0.225]
                #
                # img_vis = reverse_transform(img[si], mean, std)
                # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                # h,w = shapes[si][0]
                # img_vis = cv2.resize(img_vis, (w,h))
                #
                # # 画GT框（绿色）
                # for l in labels:
                #     cls_id = int(l[0])
                #     box = l[1:5].clone().view(1, 4)
                #     box = xywh2xyxy(box)
                #     scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # 还原坐标
                #     x1, y1, x2, y2 = box[0].int().tolist()
                #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     cv2.putText(img_vis, f'GT:{cls_id}', (x1, y1 - 5), 0, 0.6, (0, 255, 0), 2)
                #
                # # 画预测框（红色）
                # for p in predn:
                #     x1, y1, x2, y2 = p[0:4].int().tolist()
                #     conf = p[4].item()
                #     cls_id = int(p[5].item())
                #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #     cv2.putText(img_vis, f'Pred:{cls_id} {conf:.2f}', (x1, y2 + 15), 0, 0.5, (0, 0, 255), 2)
                #
                # save_dir = Path(save_dir)
                #
                # # 保存对比图
                # save_path = save_dir / 'vis' / f'{path.stem}.jpg'
                # save_path.parent.mkdir(parents=True, exist_ok=True)
                # cv2.imwrite(str(save_path), img_vis)

    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
    # if task == 'detect':
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        print("------------------------------------")
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
        ap50, ap70, ap75, ap = ap[:, 0], ap[:, 4], ap[:, 5], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # print(map70)
    # print(map75)

    # Print results per class
    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log(
                {"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(
            weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in
                                      val_loader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    da_segment_result = None
    ll_segment_result = None

    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)

    ll_segment_result = (depth_rmse_eval.avg, depth_mae_eval.avg, depth_sqrel_eval.avg)

    # print(da_segment_result)
    # print(ll_segment_result)
    detect_result = np.asarray([mp, mr, map50, map])
    # print('mp:{},mr:{},map50:{},map:{}'.format(mp, mr, map50, map))
    # print segmet_result
    t = [T_inf.avg, T_nms.avg]
    return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def tensor_to_image(tensor):
    """
    把 [C, H, W] 的 tensor 转成 [H, W, 3] 的 numpy 图像
    会自动归一化到 [0,1]
    """
    tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        if tensor.size(0) == 1:  # 单通道
            img = tensor.squeeze(0).numpy()
            img = np.stack([img] * 3, axis=-1)  # 变伪RGB
        elif tensor.size(0) == 3:
            img = tensor.permute(1, 2, 0).numpy()
        else:
            img = tensor[:3].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        return img
    else:
        raise ValueError("tensor 应该是 [C, H, W] 维度")


import os
import numpy as np
import torch
import cv2


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # C x H x W
        tensor = tensor * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)
        tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
        tensor = tensor.transpose(1, 2, 0)  # -> H x W x C
    return tensor


def tensor_to_image(tensor, denormalize=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # C x H x W
        if denormalize:
            tensor = tensor * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + \
                     np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        tensor = tensor.transpose(1, 2, 0)  # H x W x C
    if tensor.shape[2] == 1:
        tensor = np.repeat(tensor, 3, axis=2)
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return tensor


def save_image_pairs_with_denorm(imgs, preds, paths, save_dir, prefix="sample", denormalize=True):
    os.makedirs(save_dir, exist_ok=True)
    batch_size = imgs.size(0)

    for i in range(batch_size):
        image = tensor_to_image(imgs[i], denormalize=denormalize)
        pred = tensor_to_image(preds[i], denormalize=denormalize)  # 一般预测图不用反归一化

        # RGB -> BGR for OpenCV saving
        image_bgr = image[:, :, ::-1]
        pred_bgr = pred[:, :, ::-1]

        concat = np.concatenate([image_bgr, pred_bgr], axis=1)

        # 获取路径后缀标识
        full_path = os.path.normpath(paths[i])
        parts = full_path.split(os.sep)
        suffix = "_".join(parts[-3:])

        filename = f"{prefix}_{suffix}.png"
        save_path = os.path.join(save_dir, filename)

        cv2.imwrite(save_path, concat)


def reverse_transform(img_tensor, mean, std):
    """
    将 PyTorch tensor 图像（3xHxW）还原为 uint8 格式的 RGB NumPy 图像（HxWx3）。

    参数:
        img_tensor: 输入图像 tensor（shape: [3, H, W]，值通常在归一化后的范围）
        mean: 用于归一化的均值（列表或元组）
        std: 用于归一化的标准差（列表或元组）

    返回:
        img_vis: 已反归一化并转换为 uint8 的 NumPy 图像，RGB 格式，HWC 排布。
    """
    assert isinstance(img_tensor, torch.Tensor), "输入必须是 PyTorch Tensor"
    img = img_tensor.clone().detach().cpu()  # 安全复制

    # 反归一化： x = x * std + mean
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)

    img = img.clamp(0, 1)  # 保证范围在 [0, 1]
    img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
    img_vis = (img * 255).astype(np.uint8)  # 转为 uint8 格式
    return img_vis


# import time
# from lib.core.evaluate import ConfusionMatrix, SegmentationMetric
# from lib.core.general import non_max_suppression, check_img_size, scale_coords, xyxy2xywh, xywh2xyxy, box_iou, \
#     coco80_to_coco91_class, plot_images, ap_per_class, output_to_target, save_segmentation_visualizations, \
#     save_depth_visualizations, save_raw_visualizations, flatten_grads, pcgrad, write_grads_to_model, \
#     UncertaintyWeighting
# from lib.utils.utils import time_synchronized
# from lib.utils import plot_img_and_mask, plot_one_box, show_seg_result
# import torch
# from threading import Thread
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from pathlib import Path
# import json
# import random
# import cv2
# import os
# import math
# from torch.cuda import amp
# from tqdm import tqdm
# from pathlib import Path
# import pdb
#
# from unitmodule.models.data_preprocessors import unit_module
#
# grad_snapshots = {}
#
#
# def save_grad_snapshot(model, task_id):
#     # 保存当前模型的梯度副本（注意此时是缩放后的）
#     grad = []
#     for p in model.parameters():
#         if p.grad is not None:
#             grad.append(p.grad.detach().clone())
#         else:
#             grad.append(None)
#     grad_snapshots[task_id] = grad
#
#
# def train(cfg, train_loader, model, model1, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
#           writer_dict, logger, device, rank=-1):
#     """
#     train for one epoch
#
#     Inputs:
#     - config: configurations
#     - train_loader: loder for data
#     - model:
#     - criterion: (function) calculate all the loss, return total_loss, head_losses
#     - writer_dict:
#     outputs(2,)
#     output[0] len:3, [1,3,32,32,85], [1,3,16,16,85], [1,3,8,8,85]
#     output[1] len:1, [2,256,256]
#     output[2] len:1, [2,256,256]
#     target(2,)
#     target[0] [1,n,5]
#     target[1] [2,256,256]
#     target[2] [2,256,256]
#     Returns:
#     None
#
#     """
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#
#     # switch to train mode
#     model.train()
#     start = time.time()
#
#     depth_loss = []
#     det_loss = []
#     seg_loss = []
#     task_grads = {}
#     task_loss1 = {}
#
#     accumulate_steps = 0
#     num_tasks = 3
#     uncertainty_weighting_pcgrad = UncertaintyWeighting(num_tasks)
#     uncertainty_weighting_nopcgrad = UncertaintyWeighting(num_tasks)
#     uncertainty_weighting_det = UncertaintyWeighting(2)
#     uncertainty_weighting_seg = UncertaintyWeighting(2)
#     uncertainty_weighting_depth = UncertaintyWeighting(2)
#     for i, (input, target, paths, shapes, task) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
#         # for i, (input, target, paths, shapes,task) in enumerate(train_loader):
#
#         # print("trainloader",paths)
#         # print(input, target, paths, shapes,task)
#         intermediate = time.time()
#         # print('tims:{}'.format(intermediate-start))
#         num_iter = i + num_batch * (epoch - 1)
#
#         if num_iter < num_warmup:
#             # warm up
#             lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
#                            (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
#             xi = [0, num_warmup]
#             # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
#             for j, x in enumerate(optimizer.param_groups):
#                 # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
#                 x['lr'] = np.interp(num_iter, xi,
#                                     [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
#                 if 'momentum' in x:
#                     x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])
#
#         data_time.update(time.time() - start)
#         # if not cfg.DEBUG:
#         #     input = input.to(device, non_blocking=True)
#         #     assign_target = []
#         #     for tgt in target:
#         #         print(tgt)
#         #         assign_target.append(tgt.to(device))
#         #     target = assign_target
#         if not cfg.DEBUG:
#             input = input.to(device, non_blocking=True)
#             assign_target = []
#
#             # print(target)
#             for tgt in target:
#                 # print(type(tgt), tgt.shape if tgt is not None else "None")
#                 if isinstance(tgt, torch.Tensor):
#                     assign_target.append(tgt.to(device))
#                 else:
#                     assign_target.append(None)
#
#             target = assign_target
#
#         with amp.autocast(enabled=device.type != 'cpu'):
#             # input1, losses_unit_module = model1(input)
#             # weights = [5, 0.01, 0.01, 0, 0.1]  # 自定义的加权系数
#             #
#             # total_loss_enhance = 0
#             # for i, (name, loss_val) in enumerate(losses_unit_module.items()):
#             #     weight = weights[i]
#             #     if i == 3:
#             #         continue  # 把这个模块中所有子损失加起来
#             #     total_loss_enhance += weight * loss_val
#             # print(f"total_loss_enhance=={total_loss_enhance}")
#             task1 = task[0] if isinstance(task, list) else task
#             # outputs = model(input1, task)
#             # total_loss, head_losses = criterion(outputs, target, shapes, model, input)
#             # # total_loss = total_loss + sum(losses_unit_module.values())
#             #
#             # task_loss = [total_loss_enhance, total_loss]
#             # if task1 == "detect":
#             #     task_loss = uncertainty_weighting_det(task_loss)
#             # elif task1 == "seg":
#             #     task_loss = uncertainty_weighting_seg(task_loss)
#             # elif task1 == "depth":
#             #     task_loss = uncertainty_weighting_depth(task_loss)
#             # print(f"task_loss=={task_loss}")
#
#             outputs = model(input, task)
#             total_loss, head_losses = criterion(outputs, target, shapes, model, input)
#             total_loss = total_loss
#             task_loss = total_loss
#
#         # freeze_branch(model, task)
#         # if i%3== 0 :
#         # compute gradient and do update step
#         # optimizer.zero_grad()
#         # ith torch.autograd.detect_anomaly():
#
#         optimizer.zero_grad()
#         accumulate_steps += 1
#         if (accumulate_steps==1 and task1 != "detect") or (accumulate_steps==2 and task1 != "seg") or(accumulate_steps==3 and task1 != "depth"):
#             freeze_branch(model, task)
#             scaler.scale(task_loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#
#
#         else:
#             scaler.scale(task_loss).backward()
#             task_grads[accumulate_steps-1] = flatten_grads(model)
#             optimizer.zero_grad()
#
#             # # if accumulate_steps == num_tasks :
#             scale = scaler.get_scale()
#             inv = 1.0 / scale
#             task_grads[accumulate_steps-1] = task_grads[accumulate_steps-1] * inv
#
#             if accumulate_steps == num_tasks:
#                 # scaler.unscale_(optimizer)
#                 # ----- PCGrad 投影 -----
#                 if torch.rand(1).item() > 0.7:  # 70%的概率执行PCGrad
#                     # ----- PCGrad 投影 -----
#                     pcgrad_grads = pcgrad(task_grads)  # task_grads: dict {task_id: flattened_grad}
#                     print("=== PCGrad 后梯度大小 ===")
#                     for i, grad in enumerate(pcgrad_grads):
#                         print(
#                             f"[{'det' if i == 0 else 'seg' if i == 1 else 'depth'}] grad norm after  PCGrad: {grad.norm().item():.6f}")
#                     resize_loss = uncertainty_weighting_pcgrad(pcgrad_grads)
#                     print("=== Uncertainty Weighted 后梯度大小 ===")
#                     print(f"[UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
#                     resize_loss = resize_loss * scale
#                     print(f"after scale [UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
#                     # 将梯度写入 model（注意 unflatten）
#                     write_grads_to_model(model, resize_loss)
#                     # write_grads_to_model(model, pcgrad_grads)
#
#                     # 优化器 step
#                     scaler.step(optimizer)
#                     scaler.update()
#
#                     # 重置梯度累积计数器
#                     task_grads.clear()
#                     task_loss1.clear()
#
#                     optimizer.zero_grad()
#                     accumulate_steps = 0
#
#                 else:
#                     # 30%的几率不执行 PCGrad
#                     pcgrad_grads = list(task_grads.values())  # task_grads: dict {task_id: flattened_grad}
#                     print("未执行 PCGrad，直接使用原始梯度。")
#                     resize_loss = uncertainty_weighting_nopcgrad(pcgrad_grads)
#                     print("=== Uncertainty Weighted 后梯度大小 ===")
#                     print(f"[UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
#                     resize_loss = resize_loss * scale
#                     print(f"after scale [UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
#                     # 将梯度写入 model（注意 unflatten）
#                     write_grads_to_model(model, resize_loss)
#                     # write_grads_to_model(model, pcgrad_grads)
#
#                     # 优化器 step
#                     scaler.step(optimizer)
#                     scaler.update()
#
#                     # 重置梯度累积计数器
#                     task_grads.clear()
#                     task_loss1.clear()
#
#                     optimizer.zero_grad()
#                     accumulate_steps = 0
#
#                 # resize_loss = uncertainty_weighting_nopcgrad(pcgrad_grads)
#                 # print("=== Uncertainty Weighted 后梯度大小 ===")
#                 # print(f"[UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
#                 # resize_loss = resize_loss * scale
#                 # print(f"after scale [UW] grad norm after weighting: {resize_loss.norm().item():.6f}")
#                 # # 将梯度写入 model（注意 unflatten）
#                 # write_grads_to_model(model, resize_loss)
#                 # # write_grads_to_model(model, pcgrad_grads)
#                 #
#                 # # 优化器 step
#                 # scaler.step(optimizer)
#                 # scaler.update()
#                 #
#                 # # 重置梯度累积计数器
#                 # task_grads.clear()
#                 # task_loss1.clear()
#                 #
#                 # optimizer.zero_grad()
#                 # accumulate_steps = 0
#
#             # scaler.scale(total_loss).backward()
#             # 不做0.7？or 再新增一个
#
#         if rank in [-1, 0]:
#             # measure accuracy and record loss
#             losses.update(task_loss.item(), input.size(0))
#
#             # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
#             #                                  target.detach().cpu().numpy())
#             # acc.update(avg_acc, cnt)
#
#             # measure elapsed time
#             batch_time.update(time.time() - start)
#             end = time.time()
#             if i % cfg.PRINT_FREQ == 0:
#                 msg = 'Epoch: [{0}][{1}/{2}]\t' \
#                       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
#                       'Speed {speed:.1f} samples/s\t' \
#                       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
#                       'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
#                     epoch, i, len(train_loader), batch_time=batch_time,
#                     speed=input.size(0) / batch_time.val,
#                     data_time=data_time, loss=losses)
#
#                 logger.info(msg)
#
#                 writer = writer_dict['writer']
#                 global_steps = writer_dict['train_global_steps']
#                 writer.add_scalar('train_loss', losses.val, global_steps)
#                 # writer.add_scalar('train_acc', acc.val, global_steps)
#                 writer_dict['train_global_steps'] = global_steps + 1
#                 #print freq==1200?
#
#
# def freeze_branch(model, task):
#     task = task[0] if isinstance(task, list) else task
#     for i, m in enumerate(model.model):
#         if task == 'detect' and i not in [0, 1, 2]:
#             for param in m.parameters():
#                 param.requires_grad = False
#         elif task == 'seg' and i not in [0, 1] + list(range(3, 17)):
#             for param in m.parameters():
#                 param.requires_grad = False
#         elif task == 'depth' and i not in [0, 1] + list(range(17, 19)):
#             for param in m.parameters():
#                 param.requires_grad = False
#         else:
#             for param in m.parameters():
#                 param.requires_grad = True
#
#
# def validate(epoch, config, val_loader, val_dataset, model, model1, criterion, output_dir,
#              tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
#     """
#     validata
#
#     Inputs:
#     - config: configurations
#     - train_loader: loder for data
#     - model:
#     - criterion: (function) calculate all the loss, return
#     - writer_dict:
#
#     Return:
#     None
#     """
#     # setting
#     max_stride = 32
#     weights = None
#
#     save_dir = output_dir + os.path.sep + 'visualization'
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#
#     # print(save_dir)
#     _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]  # imgsz is multiple of max_stride
#     batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
#     test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
#     training = False
#     is_coco = False  # is coco dataset
#     save_conf = False  # save auto-label confidences
#     verbose = False
#     save_hybrid = False
#     log_imgs, wandb = min(16, 100), None
#
#     device = torch.device('cuda:0')
#     nc = 4
#     iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
#     niou = iouv.numel()
#
#     try:
#         import wandb
#     except ImportError:
#         wandb = None
#         log_imgs = 0
#
#     seen = 0
#     confusion_matrix = ConfusionMatrix(nc=model.nc)  # detector confusion matrix
#
#     da_metric = SegmentationMetric(config.num_seg_class)  # segment confusion matrix
#
#     # ll_metric = SegmentationMetric(2)  # segment confusion matrix
#     depth_metric = DepthMetric(valid_threshold=0.1)
#
#     names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#     coco91class = coco80_to_coco91_class()
#
#     s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
#     p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
#
#     losses = AverageMeter()
#
#     da_acc_seg = AverageMeter()
#     da_IoU_seg = AverageMeter()
#     da_mIoU_seg = AverageMeter()
#
#     # ll_acc_seg = AverageMeter()
#     # ll_IoU_seg = AverageMeter()
#     # ll_mIoU_seg = AverageMeter()
#
#     depth_rmse_eval = AverageMeter()
#     depth_sqrel_eval = AverageMeter()
#     depth_mae_eval  = AverageMeter()
#
#
#     T_inf = AverageMeter()
#     T_nms = AverageMeter()
#
#     # switch to eval mode
#     model1.eval()
#     model.eval()
#     jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
#     depth_metric.reset()
#     for batch_i, (img, target, paths, shapes, task) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Valid"):
#         device = torch.device('cuda:0')  # 指定使用第一个 GPU
#         if not config.DEBUG:
#             img = img.to(device, non_blocking=True)
#             # 使用列表推导将每个张量移动到指定的设备
#             for idx, tensor in enumerate(target):
#                 if tensor is not None:
#                     target[idx] = tensor.to(device)
#             assign_target = []
#
#             # print(target)
#             for tgt in target:
#                 print(type(tgt), tgt.shape if tgt is not None else "None")
#                 if isinstance(tgt, torch.Tensor):
#                     assign_target.append(tgt.to(device))
#                 else:
#                     assign_target.append(None)
#
#             # img = img.to(device, non_blocking=True)
#             # assign_target = []
#             # for tgt in target:
#             #     assign_target.append(tgt.to(device))
#             # target = assign_target
#             nb, _, height, width = img.shape  # batch size, channel, height, width
#
#         with torch.no_grad():
#             pad_w, pad_h = shapes[0][1][1]
#             pad_w = int(pad_w)
#             pad_h = int(pad_h)
#             ratio = shapes[0][1][0][0]
#
#             t = time_synchronized()
#
#             img = img.to(device)
#
#             # input1, loss = model1(img)
#             # if config.TEST.PLOTS and batch_i <50 :
#             #     f = save_dir + '/' + f'pic_test_batch{batch_i}_labels.jpg'
#             #     save_raw_visualizations(img, input1,f)
#             #
#             # det_out, da_seg_out, ll_seg_out = model(input1, task)
#             det_out, da_seg_out, ll_seg_out = model(img, task)
#
#             t_inf = time_synchronized() - t
#
#             if batch_i > 0:
#                 T_inf.update(t_inf / img.size(0), img.size(0))
#
#             train_out = None
#             inf_out = None
#
#             task = task[0] if isinstance(task, list) else task
#             if task == "detect":
#                 inf_out, train_out = det_out
#
#             if task == "seg":
#                 # driving area segment evaluation
#                 # print(da_seg_out.size(),target[1].size())
#                 _, da_predict = torch.max(da_seg_out, 1)
#                 da_gt = target[1]
#                 # _,da_gt=torch.max(target[1], 1)
#                 da_predict = da_predict[:, pad_h:height - pad_h, pad_w:width - pad_w]
#                 da_gt = da_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
#
#                 da_metric.reset()
#                 da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
#                 da_acc = da_metric.pixelAccuracy()
#
#                 da_IoU = da_metric.IntersectionOverUnion()
#                 da_mIoU = da_metric.meanIntersectionOverUnion()
#
#                 da_acc_seg.update(da_acc, img.size(0))
#                 da_IoU_seg.update(da_IoU, img.size(0))
#                 da_mIoU_seg.update(da_mIoU, img.size(0))
#
#                 if config.TEST.PLOTS and batch_i <50 :
#                     f = save_dir + '/' + f'seg_test_batch{batch_i}_labels.jpg'
#                     save_segmentation_visualizations(img, da_seg_out, target[1], f)
#
#             if task == "depth":
#                 # lane line segment evaluation
#                 depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, depth_est = ll_seg_out
#                 print(depth_est.size(), target[2].size())
#                 # _, ll_predict = torch.max(depth_est, 1)
#                 # _, ll_gt = torch.max(target[2], 1)
#                 ll_predict = torch.squeeze(depth_est,dim=1)
#                 ll_gt = torch.squeeze(target[2],dim=1)
#                 # ll_gt = target[2]
#                 ll_predict = ll_predict[:, pad_h:height - pad_h, pad_w:width - pad_w]
#                 ll_gt = ll_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]
#
#
#                 depth_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
#                 depth_rmse = depth_metric.get_rmse()
#                 depth_mae = depth_metric.get_mae()
#                 depth_sqrel = depth_metric.get_sqrel()
#
#                 depth_rmse_eval.update( depth_rmse, img.size(0))
#                 depth_mae_eval.update( depth_mae, img.size(0))
#                 depth_sqrel_eval.update( depth_sqrel, img.size(0))
#
#                 # ll_metric.reset()
#                 # ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
#                 # ll_acc = ll_metric.lineAccuracy()
#                 # ll_IoU = ll_metric.IntersectionOverUnion()
#                 # ll_mIoU = ll_metric.meanIntersectionOverUnion()
#                 #
#                 # ll_acc_seg.update(ll_acc, img.size(0))
#                 # ll_IoU_seg.update(ll_IoU, img.size(0))
#                 # ll_mIoU_seg.update(ll_mIoU, img.size(0))
#                 if config.TEST.PLOTS and batch_i <50 :
#                     f = save_dir + '/' + f'depth_test_batch{batch_i}_labels.jpg'
#                     save_depth_visualizations(img, depth_est, target[2], f)
#
#             total_loss, head_losses = criterion((train_out,da_seg_out,ll_seg_out), target, shapes,model, img )   #Compute loss
#             losses.update(total_loss.item(), img.size(0))
#             #losses.update(0.1, img.size(0))
#
#             if task == "detect":
#                 # NMS
#                 t = time_synchronized()
#                 # target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
#                 lb = []  # for autolabelling
#                 output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRESHOLD,
#                                              iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
#                 # output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
#                 # output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRES, iou_thres=config.TEST.NMS_IOU_THRES)
#                 t_nms = time_synchronized() - t
#                 if batch_i > 0:
#                     T_nms.update(t_nms / img.size(0), img.size(0))
#
#         # Statistics per image
#         # output([xyxy,conf,cls])
#         # target[0] ([img_id,cls,xyxy])
#         if task == 'detect':
#             nlabel = (target[0].sum(dim=2) > 0).sum(dim=1)  # number of objects # [batch, num_gt ]
#             for si, pred in enumerate(output):
#                 # gt per image
#                 nl = int(nlabel[si])
#                 labels = target[0][si, :nl, 0:5]  # [ num_gt_per_image, [cx, cy, w, h] ]
#                 # gt_classes = target[0][si, :nl, 0]
#
#                 # labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image
#                 # nl = num_gt
#                 tcls = labels[:, 0].tolist() if nl else []  # target class
#                 path = Path(paths[si])
#                 seen += 1
#
#                 if len(pred) == 0:
#                     if nl:
#                         stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
#                     continue
#
#                 # Predictions
#                 predn = pred.clone()
#                 scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
#
#                 # Append to text file
#                 if config.TEST.SAVE_TXT:
#                     gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
#                     for *xyxy, conf, cls in predn.tolist():
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#
#                 # W&B logging
#                 if config.TEST.PLOTS and len(wandb_images) < log_imgs:
#                     box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
#                                  "class_id": int(cls),
#                                  "box_caption": "%s %.3f" % (names[cls], conf),
#                                  "scores": {"class_score": conf},
#                                  "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
#                     boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
#                     wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))
#
#                 # Append to pycocotools JSON dictionary
#                 if config.TEST.SAVE_JSON:
#                     # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
#                     image_id = int(path.stem) if path.stem.isnumeric() else path.stem
#                     box = xyxy2xywh(predn[:, :4])  # xywh
#                     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
#                     for p, b in zip(pred.tolist(), box.tolist()):
#                         jdict.append({'image_id': image_id,
#                                       'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
#                                       'bbox': [round(x, 3) for x in b],
#                                       'score': round(p[4], 5)})
#
#                 # Assign all predictions as incorrect
#                 correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
#                 if nl:
#                     detected = []  # target indices
#                     tcls_tensor = labels[:, 0]
#
#                     # target boxes
#                     tbox = xywh2xyxy(labels[:, 1:5])
#                     print(shapes[si][0])
#                     print(shapes[si][1])
#                     scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
#                     if config.TEST.PLOTS:
#                         confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))
#
#                     # Per target class
#                     for cls in torch.unique(tcls_tensor):
#                         ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
#                         pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
#
#                         # Search for detections
#                         if pi.shape[0]:
#                             # Prediction to target ious
#                             # n*m  n:pred  m:label
#                             ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
#                             # Append detections
#                             detected_set = set()
#                             for j in (ious > iouv[0]).nonzero(as_tuple=False):
#                                 d = ti[i[j]]  # detected target
#                                 if d.item() not in detected_set:
#                                     detected_set.add(d.item())
#                                     detected.append(d)
#                                     correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
#                                     if len(detected) == nl:  # all targets already located in image
#                                         break
#
#                 # Append statistics (correct, conf, pcls, tcls)
#                 stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
#
#             if config.TEST.PLOTS and batch_i < 55:
#                 f = save_dir + '/' + f'det_test_batch{batch_i}_labels.jpg'  # labels
#                 plot_images(img, target[0], paths, f, names)
#                 # Thread(target=plot_images, args=(img, target[0], paths, f, names), daemon=True).start()
#                 f = save_dir + '/' + f'det_test_batch{batch_i}_pred.jpg'  # predictions
#                 plot_images(img, output_to_target(output), paths, f, names)
#                 # Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()
#                 # 可视化原图、GT和预测框
#                 # mean = [0.485, 0.456, 0.406]
#                 # std = [0.229, 0.224, 0.225]
#                 #
#                 # img_vis = reverse_transform(img[si], mean, std)
#                 # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
#                 # h,w = shapes[si][0]
#                 # img_vis = cv2.resize(img_vis, (w,h))
#                 #
#                 # # 画GT框（绿色）
#                 # for l in labels:
#                 #     cls_id = int(l[0])
#                 #     box = l[1:5].clone().view(1, 4)
#                 #     box = xywh2xyxy(box)
#                 #     scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # 还原坐标
#                 #     x1, y1, x2, y2 = box[0].int().tolist()
#                 #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 #     cv2.putText(img_vis, f'GT:{cls_id}', (x1, y1 - 5), 0, 0.6, (0, 255, 0), 2)
#                 #
#                 # # 画预测框（红色）
#                 # for p in predn:
#                 #     x1, y1, x2, y2 = p[0:4].int().tolist()
#                 #     conf = p[4].item()
#                 #     cls_id = int(p[5].item())
#                 #     cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 #     cv2.putText(img_vis, f'Pred:{cls_id} {conf:.2f}', (x1, y2 + 15), 0, 0.5, (0, 0, 255), 2)
#                 #
#                 # save_dir = Path(save_dir)
#                 #
#                 # # 保存对比图
#                 # save_path = save_dir / 'vis' / f'{path.stem}.jpg'
#                 # save_path.parent.mkdir(parents=True, exist_ok=True)
#                 # cv2.imwrite(str(save_path), img_vis)
#
#     # Compute statistics
#     # stats : [[all_img_correct]...[all_img_tcls]]
#     # if task == 'detect':
#     stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip
#
#     map70 = None
#     map75 = None
#     if len(stats) and stats[0].any():
#         print("------------------------------------")
#         p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
#         ap50, ap70, ap75, ap = ap[:, 0], ap[:, 4], ap[:, 5], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
#         mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(), ap75.mean(), ap.mean()
#         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
#     else:
#         nt = torch.zeros(1)
#
#     # Print results
#     pf = '%20s' + '%12.3g' * 6  # print format
#     print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
#     # print(map70)
#     # print(map75)
#
#     # Print results per class
#     if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
#         for i, c in enumerate(ap_class):
#             print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
#
#     # Print speeds
#     t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
#     if not training:
#         print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
#
#     # Plots
#     if config.TEST.PLOTS:
#         confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
#         if wandb and wandb.run:
#             wandb.log({"Images": wandb_images})
#             wandb.log(
#                 {"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})
#
#     # Save JSON
#     if config.TEST.SAVE_JSON and len(jdict):
#         w = Path(
#             weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
#         anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
#         pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
#         print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
#         with open(pred_json, 'w') as f:
#             json.dump(jdict, f)
#
#         try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
#             from pycocotools.coco import COCO
#             from pycocotools.cocoeval import COCOeval
#
#             anno = COCO(anno_json)  # init annotations api
#             pred = anno.loadRes(pred_json)  # init predictions api
#             eval = COCOeval(anno, pred, 'bbox')
#             if is_coco:
#                 eval.params.imgIds = [int(Path(x).stem) for x in
#                                       val_loader.dataset.img_files]  # image IDs to evaluate
#             eval.evaluate()
#             eval.accumulate()
#             eval.summarize()
#             map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
#         except Exception as e:
#             print(f'pycocotools unable to run: {e}')
#
#     # Return results
#     if not training:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
#         print(f"Results saved to {save_dir}{s}")
#     model.float()  # for training
#     maps = np.zeros(nc) + map
#     da_segment_result = None
#     ll_segment_result = None
#
#     for i, c in enumerate(ap_class):
#         maps[c] = ap[i]
#
#     da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
#
#     ll_segment_result = (depth_rmse_eval.avg, depth_mae_eval.avg, depth_sqrel_eval.avg)
#
#     # print(da_segment_result)
#     # print(ll_segment_result)
#     detect_result = np.asarray([mp, mr, map50, map])
#     # print('mp:{},mr:{},map50:{},map:{}'.format(mp, mr, map50, map))
#     # print segmet_result
#     t = [T_inf.avg, T_nms.avg]
#     return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count if self.count != 0 else 0
#
#
# import os
# import torch
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def tensor_to_image(tensor):
#     """
#     把 [C, H, W] 的 tensor 转成 [H, W, 3] 的 numpy 图像
#     会自动归一化到 [0,1]
#     """
#     tensor = tensor.detach().cpu()
#     if tensor.dim() == 3:
#         if tensor.size(0) == 1:  # 单通道
#             img = tensor.squeeze(0).numpy()
#             img = np.stack([img] * 3, axis=-1)  # 变伪RGB
#         elif tensor.size(0) == 3:
#             img = tensor.permute(1, 2, 0).numpy()
#         else:
#             img = tensor[:3].permute(1, 2, 0).numpy()
#         img = (img - img.min()) / (img.max() - img.min() + 1e-5)
#         return img
#     else:
#         raise ValueError("tensor 应该是 [C, H, W] 维度")
#
#
# import os
# import numpy as np
# import torch
# import cv2
#
#
# def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#     if isinstance(tensor, torch.Tensor):
#         tensor = tensor.detach().cpu().numpy()
#     if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # C x H x W
#         tensor = tensor * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)
#         tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
#         tensor = tensor.transpose(1, 2, 0)  # -> H x W x C
#     return tensor
#
#
# def tensor_to_image(tensor, denormalize=False):
#     if isinstance(tensor, torch.Tensor):
#         tensor = tensor.detach().cpu().numpy()
#     if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # C x H x W
#         if denormalize:
#             tensor = tensor * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + \
#                      np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
#         tensor = tensor.transpose(1, 2, 0)  # H x W x C
#     if tensor.shape[2] == 1:
#         tensor = np.repeat(tensor, 3, axis=2)
#     tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
#     return tensor
#
#
# def save_image_pairs_with_denorm(imgs, preds, paths, save_dir, prefix="sample", denormalize=True):
#     os.makedirs(save_dir, exist_ok=True)
#     batch_size = imgs.size(0)
#
#     for i in range(batch_size):
#         image = tensor_to_image(imgs[i], denormalize=denormalize)
#         pred = tensor_to_image(preds[i], denormalize=denormalize)  # 一般预测图不用反归一化
#
#         # RGB -> BGR for OpenCV saving
#         image_bgr = image[:, :, ::-1]
#         pred_bgr = pred[:, :, ::-1]
#
#         concat = np.concatenate([image_bgr, pred_bgr], axis=1)
#
#         # 获取路径后缀标识
#         full_path = os.path.normpath(paths[i])
#         parts = full_path.split(os.sep)
#         suffix = "_".join(parts[-3:])
#
#         filename = f"{prefix}_{suffix}.png"
#         save_path = os.path.join(save_dir, filename)
#
#         cv2.imwrite(save_path, concat)
#
#
# def reverse_transform(img_tensor, mean, std):
#     """
#     将 PyTorch tensor 图像（3xHxW）还原为 uint8 格式的 RGB NumPy 图像（HxWx3）。
#
#     参数:
#         img_tensor: 输入图像 tensor（shape: [3, H, W]，值通常在归一化后的范围）
#         mean: 用于归一化的均值（列表或元组）
#         std: 用于归一化的标准差（列表或元组）
#
#     返回:
#         img_vis: 已反归一化并转换为 uint8 的 NumPy 图像，RGB 格式，HWC 排布。
#     """
#     assert isinstance(img_tensor, torch.Tensor), "输入必须是 PyTorch Tensor"
#     img = img_tensor.clone().detach().cpu()  # 安全复制
#
#     # 反归一化： x = x * std + mean
#     for t, m, s in zip(img, mean, std):
#         t.mul_(s).add_(m)
#
#     img = img.clamp(0, 1)  # 保证范围在 [0, 1]
#     img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
#     img_vis = (img * 255).astype(np.uint8)  # 转为 uint8 格式
#     return img_vis
