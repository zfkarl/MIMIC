# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
from losses import SupConLoss
from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        #print('samples type: ',type(samples))
        
        samples = torch.cat([samples[0], samples[1]], dim=0)
        
        samples = samples.to(device, non_blocking=True)
        

        bsz = targets.shape[0]
        #targets = torch.tensor(targets, dtype=torch.long)

        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        criterion2 = SupConLoss(temperature=0.07)
        
        with torch.cuda.amp.autocast():
            outputs, features = model(samples)
            #outputs= model(samples)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            #print('outputs:', outputs)
            #print('targets:', torch.cat([targets, targets], dim=0))
            #print('targets:', targets.shape)
            ##loss1 = criterion(outputs, torch.cat([targets, targets], dim=0)) ##soft
            loss1 = criterion(outputs, torch.cat([targets, targets], dim=0).squeeze().long()) ##hard
            #loss = criterion(outputs, targets)
            #loss2 = criterion2(features, torch.topk(targets, 1)[1].squeeze(1)) ##soft
            loss2 = criterion2(features, targets.squeeze(-1)) ##hard
            #print('loss2:',loss2.item())
            loss = loss1 + 0.1*loss2
            #loss = loss1

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        
        images = images.to(device, non_blocking=True)
        
        target = torch.tensor(target, dtype=torch.long)
        
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output,features = model(images)
            #output = model(images)
            #print('eval_target:',target.squeeze().shape)
            #print('eval_output:',output.shape)
            #loss = criterion(output, target)
            #print('output',output.shape)
            #print('target',target.squeeze().shape)
            if target.shape[0]!=1:
                target = target.squeeze()
            loss = criterion(output, target)

        acc1, acc2 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} '
          .format(top1=metric_logger.acc1, top2=metric_logger.acc2))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}