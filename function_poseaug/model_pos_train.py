from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn

from progress.bar import Bar
from utils.utils import AverageMeter, set_grad


def train_posenet(model_pos, data_loader, optimizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    set_grad([model_pos], True)
    model_pos.train()
    end = time.time()


    bar = Bar('Train posenet', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        # here avoid bn with one sample in last batch, skip if num_poses=1
        if num_poses == 1:
            break

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return


def train_posenet_meta(model_pos, data_dict, optimizer, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    set_grad([model_pos], True)
    model_pos.train()
    end = time.time()


    bar = Bar('Train posenet', max=len(data_dict['train_det2d3d_loader']))
    #+for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):
    for i, ((weak_3d, weak_2d, _, _), (strong_3d, strong_2d, _, _), (targets_3d, inputs_2d, _, _)) in enumerate(
            zip(data_dict['train_weak2d3d_loader'], data_dict['train_strong2d3d_loader'], data_dict['train_det2d3d_loader'])):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        # here avoid bn with one sample in last batch, skip if num_poses=1
        if num_poses == 1:
            break

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        weak_3d, weak_2d = weak_3d.to(device), weak_2d.to(device)
        weak_3d = weak_3d[:, :, :] - weak_3d[:, :1, :]  # the output is relative to the 0 joint

        weak_3d, weak_2d = weak_3d.to(device), weak_2d.to(device)
        weak_3d = weak_3d[:, :, :] - weak_3d[:, :1, :]  # the output is relative to the 0 joint


        params = list(model.parameters())
        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        grads = torch.autograd.grad(loss_3d_pos, params, create_graph=True)
        params = [(param - 1e-4 * grad).requires_grad_() for param, grad in zip(params, grads)]

        weak_pred = model_pos(weak_2d)
        loss_b = criterion(weak_pred, weak_3d)

        loss_combine = (loss_3d_pos + loss_b) / 2
        optimizer.zero_grad()
        loss_combine.backward()
        
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()


        params_new = list(model.parameters())
        weak_pred_new = model_pos(weak_2d)

        optimizer.zero_grad()
        loss_3d_pos_new = criterion(weak_pred_new, weak_3d)
        grads_new = torch.autograd.grad(loss_3d_pos_new, params, create_graph=True)
        params_new = [(param_new - 1e-4 * grad_new).requires_grad_() for param_new, grad_new in zip(params_new, grads_new)]

        strong_pred = model_pos(strong_2d)
        loss_b_new = criterion(strong_pred, strong_3d)

        loss_combine_new = (loss_3d_pos_new + loss_b_new) / 2
        optimizer.zero_grad()
        loss_combine_new.backward()
        
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step() 
        
        
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return