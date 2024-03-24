from __future__ import print_function, absolute_import, division

import torch

from models_baseline.mlp.linear_model import init_weights
from models_poseaug.PosDiscriminator import Pos2dDiscriminator, Pos3dDiscriminator
from models_poseaug.gan_generator import PoseGenerator
from utils.utils import get_scheduler


def get_poseaug_model(args, dataset):
    """
    return PoseAug augmentor and discriminator
    and corresponding optimizer and scheduler
    """
    # Create model: G and D
    print("==> Creating model...")
    device = torch.device("cuda")
    num_joints = dataset.skeleton().num_joints()

    # generator for PoseAug
    model_G_w = PoseGenerator(args, num_joints * 3).to(device)
    model_G_w.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_G_w.parameters()) / 1000000.0))

    model_G_s = PoseGenerator(args, num_joints * 3).to(device)
    model_G_s.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_G_s.parameters()) / 1000000.0))

    # discriminator for 3D
    model_d3d_w = Pos3dDiscriminator(num_joints).to(device)
    model_d3d_w.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d3d_w.parameters()) / 1000000.0))

    model_d3d_s = Pos3dDiscriminator(num_joints).to(device)
    model_d3d_s.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d3d_s.parameters()) / 1000000.0))

    # discriminator for 2D
    model_d2d_w = Pos2dDiscriminator(num_joints).to(device)
    model_d2d_w.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d2d_w.parameters()) / 1000000.0))

    model_d2d_s = Pos2dDiscriminator(num_joints).to(device)
    model_d2d_s.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d2d_s.parameters()) / 1000000.0))

    # prepare optimizer
    g_optimizer_w = torch.optim.Adam(model_G_w.parameters(), lr=args.lr_g)
    d3d_optimizer_w = torch.optim.Adam(model_d3d_w.parameters(), lr=args.lr_d)
    d2d_optimizer_w = torch.optim.Adam(model_d2d_w.parameters(), lr=args.lr_d)

    g_optimizer_s = torch.optim.Adam(model_G_s.parameters(), lr=args.lr_g)
    d3d_optimizer_s = torch.optim.Adam(model_d3d_s.parameters(), lr=args.lr_d)
    d2d_optimizer_s = torch.optim.Adam(model_d2d_s.parameters(), lr=args.lr_d)

    # prepare scheduler
    g_lr_scheduler_w = get_scheduler(g_optimizer_w, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d3d_lr_scheduler_w = get_scheduler(d3d_optimizer_w, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d2d_lr_scheduler_w = get_scheduler(d2d_optimizer_w, policy='lambda', nepoch_fix=0, nepoch=args.epochs)

    g_lr_scheduler_s = get_scheduler(g_optimizer_s, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d3d_lr_scheduler_s = get_scheduler(d3d_optimizer_s, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d2d_lr_scheduler_s = get_scheduler(d2d_optimizer_s, policy='lambda', nepoch_fix=0, nepoch=args.epochs)

    return {
        'model_G_weak': model_G_w,
        'model_d3d_weak': model_d3d_w,
        'model_d2d_weak': model_d2d_w,
        'optimizer_G_weak': g_optimizer_w,
        'optimizer_d3d_weak': d3d_optimizer_w,
        'optimizer_d2d_weak': d2d_optimizer_w,
        'scheduler_G_weak': g_lr_scheduler_w,
        'scheduler_d3d_weak': d3d_lr_scheduler_w,
        'scheduler_d2d_weak': d2d_lr_scheduler_w,
        'model_G_strong': model_G_s,
        'model_d3d_strong': model_d3d_s,
        'model_d2d_strong': model_d2d_s,
        'optimizer_G_strong': g_optimizer_s,
        'optimizer_d3d_strong': d3d_optimizer_s,
        'optimizer_d2d_strong': d2d_optimizer_s,
        'scheduler_G_strong': g_lr_scheduler_s,
        'scheduler_d3d_strong': d3d_lr_scheduler_s,
        'scheduler_d2d_strong': d2d_lr_scheduler_s,
    }
