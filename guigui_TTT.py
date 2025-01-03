import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import signal

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm
from torchvision import datasets
import glob
import util.misc as misc
import models_mae_shared
from engine_test_time import train_on_test, get_prameters_from_args
from data import tt_image_folder
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from .args_TTT import *


def load_combined_model(args, num_classes: int = 1000):
    if args.model == 'mae_vit_small_patch16':
        args.classifier_depth = 8
        args.classifier_embed_dim = 512
        args.classifier_num_heads = 16
    else:
        assert 'mae_vit_large_patch16' in args.model or 'mae_vit_huge_patch14' in args.model
        args.classifier_embed_dim = 768
        args.classifier_depth = 12
        args.classifier_num_heads = 12
    
    model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type, norm_pix_loss=args.norm_pix_loss, 
                                                   classifier_depth=args.classifier_depth, classifier_embed_dim=args.classifier_embed_dim, 
                                                   classifier_num_heads=args.classifier_num_heads,
                                                   rotation_prediction=False)
    model_checkpoint = torch.load(args.resume_model, map_location='cpu')
    head_checkpoint = torch.load(args.resume_finetune, map_location='cpu')
    
  
    assert args.classifier_depth != 0, 'Please provide classifier_depth parameter.'
    for key in head_checkpoint['model']:
        if key.startswith('classifier'):
            model_checkpoint['model'][key] = head_checkpoint['model'][key]

    model.load_state_dict(model_checkpoint['model'])
    optimizer = None
    if args.load_loss_scalar:
        loss_scaler = NativeScaler()
        loss_scaler.load_state_dict(model_checkpoint['scaler'])
    else:
        loss_scaler = None
    return model, optimizer, loss_scaler


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    is_available = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{is_available} used for test-time-training")

    device = torch.device(is_available)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    max_known_file = max([int(i.split('results_')[-1].split('.npy')[0]) for i in glob.glob(os.path.join(args.output_dir, 'results_*.npy'))] + [-1])
    if max_known_file != -1:
        print(f'Found {max_known_file} values, continues from next iterations.')
        
    # simple augmentation    
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if not args.single_crop:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

    dataset_train = tt_image_folder.ExtendedImageFolder(args.data_path, transform=transform_train, minimizer=None, 
                                                        batch_size=args.batch_size, steps_per_example=args.steps_per_example * args.accum_iter, 
                                                        single_crop=args.single_crop, start_index=max_known_file+1)

    dataset_val = tt_image_folder.ExtendedImageFolder(args.data_path, transform=transform_val, 
                                                        batch_size=1, minimizer=None, 
                                                        single_crop=args.single_crop, start_index=max_known_file+1)

    num_classes = 1000

    # define the model
    model, optimizer, scalar = load_combined_model(args, num_classes)
         
    print("Model = %s" % str(model))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    wandb_config = vars(args)
    base_lr = (args.lr * 256 / eff_batch_size)
    wandb_config['base_lr'] = base_lr
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    start_time = time.time()
    test_stats = train_on_test(
        model, optimizer, scalar, dataset_train, dataset_val,
        device,
        log_writer=None,
        args=args,
        num_classes=num_classes,
        iter_start=max_known_file+1
    )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
