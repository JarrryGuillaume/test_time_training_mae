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
from typing import Iterable
import copy
import torch
import models_mae_shared
import os.path
import numpy as np
from scipy import stats
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
from torch.utils.data import Subset, DataLoader
import glob


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output is (B, classes)
    # target is (B)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def get_prameters_from_args(model, args):
    if args.finetune_mode == 'encoder':
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == 'all':
        parameters = model.parameters()
    elif args.finetune_mode == 'encoder_no_cls_no_msk':
        for name, p in model.named_parameters():
            if name.startswith('decoder') or name == 'cls_token' or name == 'mask_token':
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    return parameters


def _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device):
    if args.stored_latents:
        # We don't need to change the model, as it is never changed
        base_model.train(True)
        base_model.to(device)
        return base_model, base_optimizer, base_scalar
    clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    clone_model.train(True)
    clone_model.to(device)
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr, momentum=args.optimizer_momentum)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    else:
        assert args.optimizer_type == 'adam_w'
        optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    if args.load_loss_scalar:
        loss_scaler.load_state_dict(base_scalar.state_dict())
    return clone_model, optimizer, loss_scaler


import sys
import math
import os
import glob
import numpy as np
import torch

def train_on_test(base_model: torch.nn.Module,
                  base_optimizer,
                  base_scalar,
                  dataset_train,
                  dataset_val,
                  device: torch.device,
                  log_writer=None,
                  args=None,
                  num_classes: int = 1000,
                  iter_start: int = 0,
                ):

    # 1. Configure classifier depth and embedding sizes
    if args.model == 'mae_vit_small_patch16':
        classifier_depth = 8
        classifier_embed_dim = 512
        classifier_num_heads = 16
    else:
        assert ('mae_vit_huge_patch14' in args.model or args.model == 'mae_vit_large_patch16')
        classifier_embed_dim = 768
        classifier_depth = 12
        classifier_num_heads = 12

    clone_model = models_mae_shared.__dict__[args.model](
        num_classes=num_classes,
        head_type=args.head_type,
        norm_pix_loss=args.norm_pix_loss,
        classifier_depth=classifier_depth,
        classifier_embed_dim=classifier_embed_dim,
        classifier_num_heads=classifier_num_heads,
        rotation_prediction=False
    )

    # 2. Instead of re-initializing for partial saves, keep big containers in memory
    #    We'll store results for *the entire run*, i.e. across *all datapoints*.
    #    Each index i in all_results_global[i] or all_losses_global[i] corresponds
    #    to that step_per_example (i.e. 0 <= i < args.steps_per_example).
    all_results_global = [list() for _ in range(args.steps_per_example)]
    all_losses_global = [list() for _ in range(args.steps_per_example)]


    if args.shuffle_dataset: 
        subset_val = args.rng.permutation(args.max_iter)
        subset_train = np.array([[index * args.steps_per_example + i for i in range(args.steps_per_example)] for index in subset_val]).flatten()
        dataset_train = Subset(dataset_train, subset_train)
        dataset_val = Subset(dataset_val, subset_val)

    metric_logger = misc.MetricLogger(delimiter="  ")

    train_loader = iter(torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=args.shuffle_dataset, num_workers=args.num_workers, 
    ))
    val_loader = iter(torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=args.shuffle_dataset, num_workers=args.num_workers, 
    ))

    accum_iter = args.accum_iter
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # Reinitialize model
    model, optimizer, loss_scaler = _reinitialize_model(
        base_model, base_optimizer, base_scalar, clone_model, args, device
    )

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    dataset_len = min(len(dataset_val), args.max_iter)

    # 3. Main loop
    for data_iter_step in range(iter_start, dataset_len):
        val_data = next(val_loader)
        (test_samples, test_label) = val_data
        test_samples = test_samples.to(device, non_blocking=True)[0]
        test_label = test_label.to(device, non_blocking=True)

        # We do test-time training for args.steps_per_example * accum_iter steps
        for step_per_example in range(args.steps_per_example * accum_iter):
            train_data = next(train_loader)
            samples, _ = train_data
            samples = samples.to(device, non_blocking=True)[0]

            # Forward
            loss_dict, _, _, _ = model(samples, None, mask_ratio=args.mask_ratio)
            loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
            loss_value = loss.item()
            loss /= accum_iter

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Gradient update
            loss_scaler(
                loss, 
                optimizer, 
                parameters=model.parameters(),
                update_grad=((step_per_example + 1) % accum_iter == 0)
            )

            # Only log & zero_grad every accum_iter steps
            if (step_per_example + 1) % accum_iter == 0:
                if args.verbose:
                    print(f'datapoint {data_iter_step} iter {step_per_example}: rec_loss {loss_value}')

                # Append the reconstruction loss (div by accum_iter) to the losses container
                step_index = step_per_example // accum_iter
                all_losses_global[step_index].append(loss_value / accum_iter)

                optimizer.zero_grad()

            # Log metrics
            metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            # Test classification performance every accum_iter steps
            if (step_per_example + 1) % accum_iter == 0:
                with torch.no_grad():
                    model.eval()
                    all_pred = []
                    for _ in range(accum_iter):
                        loss_d, _, _, pred = model(
                            test_samples,
                            test_label,
                            mask_ratio=0,
                            reconstruct=False
                        )
                        if args.verbose:
                            cls_loss = loss_d['classification'].item()
                            print(f'datapoint {data_iter_step} iter {step_per_example}: class_loss {cls_loss}')
                        all_pred.extend(pred.argmax(dim=1).cpu().numpy())

                    if len(all_pred) > 0:
                        # majority vote
                        acc1 = (stats.mode(all_pred).mode == test_label[0].cpu().item()) * 100.
                    else:
                        acc1 = 0.
                        print("There were no predictions done")

                    # Keep track of the accuracy for that step
                    all_results_global[step_index].append(acc1)

                    if step_index + 1 == args.steps_per_example:
                        metric_logger.update(top1_acc=acc1)
                        metric_logger.update(loss=loss_value)

                    # Return to training mode
                    model.train()

        # Optional: print progress every N steps
        if data_iter_step % 50 == 1:
            print('step: {}, acc {:.2f}, rec-loss {:.4f}'.format(
                data_iter_step, np.mean(all_results_global[-1]), loss_value)
            )

        # IMPORTANT: do not reset all_results / all_losses here.
        # Reinitialize the model after each data point
        if not args.online: 
            model, optimizer, loss_scaler = _reinitialize_model(
                base_model, base_optimizer, base_scalar, clone_model, args, device
            )

    # 4. Now that the loop is done, do a single final save in memory once
    #    Save both arrays to disk
    online = "online" if args.online else "offline"
    shuffle = "shuffled" if args.shuffle_dataset else ""
    with open(os.path.join(args.output_dir, f'final_results_{dataset_len}_{online}_{args.steps_per_example}_{shuffle}.npy'), 'wb') as f:
        np.save(f, np.array(all_results_global, dtype=object))

    with open(os.path.join(args.output_dir, f'final_losses_{dataset_len}_{online}_{args.steps_per_example}_{shuffle}.npy'), 'wb') as f:
        np.save(f, np.array(all_losses_global, dtype=object))

    # 5. Optionally compute final metrics and write them to a file.
    #    We can replace the old "save_accuracy_results(args)" with a new helper that
    #    just takes the final in-memory arrays:
    save_accuracy_results_in_memory(
        args,
        all_results_global
    )

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
  


def save_accuracy_results_in_memory(args, all_results_global):
    """
    Example function to compute final accuracy from your in-memory results
    and dump to disk. You no longer need to read multiple results_*.npy files.
    """
    # For each test-time-training step (0..args.steps_per_example-1),
    # compute the global average:
    with open(os.path.join(args.output_dir, 'model-final.pth'), 'w') as f:
        f.write('Done!\n')

    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        for step_i, step_results in enumerate(all_results_global):
            step_acc = np.mean(step_results) if len(step_results) > 0 else 0.0
            f.write(f'Step {step_i}\t{step_acc:.2f}\n')