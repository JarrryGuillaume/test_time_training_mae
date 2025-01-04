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
import glob

def train_on_test(base_model: torch.nn.Module,
                  base_optimizer,
                  base_scalar,
                  dataset_train, dataset_val,
                  device: torch.device,
                  log_writer=None,
                  args=None,
                  num_classes: int = 1000, 
                  iter_start: int = 0):
    if args.model == 'mae_vit_small_patch16':
        classifier_depth = 8
        classifier_embed_dim = 512
        classifier_num_heads = 16
    else: 
        assert ('mae_vit_huge_patch14' in args.model or args.model == 'mae_vit_large_patch16') 
        classifier_embed_dim = 768
        classifier_depth = 12
        classifier_num_heads = 12
    clone_model = models_mae_shared.__dict__[args.model](num_classes=num_classes, head_type=args.head_type, 
                                                         norm_pix_loss=args.norm_pix_loss, 
                                                         classifier_depth=classifier_depth, classifier_embed_dim=classifier_embed_dim, 
                                                         classifier_num_heads=classifier_num_heads,
                                                         rotation_prediction=False)
    # Intialize the model for the current run
    all_results = [list() for i in range(args.steps_per_example)]
    all_losses =  [list() for i in range(args.steps_per_example)]
    metric_logger = misc.MetricLogger(delimiter="  ")
    train_loader = iter(torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=args.num_workers))
    val_loader = iter(torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers))
    accum_iter = args.accum_iter
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    dataset_len = len(dataset_val)
    for data_iter_step in range(iter_start, dataset_len):
        val_data = next(val_loader)
        (test_samples, test_label) = val_data
        test_samples = test_samples.to(device, non_blocking=True)[0]
        test_label = test_label.to(device, non_blocking=True)
        pseudo_labels = None
        # Test time training:
        for step_per_example in range(args.steps_per_example * accum_iter):
            train_data = next(train_loader)
            # Train data are 2 values [image, class]
            mask_ratio = args.mask_ratio
            samples, _ = train_data
            targets_rot, samples_rot = None, None
            samples = samples.to(device, non_blocking=True)[0] # index [0] becuase the data is batched to have size 1.
            loss_dict, _, _, _ = model(samples, None, mask_ratio=mask_ratio)
            loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
            loss_value = loss.item()
            loss /= accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(step_per_example + 1) % accum_iter == 0)
            if (step_per_example + 1) % accum_iter == 0:
                if args.verbose:
                    print(f'datapoint {data_iter_step} iter {step_per_example}: rec_loss {loss_value}')
                
                all_losses[step_per_example // accum_iter].append(loss_value/accum_iter)
                optimizer.zero_grad()
                    
            metric_logger.update(**{k:v.item() for k,v in loss_dict.items()})
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            # Test:
            if (step_per_example + 1) % accum_iter == 0:
                with torch.no_grad():
                    model.eval()
                    all_pred = []
                    for _ in range(accum_iter):
                        loss_d, _, _, pred = model(test_samples, test_label, mask_ratio=0, reconstruct=False)
                        if args.verbose:
                            cls_loss = loss_d['classification'].item()
                            print(f'datapoint {data_iter_step} iter {step_per_example}: class_loss {cls_loss}')
                        all_pred.extend(list(pred.argmax(axis=1).detach().cpu().numpy()))
                    acc1 = (stats.mode(all_pred).mode[0] == test_label[0].cpu().detach().numpy()) * 100.
                    if (step_per_example + 1) // accum_iter == args.steps_per_example:
                        metric_logger.update(top1_acc=acc1)
                        metric_logger.update(loss=loss_value)
                    all_results[step_per_example // accum_iter].append(acc1)
                    model.train()
        if data_iter_step % 50 == 1:
            print('step: {}, acc {} rec-loss {}'.format(data_iter_step, np.mean(all_results[-1]), loss_value))
        if data_iter_step % 500 == 499 or (data_iter_step == dataset_len - 1):
            with open(os.path.join(args.output_dir, f'results_{data_iter_step}.npy'), 'wb') as f:
                np.save(f, np.array(all_results))
            with open(os.path.join(args.output_dir, f'losses_{data_iter_step}.npy'), 'wb') as f:
                np.save(f, np.array(all_losses))
            all_results = [list() for i in range(args.steps_per_example)]
            all_losses = [list() for i in range(args.steps_per_example)]
        model, optimizer, loss_scaler = _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device)
    save_accuracy_results(args)
    # gather the stats from all processes
    try:
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    except:
        pass
    return 


def save_accuracy_results(args):
    all_all_results = [list() for i in range(args.steps_per_example)]
    for file_number, f_name in enumerate(glob.glob(os.path.join(args.output_dir, 'results_*.npy'))):
        all_data = np.load(f_name)
        for step in range(args.steps_per_example):
            all_all_results[step] += all_data[step].tolist()
    with open(os.path.join(args.output_dir, 'model-final.pth'), 'w') as f:
        f.write(f'Done!\n')
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        for i in range(args.steps_per_example):
            assert len(all_all_results[i]) == 50000, len(all_all_results[i])
            f.write(f'{i}\t{np.mean(all_all_results[i])}\n')