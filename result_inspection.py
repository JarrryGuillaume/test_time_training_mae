
import torch
import torchvision.transforms as transforms

import util.misc as misc
import models_mae_shared
from data import tt_image_folder

import numpy as np 
import matplotlib.pyplot as plt

import test_time_training_mae.models_mae_shared as models_mae_shared
from test_time_training_mae.engine_test_time import _reinitialize_model
from test_time_training_mae.guigui_TTT import load_combined_model

def get_clone_model(args, num_classes=1000):
    is_available = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{is_available} used for test-time-training")

    device = torch.device(is_available)

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

    return clone_model


def load_dataset(args): 
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
                                                        single_crop=args.single_crop, start_index=0)

    dataset_val = tt_image_folder.ExtendedImageFolder(args.data_path, transform=transform_val, 
                                                        batch_size=1, minimizer=None, 
                                                        single_crop=args.single_crop, start_index=0)

    return dataset_train, dataset_val

def get_image(chosen_image, dataset_train, dataset_val, args):
    test_image = dataset_val[chosen_image]
    train_images = [dataset_train[chosen_image * args.steps_per_example + i] for i in range(args.steps_per_example)]
    return test_image, train_images

def plot_TTT(base_model: torch.nn.Module,
             base_optimizer,
             base_scaler,
             test_image,
             train_images,
            device: torch.device,
            log_writer=None,
            args=None,
            num_classes: int = 1000,
            iter_start: int = 0):

    clone_model = get_clone_model(args, num_classes)
    model, optimizer, loss_scaler = _reinitialize_model(
        base_model, base_optimizer, base_scaler, clone_model, args, device
    )
    accum_iter = args.accum_iter

    train_images = iter(train_images)

    all_losses = []
    predictions = []
    (test_samples, test_label) = test_image
    test_samples = test_samples.to(device, non_blocking=True)[0]
    test_label = torch.tensor(test_label).to(device, non_blocking=True)

    for step_per_example in range(args.steps_per_example * accum_iter):
            train_data = next(train_images)
            samples, _ = train_data
            samples = samples.to(device, non_blocking=True)

            # Forward
            loss_dict, pred, latent, head = model(samples, None, mask_ratio=args.mask_ratio)
            loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
            loss_value = loss.item()
            loss /= accum_iter

            # Gradient update
            loss_scaler(
                loss, 
                optimizer, 
                parameters=model.parameters(),
                update_grad=((step_per_example + 1) % accum_iter == 0)
            )

            if (step_per_example + 1) % accum_iter == 0:
                all_losses.append(loss_value / accum_iter)

                if args.verbose:
                    print(f'datapoint {step_per_example} iter {step_per_example}: rec_loss {loss_value}')

                optimizer.zero_grad()

            predictions.append(pred)
    
    return predictions, all_losses

def load_statistics(dataset_name, n_samples, comp_folder=""): 
    final_results = np.load(f"{comp_folder}output_mae/{dataset_name}/final_results_{n_samples}.npy", allow_pickle=True)
    final_losses = np.load(f"{comp_folder}output_mae/{dataset_name}/final_losses_{n_samples}.npy", allow_pickle=True)
    return final_results, final_losses

def plot_statistics(dataset_name, n_samples, comp_folder=""):
    final_results, final_losses = load_statistics(dataset_name, n_samples, comp_folder=comp_folder)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  
    fig.suptitle(f"{dataset_name} TTT results")

    mean_accuracy = np.mean(final_results, axis=1)
    mean_reconstruction_loss = np.mean(final_losses, axis=1)

    indices = np.arange(len(final_results))
    ax1.bar(indices, mean_accuracy, color='blue', label='Final Results')
    ax1.set_title("Mean Accuracy")
    ax1.set_xlabel("Nth step")
    ax1.set_ylabel("Values")
    ax1.set_xticks(indices)

    ax2.bar(indices, mean_reconstruction_loss, color='orange', label='Final Losses')
    ax2.set_title("Average reconstruction loss")
    ax2.set_xlabel("N step")
    ax2.set_ylabel("Values")
    ax2.set_xticks(indices)

    plt.tight_layout()
    plt.show()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    denorm = tensor.clone()
    
    for c, (m, s) in enumerate(zip(mean, std)):
        denorm[c] = denorm[c] * s + m
        
    denorm = torch.clamp(denorm, 0.0, 1.0)
    return denorm.permute(1, 2, 0).cpu().numpy()