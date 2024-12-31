import models_mae_shared
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import os 
from util.pos_embed import interpolate_pos_embed

num_classes = 1000

finetune_path = "Drive/MyDrive/"

model_args = 'mae_vit_small_patch16'
head_type = "vit_head"
classifier_depth = 12
norm_pix_loss = 'store_true'
drop_path = 0
input_size = 224
batch_size = 64
num_workers = 8
blr = 0.1
pin_mem = 'store_true'

eff_batch_size = batch_size * args.accum_iter * misc.get_world_size()
    
if args.lr is None:  # only base_lr is specified
    args.lr = args.blr * eff_batch_size / 256

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models_mae_shared.mae_vit_base_patch16(
        num_classes=num_classes,
        img_size=input_size,
        no_decoder=True,
        head_type=head_type,
        classifier_depth=classifier_depth,
        norm_pix_loss=norm_pix_loss,
        drop_path_rate=drop_path,
)


transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)

criterion = torch.nn.CrossEntropyLoss()

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)


data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )

data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False
    )

checkpoint = torch.load(finetune_path, map_location='cpu')

print("Load pre-trained checkpoint from: %s" % finetune)
checkpoint_model = checkpoint['model']
# for k in list(checkpoint_model.keys()):
#     if k.startswith('decoder') or k == 'mask_token':
#         print(f"Removing key {k} from pretrained checkpoint")
#         del checkpoint_model[k]

# interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)


for name, p in model.named_parameters():
    if not name.startswith('classifier'):
        p.requires_grad = False

model.to(device)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

parameters = [p[1] for p in model_without_ddp.named_parameters() if p[0].startswith('classifier')]
optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)