
data_path = "/content/drive/MyDrive/Colab Notebooks/blur_subset"
print_freq = 50
finetune_mode = 'encoder'
model = 'mae_vit_large_patch16'
input_size = 224
classifier_depth = 0
mask_ratio = 0.75
steps_per_example = 1
stored_latents = ''
weight_decay = 0.05
blr = 1e-3
batch_size = 256
data_path = ''
dataset_name = 'imagenet_c'
output_dir = './output_dir'
log_dir = './output_dir'
device = 'cuda'
accum_iter = 1
load_loss_scalar = False
optimizer_type = 'sgd'
optimizer_momentum = 0.9
seed = 0
resume_model = ''
resume_finetune = ''
num_workers = 10
pin_mem = False
norm_pix_loss = False
verbose = False
head_type = 'vit_head'
single_crop = False
world_size = 1
local_rank = -1
dist_on_itp = False
dist_url = 'env://'
