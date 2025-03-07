data_root = "/Datasets/tvl_dataset"
data = dict(
    type="TVLData",
    root="hct",
    json_file="hct_caption.json",
    transform="default_train",
    load_vae_feat=True,
    prompt_clear=True,
)

image_size = 512
train_batch_size = 32
eval_batch_size = 16
use_fsdp = False
valid_num = 0

# model setting
window_block_indexes = []  
window_size = 0  
use_rel_pos = False  
model = "TextToucher_XL_2"
fp32_attention = True
aspect_ratio_type = None  
multi_scale = False
lewei_scale = 1.0
# training setting
num_workers = 4
train_sampling_steps = 1000
eval_sampling_steps = 250
model_max_length = 120

num_epochs = 200
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01  
gc_step = 1  
auto_lr = dict(rule="sqrt")  

optimizer = dict(type="AdamW", lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule = "constant"
lr_schedule_args = dict(num_warmup_steps=1000)

save_image_epochs = 1
save_model_epochs = 1
save_model_steps = 1000000

mixed_precision = "fp16"  # set for accelerator
scale_factor = 0.18215
ema_rate = 0.9999
log_interval = 50
cfg_scale = 4

# vae_pretrained = ""
load_from = "pretrained_model/PixArt-XL-2-512x512.pth"
resume_from = dict(
    checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True
)
snr_loss = False

work_dir = "results/hct"
seed = 43

bg_embed = 3
bg_token_num = 4
bg_by_time = 600
bg_layer = True
resample = False
cond_mechanism="cross_attention"