# tvl encoder
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 ./tvl_enc/main_pretrain.py \
--batch_size 256 --epochs 400 --warmup_epochs 10 --weight_decay 0.05 --active_modality_names  tactile text --find_unused_parameters --multi_epochs_dataloader \
--log_name tvl_vittiny_tactile_encoder --num_workers 0  --tactile_model vit_tiny_patch16_224 --blr 3e-4 \
--json_file tvl_caption.json --use_tac_text_loss --disable_vision_text_loss 
# --enable_flash_attention2