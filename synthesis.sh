# synthesis hct dataset
python ./tools/synthesis.py  \
--model_path '/TextToucher_path/results/hct/checkpoints/model.pth' \
--vae_path '/TextToucher_path/pretrained_model/sd-vae-ft-ema'
--save_dir '/TextToucher_path/results/hct/samples/dpm50_seed43_cfg4.5' \
--device 'cuda:0' --vae_device 'cuda:0' \
--eval_prompts 'hct_eval/prompt_embed.ckpt' \ 
--gt_dir 'hct_eval/tactile/' \
--sampling_algo 'dpm-solver' \
--bg_idx  'hct_eval/bg_idx.ckpt' \ # the gel status idx
--batch_size 25 \
--config '/TextToucher_path/configs/TextToucher_img512_HCT.py' \
--seed 43  --cfg 4.5  --steps 50 
