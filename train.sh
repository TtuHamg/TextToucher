TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master-port=11113 \
    tools/train.py --config configs/TextToucher_img512_HCT.py




