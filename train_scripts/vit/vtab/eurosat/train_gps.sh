gpu_id=5
bz=16
lr=0.002

python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/eurosat \
    --dataset eurosat \
    --num-classes 10 --direct-resize --no-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz \
    --opt adam  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	  --mixup 0 --cutmix 0 --smoothing 0 \
    --output output/ \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 15 \
    --gpu_id $gpu_id \
    --log-wandb \
    --experiment vtab \
    --no-prefetcher 

# python train_gps.py path/to/vtab-1k/eurosat \
#     --dataset eurosat \
#     --num-classes 10 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz --validation-batch-size 128 \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 1 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastive

