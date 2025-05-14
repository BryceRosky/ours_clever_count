gpu_id=4
bz=16
lr=0.001

python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/patch_camelyon \
    --dataset patch_camelyon \
    --num-classes 2 --direct-resize --no-aug \
    --model vit_base_patch16_224_in21k \
    --epochs 100 \
    --batch-size $bz --validation-batch-size 128 \
    --opt adam  --weight-decay 0 \
    --warmup-lr 1e-7 --warmup-epochs 10 \
    --lr $lr --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	  --mixup 0 --cutmix 0 --smoothing 0 \
    --output best_output/0.95_0.03_camelyon_seperate_nohead_1e-3_1e_3 \
    --amp --tuning-mode part --pretrained \
    --pruning --pruning_method gradient_perCell \
    --times_para 15 \
    --gpu_id $gpu_id \
    --log-wandb \
    --experiment vtab \
    --no-prefetcher --keep_rate 0.95 --top_param 0.03 
    #--head_reparam
    # --head_reparam
    # 


# python train_gps.py path/to/vtab-1k/patch_camelyon \
#     --dataset patch_camelyon \
#     --num-classes 2 --direct-resize --no-aug \
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
#     --times_para 2 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastive



