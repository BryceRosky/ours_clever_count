gpu_id=0
bz=16
lr=0.005
# 定义学习率列表

lr=6e-4
warmup_lr=5e-4
# learning_rates=("6e-4" "5e-5" "6e-5" "1e-4")
# # 定义 warmup 学习率列表
# warmup_lrs=("5e-4" "1e-3"  "2e-3" "4e-3")  # 根据需求可以修改或增加更多 warmup 学习率

keep_rates=("0.95" "0.9" "0.8" "0.5")
for keep_rate in "${keep_rates[@]}"; do
    echo "Training with token: $keep_rate"
    python train_gps.py /home/yhr/GPS1/data/vtab/vtab-1k/sun397 \
        --dataset sun397 \
        --num-classes 397 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz \
        --opt adam --weight-decay 0 \
        --warmup-lr $warmup_lr --warmup-epochs 10 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output sun_output/noinit_sun_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --keep_rate 1 --top_param 0.03  --head_reparam --init_lr
done

        # --output 5811_output/${keep_rate}_sun_output/ \
# # 外层循环遍历每个学习率
# for lr in "${learning_rates[@]}"; do
#     # 输出当前学习率，便于调试
#     echo "Training with learning rate: $lr"
    
#     # 内层循环遍历每个 warmup 学习率
#     for warmup_lr in "${warmup_lrs[@]}"; do
#         # 输出当前 warmup 学习率，便于调试
#         echo "Training with warmup learning rate: $warmup_lr"
        
#         # 运行训练脚本，传递当前的学习率和 warmup 学习率
#         python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/sun397 \
#             --dataset sun397 \
#             --num-classes 397 --direct-resize --no-aug \
#             --model vit_base_patch16_224_in21k \
#             --epochs 100 \
#             --batch-size $bz \
#             --opt adam --weight-decay 0 \
#             --warmup-lr $warmup_lr --warmup-epochs 10 \
#             --lr $lr --min-lr 1e-8 \
#             --drop-path 0 --img-size 224 \
#             --mixup 0 --cutmix 0 --smoothing 0 \
#             --output best_output/0.95_sun_${lr}_${warmup_lr}_output/ \
#             --amp --tuning-mode part --pretrained \
#             --pruning --pruning_method gradient_perCell \
#             --times_para 15 \
#             --gpu_id $gpu_id \
#             --log-wandb \
#             --experiment vtab \
#             --no-prefetcher --keep_rate 0.95 --top_param 0.03  --head_reparam
#     done
# done

echo "Training completed for all learning rates and warmup learning rates."
# python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/sun397 \
#     --dataset sun397 \
#     --num-classes 397 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0 --img-size 224 \
# 	  --mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 15 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --no-prefetcher 

# python train_gps.py path/to/vtab-1k/sun397 \
#     --dataset sun397 \
#     --num-classes 397 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz \
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


