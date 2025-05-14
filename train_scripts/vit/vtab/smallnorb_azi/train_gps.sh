gpu_id=1
bz=32
lr=0.008



lr=3e-5
warmup_lr=3e-3
# learning_rates=("6e-4" "5e-5" "6e-5" "1e-4")
# # 定义 warmup 学习率列表
# warmup_lrs=("5e-4" "1e-3"  "2e-3" "4e-3")  # 根据需求可以修改或增加更多 warmup 学习率

keep_rates=("0.95" "0.9" "0.8" "0.5")
for keep_rate in "${keep_rates[@]}"; do
    echo "Training with token: $keep_rate"
    python train_gps.py /home/yhr/GPS1/data/vtab/vtab-1k/smallnorb_azi \
        --dataset smallnorb_azi \
        --num-classes 18 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz --validation-batch-size 128 \
        --opt adam --weight-decay 0 \
        --warmup-lr $warmup_lr --warmup-epochs 20 \
        --lr $lr --min-lr 1e-8 \
        --drop-path 0 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output token_output/${keep_rate}_azi_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --keep_rate $keep_rate --top_param 0.03  --head_reparam
done


# # # 定义学习率和 warmup-lr 的范围
# # learning_rates=("5e-3" "2e-4" "3e-5" "1e-3" "1e-5")
# # warmup_lrs=("3e-3" "4e-3" "1e-3" "2e-3"  "2e-4" "5e-5")  # 你可以根据需求设置 warmup-lr 的值
# learning_rates=("3e-5")
# warmup_lrs=("3e-3")
# # 外部循环遍历每个 warmup-lr
# for warmup_lr in "${warmup_lrs[@]}"; do
#     # 输出当前 warmup-lr，便于调试
#     echo "Training with warmup-lr: $warmup_lr"
    
#     # 内部循环遍历每个学习率
#     for lr in "${learning_rates[@]}"; do
#         # 输出当前学习率，便于调试
#         echo "Training with learning rate: $lr"
        
#         # 运行训练脚本，传递当前的学习率和 warmup-lr
#         python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/smallnorb_azi \
#             --dataset smallnorb_azi \
#             --num-classes 18 --direct-resize --no-aug \
#             --model vit_base_patch16_224_in21k \
#             --epochs 100 \
#             --batch-size $bz --validation-batch-size 128 \
#             --opt adam --weight-decay 0 \
#             --warmup-lr $warmup_lr --warmup-epochs 20 \
#             --lr $lr --min-lr 1e-8 \
#             --drop-path 0 --img-size 224 \
#             --mixup 0 --cutmix 0 --smoothing 0 \
#             --output best_output/open_pos_embed_0.95_azi_${lr}_${warmup_lr}_output/ \
#             --amp --tuning-mode part --pretrained \
#             --pruning --pruning_method gradient_perCell \
#             --times_para 15 \
#             --gpu_id $gpu_id \
#             --log-wandb \
#             --experiment vtab \
#             --no-prefetcher --keep_rate 0.95 --top_param 0.03  --head_reparam
#     done
# done

# echo "Training completed for all learning rates and warmup-lr combinations."

# python train_gps.py path/to/vtab-1k/smallnorb_azi \
#     --dataset smallnorb_azi \
#     --num-classes 18 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz --validation-batch-size 128 \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-7 --warmup-epochs 10 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0.1 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 1 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --contrast-aug --no-prefetcher --contrastive

