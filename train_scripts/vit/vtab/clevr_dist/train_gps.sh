gpu_id=1
bz=32
lr=0.05
# 定义学习率列表
# learning_rates=("45e-5" "44e-5" "46e-5" "42e-5"  )
# learning_rates=("45e-5")
# 定义对应的 warmup 学习率列表
# warmup_lrs=("4e-3" "5e-2" "3e-3"   "2e-3"  "2e-4" "5e-5" "1e-3") 
# warmup_lrs=("45e-5") 



keep_rates=("0.95" "0.9" "0.8" "0.5")
for keep_rate in "${keep_rates[@]}"; do
    echo "Training with token: $keep_rate"
    
    # 运行训练脚本，传递当前的学习率和 warmup 学习率
    python train_gps.py /home/yhr/GPS1/data/vtab/vtab-1k/clevr_dist \
        --dataset clevr_dist \
        --num-classes 6 --direct-resize --no-aug \
        --model vit_base_patch16_224_in21k \
        --epochs 100 \
        --batch-size $bz --validation-batch-size 128 \
        --opt adam --weight-decay 0 \
        --warmup-lr 4e-3 --warmup-epochs 20 \
        --lr 45e-5 --min-lr 1e-8 \
        --drop-path 0.1 --img-size 224 \
        --mixup 0 --cutmix 0 --smoothing 0 \
        --output 5811_output/${keep_rate}_dist_output/ \
        --amp --tuning-mode part --pretrained \
        --pruning --pruning_method gradient_perCell \
        --times_para 15 \
        --gpu_id $gpu_id \
        --log-wandb \
        --experiment vtab \
        --no-prefetcher --keep_rate $keep_rate --top_param 0.03  --head_reparam --init_lr
done



# warmup_lrs=("4e-3") 
# learning_rates=("45e-5")

# # 循环遍历每个学习率
# for lr in "${learning_rates[@]}"; do
#     # 循环遍历每个 warmup 学习率
#     for warmup_lr in "${warmup_lrs[@]}"; do
#         # 输出当前学习率和 warmup 学习率，便于调试
#         echo "Training with learning rate: $lr and warmup-lr: $warmup_lr"
        
#         # 运行训练脚本，传递当前的学习率和 warmup 学习率
#         python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/clevr_dist \
#             --dataset clevr_dist \
#             --num-classes 6 --direct-resize --no-aug \
#             --model vit_base_patch16_224_in21k \
#             --epochs 100 \
#             --batch-size $bz --validation-batch-size 128 \
#             --opt adam --weight-decay 0 \
#             --warmup-lr $warmup_lr --warmup-epochs 20 \
#             --lr $lr --min-lr 1e-8 \
#             --drop-path 0.1 --img-size 224 \
#             --mixup 0 --cutmix 0 --smoothing 0 \
#             --output best_output/0.95_dist_${lr}_drop_${warmup_lr}_output/ \
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
# python train_gps.py /home1/yhr/GPS1/data/vtab/vtab-1k/clevr_dist \
#     --dataset clevr_dist \
#     --num-classes 6 --direct-resize --no-aug \
#     --model vit_base_patch16_224_in21k \
#     --epochs 100 \
#     --batch-size $bz \
#     --opt adam  --weight-decay 0 \
#     --warmup-lr 1e-4 --warmup-epochs 20 \
#     --lr $lr --min-lr 1e-8 \
#     --drop-path 0.1 --img-size 224 \
# 	--mixup 0 --cutmix 0 --smoothing 0 \
#     --output output/ \
#     --amp --tuning-mode part --pretrained \
#     --pruning --pruning_method gradient_perCell \
#     --times_para 15 \
#     --gpu_id $gpu_id \
#     --log-wandb \
#     --experiment vtab \
#     --no-prefetcher 

# python train_gps.py path/to/vtab-1k/clevr_dist \
#     --dataset clevr_dist \
#     --num-classes 6 --direct-resize --no-aug \
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

