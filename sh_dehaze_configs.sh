#!/bin/sh
num_gpus=1

# batch size,  for 3090ti 4,  for HPC 6?


python -W ignore gen_configs.py --config_save_dir=./dehaze_configs --task=restore --nepochs=50 --log_dir=exp_dehaze --nepochs_decay=350 --gpus=$num_gpus --use_amp=1 --batch_size=6 --data_Path=../Data/garda/garda1_Und \
--is_underwater=1 --weight_decay=0 --lr=5e-4 --cuda_visiable_device=0 --seed=98 \
--height=320 --width=512 --eval_batch_size=4 \
--network_type=dehaze --predict_depth=0 --share_depth_encoder=0 \
--dual_transmiss_net=1 --last_flag=4 \
--lambda_ls=1 --lambda_spw=1 --lambda_spb=1 --lambda_gw=0.1 --lambda_tv=0 --lambda_dual_tr=1 \
--eval_depth=1 --eval_restore=1 --lambda_person_td=1 --FLAG_SEP_TRAIN=1


