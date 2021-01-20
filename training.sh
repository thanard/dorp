#!/usr/bin/env bash

### Optimizing for speed ###
# 1/19/21
CUDA_VISIBLE_DEVICES=0 python main.py --n_agents 3 --seed 80 --plan_freq 400 --switching_factor_freq 20 --n_traj 300 --len_traj 100 --random_step_size --batch_size 1024 --lr 1e-2

### key-wall ###
CUDA_VISIBLE_DEVICES=0 python main.py --env key-wall --n_traj 50 --len_traj 1200 --ce_temp .5 --temp .05  --z_dim 16 --encoder cswm-key --vis_freq 1 --seed 101  --lr 1e-4
# results/grid/n_agents-1/num_onehots-0/grid_n-16/step_size-1/encoder-cswm-key/batch_size-128/num_filters-32/num_layers-2/z_dim-16/W-3/separate_W-0/temp-0.05/scale_weights-1.0/seed-101/lr-0.0001/normalization-batchnorm/env-key-wall/n_agents-1/n_traj-50/len_traj-1200/ce_temp-0.5/

### k objects ###
for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=$i python main.py --n_agents $((i+3)) --seed 80 \
    --plan_freq 400 &
done
# output_dir: results/grid/n_agents-2/num_onehots-0/grid_n-16/step_size-1/encoder-cswm/batch_size-128/num_filters-32/num_layers-2/z_dim-32/W-3/separate_W-0/temp-0.1/scale_weights-1.0/seed-80/lr-0.001/normalization-batchnorm

# k objects
# Switching every 20 or 50
# Step size 1 5 or 20
for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=$i python main.py --n_agents $((i+3)) --seed 80 \
    --savepath results/grid --step_size 1 --n_traj 60 --len_traj 1000 --switching_factor_freq 50 --plan_freq 400 &
done

for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=$i python main.py --n_agents $((i+3)) --seed 80 \
    --savepath results/grid --random_step_size --step_size 5 --n_traj 60 --len_traj 1000 --switching_factor_freq 50 --plan_freq 400 &
done

for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=$i python main.py --n_agents $((i+3)) --seed 80 \
    --savepath results/grid --random_step_size --step_size 20 --n_traj 60 --len_traj 1000 --switching_factor_freq 50 --plan_freq 400 &
done

for i in 0 1 2 3
do
    CUDA_VISIBLE_DEVICES=$i python main.py --n_agents $((i+3)) --seed 80 \
    --savepath results/grid --random_step_size --step_size 5 --n_traj 60 --len_traj 1000 --switching_factor_freq 20 --plan_freq 400 &
done