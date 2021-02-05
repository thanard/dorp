#!/usr/bin/env bash


for n_agents in 5
do
    export model=sv2p
    export data_dir=data/pushenv/n_agents_${n_agents}/n_trajs_10000/len_traj_20/switching_factor_freq_1
    export output_dir=results/sv2p/pushenv/model.sv2p.context_1.kl_0._stochastic_false.n_agents_${n_agents}
    export config_dir=video_prediction/configs/pushenv_sv2p
    while [ ! -f ${data_dir}/train_observations.npy -o ! -f ${data_dir}/train_actions.npy ]
    do
        sleep 10 && echo "...checking if data for ${n_agents} objects exists"
    done
    CUDA_VISIBLE_DEVICES=${n_agents} python video_prediction/train_videopred.py --dataset NPYDatasetLoad --input_dirs ${data_dir} --experiment_dir ${config_dir} --output_dir ${output_dir} --model $model &
done
