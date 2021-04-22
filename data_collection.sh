#!/usr/bin/env bash

### Key command
python push_env.py --savepath test.hdf5 --n_samples 100 --episode_length 100 --headless true


### Train Data
export n_object=2
export epl=100
export n_trial=150
export fname=data/${n_object}_objects_n_trajs_${n_trial}_ep_length_${epl}
export n_seeds=8
for i in {1..${n_seeds}}
do
    python push_env.py --savepath ${fname}_$i.hdf5 --n_samples $((epl*n_trial)) --episode_length $epl --n_objects $n_object &
done

wait && \
python hdf5_to_npy.py --read $fname --num 8 && \
mv actions.npy ${fname}_actions.npy && \
mv observations.npy ${fname}_observations.npy && \

ssh thanard@pabti2.ist.berkeley.edu "mkdir -p ~/dorp/data/pushenv/n_agents_${n_object}/n_trajs_$((epl*n_trial))/len_traj_${epl}/switching_factor_freq_1/" && \
scp ${fname}_actions.npy thanard@pabti2.ist.berkeley.edu:~/dorp/data/pushenv/n_agents_${n_object}/n_trajs_$((n_seeds*n_trial))/len_traj_${epl}/switching_factor_freq_1/train_actions.npy && \
scp ${fname}_observations.npy thanard@pabti2.ist.berkeley.edu:~/dorp/data/pushenv/n_agents_${n_object}/n_trajs_$((n_seeds*n_trial))/len_traj_${epl}/switching_factor_freq_1/train_observations.npy

### Validation Data
export valid_trial=10
export vname=${fname}_valid
for i in {1..10}
do
    python push_env.py --savepath ${vname}_$i.hdf5 --n_samples $((epl*valid_trial)) --episode_length $epl --n_objects $n_object &
done

wait && \
python hdf5_to_npy.py --read $vname --num 10  && \
mv actions.npy ${vname}_actions.npy  && \
mv observations.npy ${vname}_observations.npy && \

scp ${vname}_actions.npy thanard@pabti2.ist.berkeley.edu:~/dorp/data/pushenv/n_agents_${n_object}/n_trajs_$((n_seeds*n_trial))/len_traj_${epl}/switching_factor_freq_1/valid_actions.npy && \
scp ${vname}_observations.npy thanard@pabti2.ist.berkeley.edu:~/dorp/data/pushenv/n_agents_${n_object}/n_trajs_$((n_seeds*n_trial))/len_traj_${epl}/switching_factor_freq_1/valid_observations.npy
