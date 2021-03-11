#!/usr/bin/env bash


python eval.py --logdir logdir/gridworld/1_agent/gridworld_1_obj_16_size_l1_reward_no_explore_sparse_reward/ --n_agents 1 --reward_type l1_dist_norm --time_limit 59 --vp_modeldir results/sv2p/pushenv/model.sv2p.context_1.kl_0._stochastic_false.n_agents_2/

python eval.py --n_agents 1 --vp_modeldir --grid_n results/gridworld/model.sv2p.context_1.kl_0._stochastic_false.n_agents_1