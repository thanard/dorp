#!/usr/bin/env bash


python eval.py \
--model_dir results/cpc/pushenv/n_agents-2/num_onehots-0/grid_n-64/step_size-1/encoder-cswm/batch_size-128/num_filters-32/num_layers-3/z_dim-32/W-3/separate_W-0/temp-0.1/scale_weights-1.0/seed-80/lr-0.0001/loss-ce/normalization-batchnorm/ \
--n_agents 2 \
--time_limit 59 \
--vp_modeldir results/sv2p/pushenv/model.sv2p.context_1.kl_0._stochastic_false.n_agents_2/

python eval.py --n_agents 1 --vp_modeldir --grid_n results/gridworld/model.sv2p.context_1.kl_0._stochastic_false.n_agents_1