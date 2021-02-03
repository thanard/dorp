import numpy as np
import torch
from torch.autograd import Variable
import os
from envs.gridworld import GridWorld
from envs.key_door import *
from envs.maze import Maze


def get_env(env_name, args):
    if env_name == "gridworld":
        env = GridWorld(args.n_agents, args.grid_n)
    elif env_name == 'key-wall':
        env = KeyWall(1)
    elif env_name == 'key-corridor':
        env = KeyCorridor(1, 1)
    elif env_name == "key-wall-random":
        args.n_keys = 1
        env = KeyWallRandom(1)
    elif env_name == 'key-corridor-random':
        args.n_keys = 1
        env = KeyCorridorRandom(1)
    elif env_name == 'key-wall-2':
        env = KeyWall2Keys(1)
        args.n_keys = 2
    elif env_name == 'key-wall-sequential':
        args.n_keys = 2
        env = KeyWallSequential(1)
    elif env_name== 'key-corridor-sequential':
        args.n_keys = 3
        env = KeyCorridorSequential(1)
    elif env_name == 'maze':
        env = Maze()
    else:
        raise NotImplementedError("Environment not recognized: %s" % env_name)
    return env

def im2cuda(npx, normalize=True):
    var = torch.from_numpy(npx).float()
    var = var.cuda(non_blocking=True)
    if normalize:
        var /= 255
    return var


def single_im_to_torch(npx, normalize=True):
    var = torch.from_numpy(npx).float().cuda().unsqueeze(0)
    if normalize:
        var /= 255
    return var

def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

def log_sum_exp(arr):
    max_arr = torch.max(arr, dim=1, keepdim=True)[0]
    return max_arr + torch.log(torch.sum(torch.exp(arr - max_arr), dim=1))

def setup_savepath(kwargs):
    savepath = kwargs["savepath"]
    included = ["n_agents",
                "num_onehots",
                "rgb",
                "grid_n",
                "step_size",
                "allow_agent_overlap",
                "encoder",
                "batch_size",
                "fast",
                "n_negs",
                "num_filters",
                "num_layers",
                "z_dim",
                "W",
                "separate_W",
                "temp",
                "scale_weights",
                "h_distance",
                "seed",
                "lr",
                "circular",
                "loss",
                ]
    if kwargs['encoder'].startswith("attention") or kwargs['encoder'].startswith("cswm"):
        for key in ["activation", "normalization", "scope", "reg"]:
            if key in kwargs.keys():
                included.append(key)
    if kwargs['encoder'].startswith("cswm-key") or kwargs['encoder'].startswith("cswm-multi"):
        included.extend(["env", "n_agents", "n_keys", "n_traj", "len_traj", "key_alpha", 'ce_temp'])
    if kwargs['random_step_size']:
        included.append('random_step_size')
    if kwargs['switching_factor_freq'] > 1:
        included.append('switching_factor_freq')
    final_savepath = savepath
    for key in included:
        if key in kwargs.keys():
            final_savepath = os.path.join(final_savepath, key + "-%s" % str(kwargs[key]))
    return final_savepath

def save_python_cmd(savepath, kwargs, script):
    excluded = ['hamming_reg', 'hamming_alpha', 'vis_freq', 'label_online']
    cmd = 'python ' + script + " "
    for key, val in kwargs.items():
        if key not in excluded:
            cmd += " --%s %s" % (key, str(val))
    with open(os.path.join(savepath, "cmd.txt"), "w") as text_file:
        text_file.write(cmd)

def tensor_to_label(tensor, num_factors, z_dim):
    '''
    Converts a single tensor onehot output to a label
    :param tensor: single tensor
    :param num_factors: num onehots outputs for model
    :param z_dim: latent z dim per onehot
    :return: int label
    '''
    return int(sum(tensor[i] * z_dim ** (num_factors - i - 1) for i in range(num_factors)))

def label_to_tensor(label, num_factors, z_dim):
    # inverse of tensor_to_label
    output = []
    while label:
        output.append(int(label % z_dim))
        label //= z_dim
    output = np.array(output[::-1])
    final_label = np.zeros(num_factors)
    final_label[num_factors - len(output):] = output
    return final_label.astype('uint8')

def tensor_to_label_array(idx_array, num_factors, z_dim):
    '''
    Convert a batch of idx_array to labels
    :param idx_array: batch_size x z_dim
    :param num_factors: the number of onehots
    :param z_dim:
    :return: labels: 1D array of size batch_size
    '''
    return (np.tile(np.power(z_dim, np.flip(np.arange(num_factors)))[None],
                    (idx_array.shape[0], 1)) * idx_array).sum(1)

def tensor_to_label_grouped(tensor, z_dim, groups):
    final_label = []
    group_idx = 0
    idx = 0
    while group_idx < len(groups):
        n_onehots_in_group = groups[group_idx]
        label = int(sum(tensor[i] * z_dim ** (n_onehots_in_group - (i - idx) - 1) for i in
                          range(idx, n_onehots_in_group + idx)))
        final_label.append(label)
        idx += n_onehots_in_group
        group_idx+=1
    return final_label

def tensor_to_label_grouped_batch(tensors, z_dim, groups):
    final_labels = []
    group_idx = 0
    idx = 0
    while group_idx < len(groups):
        n_onehots_in_group = groups[group_idx]
        x = np.array(tensors[:,i] * z_dim ** (n_onehots_in_group - (i - idx) - 1) for i in
                        range(idx, n_onehots_in_group + idx))
        labels = np.sum(x, axis=0).astype('int')
        final_labels.append(labels)
        idx += n_onehots_in_group
        group_idx += 1
    final_labels = np.stack(final_labels).T
    return final_labels





