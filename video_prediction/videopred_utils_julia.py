from video_prediction.sv2p_julia import *
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import os
from grid_env.grid_helpers import from_numpy_to_var

MAX_BATCH_SIZE=16


def load_vp_model(model_dir, n_agents, z_dim):
    pass

def vp_process_batch(vp_model, start_ims, action_seqs, max_batch_size=MAX_BATCH_SIZE):
    # start_ims must be normalized RGB images
    i = 0
    n_seqs = len(start_ims)
    ret_imgs = []
    while i < n_seqs:
        batch_starts, batch_actions = start_ims[i:i+max_batch_size], action_seqs[i:i+max_batch_size]
        pred_ims, mus, logvars = vp_model(batch_starts, batch_actions, eval=True)
        pred_ims = torch.stack(pred_ims).permute(1,0,2,3,4)
        ret_imgs.append(pred_ims)
        i+=max_batch_size
    ret_imgs = torch.cat(ret_imgs)
    return ret_imgs

def grid_to_rgb(grid_im, input_dim):
    grid_im = from_numpy_to_var(grid_im).unsqueeze(0).permute(0,3,1,2)
    rgb_im = convert_to_rgb(grid_im, input_dim).squeeze(0)
    return rgb_im