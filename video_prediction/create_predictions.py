import numpy as np
from video_prediction.control.policy.cem_controllers.cem_base_controller import CEMBaseController
import imp
from video_prediction.pred_util import get_context, rollout_predictions
from collections import OrderedDict
import video_prediction
import moviepy.editor as mpy

parent_dir = '/'.join(video_prediction.__file__.split('/')[:-2])


# LOADS CONTEXT FRAMES / ACTIONS FROM TRAJECTORY
data = np.load('data/randact_traj_length_20_n_trials_50_n_contexts_150.npy', allow_pickle=True)[0]    # take 0th traj as example
actions, images, states = [], [], []
for i in range(15):  # vid pred model has sequence length of 15
    img = data[i, 0][:,:,:3].astype(np.float32) / 255
    images.append(img[None][None])
    states.append(data[i, 1]['state'].reshape(-1)[None])
    actions.append(data[i + 1, 1]['action'].reshape(-1)[None])
actions, images, states = [np.concatenate(x, axis=0) for x in [actions, images, states]]
real = (images[2:] * 255.).astype(np.uint8)
images, states = images[:2][None], states[:2][None]        # vid pred model has context of 2, want to 0 out future to avoid accidental future input
actions=actions[None]                                      # pad actions with batch dimesnion
import ipdb
ipdb.set_trace()
# LOADS VIDEO PREDICTION MODEL
params = imp.load_source('params', parent_dir + '/experiments/block/control/conf.py')
net_conf = params.configuration
net_conf['batch_size'] = 1
net_context = net_conf['context_frames']
predictor = net_conf['setup_predictor'](dict(), net_conf)


# Generate predictions
gen_images = rollout_predictions(predictor, net_conf['batch_size'], actions, images, states)[0]
gen_images = (np.concatenate(gen_images, 0) * 255.).astype(np.uint8)                    # by default output images are 0-1, float32

mpy.ImageSequenceClip([gen_images[0, i, 0] for i in range(13)], fps=4).write_gif('./pred.gif')
mpy.ImageSequenceClip([real[i, 0] for i in range(13)], fps=4).write_gif('./real.gif')
