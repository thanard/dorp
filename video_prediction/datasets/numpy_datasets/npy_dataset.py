import tensorflow as tf
import json
import numpy as np
from tensorflow.contrib.training import HParams
from collections import OrderedDict
from video_prediction.datasets.numpy_datasets.grid_helpers import *

def _slice_helper(tensor, start_i, N, axis):
    shape = tensor.get_shape().as_list()
    assert 0 <= axis < len(shape), "bad axis!"

    starts = [0 for _ in shape]
    ends = shape
    starts[axis], ends[axis] = start_i, N

    return tf.slice(tensor, starts, ends)


class NPYDataset:
    MODES = ['train', 'val', 'test']
    # maybe fix the inheritance here more later?
    def __init__(self, file, config_path, batch_size):
        self._batch_size = batch_size
        self._config = {}
        with open(config_path) as f:
            self._config.update(json.loads(f.read()))
        self._hp = self._get_default_hparams().override_from_dict(self._config)
        self._T = self._hp.T
        self._img_dims = self._hp.input_dims
        self._sdim = self._hp.sdim
        self._adim = self._hp.adim
        self._env = self._hp.env
        if self._env == 'gridworld':
            self._n_agents = self._hp.n_agents

        self._datasets = {}

        # Manual unrolling
        m = 'train'
        dataset = tf.data.Dataset.from_generator(lambda:self._gen_data(m), (tf.float32, tf.float32, tf.float32)).batch(self._batch_size)
        dataset = dataset.map(self._extract_act_img_state).prefetch(10)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        output_element = {}
        for k in list(next_element.keys()):
            output_element[k] = tf.reshape(next_element[k],
                                           [self._batch_size] + next_element[k].get_shape().as_list()[1:])

        self._datasets[m] = output_element

        m = 'val'
        dataset = tf.data.Dataset.from_generator(lambda: self._gen_data(m), (tf.float32, tf.float32, tf.float32)).batch(
            self._batch_size)
        dataset = dataset.map(self._extract_act_img_state).prefetch(10)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        output_element = {}
        for k in list(next_element.keys()):
            output_element[k] = tf.reshape(next_element[k],
                                           [self._batch_size] + next_element[k].get_shape().as_list()[1:])

        self._datasets[m] = output_element

        m = 'test'
        dataset = tf.data.Dataset.from_generator(lambda: self._gen_data(m), (tf.float32, tf.float32, tf.float32)).batch(
            self._batch_size)
        dataset = dataset.map(self._extract_act_img_state).prefetch(10)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        output_element = {}
        for k in list(next_element.keys()):
            output_element[k] = tf.reshape(next_element[k],
                                           [self._batch_size] + next_element[k].get_shape().as_list()[1:])

        self._datasets[m] = output_element

        self._rand_start = None

    def _gen_data(self, mode):
        i, n_epochs = 0, 0

        while True:
            images, actions = get_trajectories_with_actions_no_overlap(self._n_agents, 1, self._T,
                                                                       self._hp.grid_n, circular=False, step_size=1)
            actions = actions
            # To RGB
            images = np.clip(to_rgb_np(images.reshape(-1, self._hp.grid_n, self._hp.grid_n, self._n_agents))/10., 0, 1)
            # Scale up
            scale_fac = [self._img_dims[0] // self._hp.grid_n, self._img_dims[1] // self._hp.grid_n]
            images = np.repeat(np.repeat(images,
                                         scale_fac[0], axis=1),
                               scale_fac[1], axis=2)
            # Reshape back
            images = images.reshape(1, self._T, self._img_dims[0], self._img_dims[1], 3)
            states = np.zeros((1, self._T, self._sdim))
            yield actions[0], images[0], states[0]
    
    def _extract_act_img_state(self, actions, images, states):
        actions = tf.reshape(actions, [self._batch_size, self._T, self._adim])
        states = tf.reshape(states, [self._batch_size, self._T, self._sdim])
        images = tf.reshape(images, [self._batch_size, self._T,  self._img_dims[0], self._img_dims[1], 3])

        if self._hp.actions_shifted:
            actions=actions[:, 1:]
            states, images = states[:, :-1], images[:, :-1]

        return {'actions': actions, 'images': tf.cast(images, tf.float32), 'states': states}
        
    def _get_default_hparams(self):
        default_params = {
            "sdim": 10,
            "adim": 10,
            "input_dims": [64, 64],
            "grid_n": 16,
            "env": "gridworld",
            "n_agents": 5,
            "actions_shifted": False,
            "num_epochs": None,
            "train_split": [
                0.9,
                0.05,
                0.05
            ],
            "use_states": False,
            "T": 20
        }
        
        return HParams(**default_params)

    def get(self, key, mode='TRAIN'):
        assert key in self._datasets[mode], "Key {} is not recognized for mode {}".format(key, mode)

        return self._datasets[mode][key]
    
    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise KeyError('Index should be in format: [Key, Mode] or [Key] (assumes default train mode)')
            key, mode = item
            return self.get(key, mode)

        return self.get(item)

    def make_input_targets(self, n_frames, n_context, mode, img_dtype=tf.float32):
        assert n_frames > 0

        if self._rand_start is None:
            img_T = self.get('images', mode).get_shape().as_list()[1]
            self._rand_start = tf.random_uniform((), maxval=img_T - n_frames + 1, dtype=tf.int32)
        
        inputs = OrderedDict()
        img_slice =  _slice_helper(self.get('images', mode), self._rand_start, n_frames, 1)
        
        inputs['images'] = tf.cast(img_slice, img_dtype)
        if self._hp.use_states:
            inputs['states'] = _slice_helper(self.get('states', mode), self._rand_start, n_frames, 1)
        inputs['actions'] = _slice_helper(self.get('actions', mode), self._rand_start, n_frames-1, 1)
        
        targets = _slice_helper(self.get('images', mode), self._rand_start + n_context, n_frames - n_context, 1)
        targets = tf.cast(targets, img_dtype)
        return inputs, targets

    @property
    def hparams(self):
        return self._hp

    def num_examples_per_epoch(self):
        return self._batch_size
