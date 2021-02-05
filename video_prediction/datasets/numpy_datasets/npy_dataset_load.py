import tensorflow as tf
import json
import numpy as np
import cv2
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

def transform_act(actions):
    if len(actions.shape) >= 3:
        actions = actions[:, :, 0]
    zone = (actions // 81)
    is_x = zone < 2
    which_direction = (zone % 2 - 0.5)*2
    output = [
        actions / 9,
        actions % 9,
        is_x * which_direction,
        (1 - is_x) * which_direction,
    ]
    return np.stack(output, axis=-1)

def transform_obs(obs):
    if obs.shape[2] !=64:
        output = np.zeros((obs.shape[0],
                           obs.shape[1],
                           64,
                           64,
                           3))
        for i in range(obs.shape[0]):
            for t in range(obs.shape[1]):
                output[i, t] = cv2.resize(obs[i, t], dsize=(64, 64), interpolation=cv2.INTER_AREA)
        return output/255.
    return obs/255.

class NPYDatasetLoad:
    MODES = ['train', 'val', 'test']

    # maybe fix the inheritance here more later?
    def __init__(self, file, config_path, batch_size):
        self._batch_size = batch_size
        self._config = {}
        with open(config_path) as f:
            self._config.update(json.loads(f.read()))
        self._hp = self._get_default_hparams().override_from_dict(self._config)

        train_obs = np.load(os.path.join(file, 'train_observations.npy'))
        train_acts = np.load(os.path.join(file, 'train_actions.npy'))
        if self._hp.env == 'pushenv':
            train_acts = transform_act(train_acts)
            train_obs = transform_obs(train_obs)
            assert train_acts.shape[-1] == self._hp.adim
        self._T = self._hp.T
        self._img_dims = train_obs.shape[2:]
        self._sdim = self._hp.sdim
        self._adim = self._hp.adim

        last = 0
        self._data = {}
        for i, m in enumerate(self.MODES):
            n_samps = int(self._hp.train_split[i] * train_obs.shape[0])
            self._data[m] = {'obs': train_obs[last:last + n_samps],
                             'acts': train_acts[last:last + n_samps]}
            last += n_samps

        self._datasets = {}

        # Manual unrolling
        m = 'train'
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

    def _get_default_hparams(self):
        default_params = {
            "sdim": 10,
            "adim": 10,
            "env": "gridworld",
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


    def _gen_data(self, mode):
        # i, n_epochs = 0, 0

        while True:
            # if i + self._batch_size > self._data[mode]['obs'].shape[0]:
            #     i, n_epochs = 0, n_epochs + 1
            #     if mode == 'train' and self._hp.num_epochs is not None and n_epochs >= self._hp.num_epochs:
            #         break
            #     [np.random.shuffle(self._data[m]) for m in self.MODES]
            #
            # i += 1
            i = np.random.choice(self._data[mode]['obs'].shape[0])
            t = np.random.choice(self._data[mode]['obs'].shape[1] - self._hp.T)
            actions = self._data[mode]['acts'][i, t:t + self._hp.T]
            images = self._data[mode]['obs'][i, t:t + self._hp.T]
            # actions = np.zeros((self._T, self._adim))
            # images = np.zeros((self._T, self._img_dims[0], self._img_dims[1], 3))
            states = np.zeros((self._T, self._sdim))
            yield actions, images, states


    def _extract_act_img_state(self, actions, images, states):
        actions = tf.reshape(actions, [self._batch_size, self._T, self._adim])
        states = tf.reshape(states, [self._batch_size, self._T, self._sdim])
        images = tf.reshape(images, [self._batch_size, self._T, self._img_dims[0], self._img_dims[1], 3])

        if self._hp.actions_shifted:
            actions = tf.concat([actions[:, 1:], tf.zeros_like(actions[:, 0, None])], axis=1)
            # states, images = states[:, :-1], images[:, :-1]

        return {'actions': actions, 'images': tf.cast(images, tf.float32), 'states': states}


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
        img_slice = _slice_helper(self.get('images', mode), self._rand_start, n_frames, 1)

        inputs['images'] = tf.cast(img_slice, img_dtype)
        if self._hp.use_states:
            inputs['states'] = _slice_helper(self.get('states', mode), self._rand_start, n_frames, 1)
        inputs['actions'] = _slice_helper(self.get('actions', mode), self._rand_start, n_frames - 1, 1)

        targets = _slice_helper(self.get('images', mode), self._rand_start + n_context, n_frames - n_context, 1)
        targets = tf.cast(targets, img_dtype)
        return inputs, targets


    @property
    def hparams(self):
        return self._hp


    def num_examples_per_epoch(self):
        return self._batch_size
