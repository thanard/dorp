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

        # data_dir = '/home/thanard/Downloads/bco'
        # data1 = np.load(data_dir + "/bcov5_0.npy")#[:,:20,:]
        # # import ipdb
        # # ipdb.set_trace()
        # data2 = np.load(data_dir + "/bcov5_1.npy")
        # data3 = np.load(data_dir + "/bcov5_2.npy")
        # data4 = np.load(data_dir + "/bcov5_3.npy")
        # all_data = np.concatenate((data1, data2, data3, data4), axis=0)
        all_data = np.load(file, allow_pickle=True)
        self._T = all_data.shape[1]
        if self._hp.actions_shifted:
            print("Shifting actions back! T is now", self._T)
        self._img_dims = all_data[0, 0, 0].shape[:2]
        self._sdim = all_data[0, 0, 1]['obj_pos'].reshape(-1).shape[0] + all_data[0, 0, 1]['robot_pos'].reshape(-1).shape[0]
        self._adim = all_data[0, 0, 1]['action'].reshape(-1).shape[0]
        np.random.shuffle(all_data)

        last = 0
        self._data = {}
        for i, m in enumerate(self.MODES):
            n_samps = int(self._hp.train_split[i] * all_data.shape[0])
            self._data[m] = all_data[last:last + n_samps]
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


def _gen_data(self, mode):
    i, n_epochs = 0, 0

    while True:
        if i + self._batch_size > self._data[mode].shape[0]:
            i, n_epochs = 0, n_epochs + 1
            if mode == 'train' and self._hp.num_epochs is not None and n_epochs >= self._hparams.num_epochs:
                break
            [np.random.shuffle(self._data[m]) for m in self.MODES]

        ex_data = self._data[mode][i]
        i += 1

        actions, images, states = [], [], []
        for t in range(ex_data.shape[0]):
            actions.extend(ex_data[t, 1]['action'].reshape(-1)[None])
            states.extend(np.concatenate([ex_data[t, 1]['obj_pos'], ex_data[t, 1]['robot_pos']]).reshape(-1)[None])
            images.extend(ex_data[t, 0][:, :, :3][None])
        actions, images, states = [np.concatenate(x, axis=0) for x in (actions, images, states)]

        yield actions, images, states


def _extract_act_img_state(self, actions, images, states):
    actions = tf.reshape(actions, [self._batch_size, self._T, self._adim])
    states = tf.reshape(states, [self._batch_size, self._T, self._sdim])
    images = tf.reshape(images, [self._batch_size, self._T, self._img_dims[0], self._img_dims[1], 3])

    if self._hp.actions_shifted:
        actions = actions[:, 1:]
        states, images = states[:, :-1], images[:, :-1]

    return {'actions': actions, 'images': tf.cast(images, tf.float32), 'states': states}


def _get_default_hparams(self):
    default_params = {
        'train_split': [0.9, 0.05, 0.05],
                       'num_epochs': None,
        'actions_shifted': False,
        'use_states': False
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
