import torch
import numpy as np

def sample_anchors_positives(data, batch_size, step_size=1, random_step_size=False):
    idx, t = get_idx_t(batch_size, step_size, data)
    anchors = data[idx, t]
    if random_step_size:
        sample_step_size = np.random.choice(step_size, size=batch_size) + 1
        positives = data[idx, t + sample_step_size]
    else:
        positives = data[idx, t + step_size]
    return anchors, positives

def get_idx_t(batch_size, step_size, data):
    idx = np.random.choice(len(data), size=batch_size)
    t = np.random.choice(len(data[0]) - step_size, size=batch_size)
    # t = np.array([np.random.randint(len(data[i]) - k_steps) for i in idx])
    return idx, t

def get_sample_transitions(env, n_traj, len_traj, switching_factor_freq=1):
    """
    :param env:
    :param n_traj:
    :param len_traj:
    :param switching_factor_freq:
    :return: buffer of size n_traj x len_traj x C x W x H
    """
    env.reset()
    observations = []
    actions = []
    for n in range(n_traj):
        env.reset()
        traj_ims = []
        traj_acts = []
        count = entity_idx = 0
        for t in range(len_traj):
            if switching_factor_freq > 1:
                if count % switching_factor_freq == 0:
                    count = 0
                    entity_idx = np.random.choice(env.n_agents)
                action = env.sample_action_by_idx(entity_idx)
                count += 1
            else:
                action = env.sample_action()
            env.step(action)
            traj_acts.append(action)
            traj_ims.append(env.get_obs())
        actions.append(np.array(traj_acts))
        observations.append(np.array(traj_ims))
    observations = np.array(observations).transpose((0, 1, 4, 2, 3))
    return observations, actions


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data, step_size, use_random_step_size=False):
        'Initialization'
        self.observations = data['obs'].astype('float32') / 255
        self.actions = data['act']
        self.step_size = step_size
        self.use_random_step_size = use_random_step_size
        self.n_samples = len(self.observations)
        self.len_traj = len(self.observations[0])
        assert len(self.actions) == self.n_samples

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples * (self.len_traj - self.step_size)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = index % self.n_samples
        t = index // self.n_samples
        anchor = self.observations[idx, t]
        if self.use_random_step_size:
            sample_step_size = np.random.choice(self.step_size) + 1
            positive = self.observations[idx, t + sample_step_size]
            action = self.actions[idx, t + 1: t + sample_step_size + 1].sum(0)
        else:
            positive = self.observations[idx, t + self.step_size]
            action = self.actions[idx, t + 1: t + self.step_size + 1].sum(0)
        return anchor, positive, action
