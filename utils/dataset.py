import numpy as np

def sample_anchors_positives(data, batch_size, step_size=1, random_step_size=False):
    data = np.stack(data)
    idx, t = get_idx_t(batch_size, step_size, data)
    anchors = data[idx, t]
    if random_step_size:
        sample_step_size = np.random.choice(step_size) + 1
        positives = data[idx, t + sample_step_size]
    else:
        positives = data[idx, t + step_size]
    return anchors, positives

def get_idx_t(batch_size, k_steps, data):
    idx = np.random.choice(len(data), size=batch_size)
    t = np.array([np.random.randint(len(data[i]) - k_steps) for i in idx])
    return idx, t

def get_sample_transitions(env, n_traj, len_traj, switching_factor_freq=1):
    env.reset()
    buffer = []
    for n in range(n_traj):
        env.reset()
        traj_ims = []
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
            traj_ims.append(env.get_obs())
        buffer.append(np.array(traj_ims))
    return np.array(buffer)