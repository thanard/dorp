import numpy as np

def sample_anchors_positives(data, batch_size, step_size=1):
    data = np.stack(data)
    idx, t = get_idx_t(batch_size, step_size, data)
    anchors = data[idx, t]
    positives = data[idx, t + step_size]
    return anchors, positives

def get_idx_t(batch_size, k_steps, data):
    idx = np.random.choice(len(data), size=batch_size)
    t = np.array([np.random.randint(len(data[i]) - k_steps) for i in idx])
    return idx, t

def get_sample_transitions(env, n_traj, len_traj):
    env.reset()
    buffer = []
    for n in range(n_traj):
        env.reset()
        traj_ims = []
        for t in range(len_traj):
            action = env.sample_action()
            env.step(action)
            traj_ims.append(env.get_obs())
        buffer.append(np.array(traj_ims))
    return np.array(buffer)