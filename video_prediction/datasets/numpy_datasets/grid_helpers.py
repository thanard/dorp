'''
Date: 9/28/20
Commit: 74d3dfbca53390891ec2a620494e79e32c1fd2b5
'''
import numpy as np
# import torch
# from torch.autograd import Variable
import sys
import os
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

RGB_COLORS = {
    "yellow": [255, 255, 50],
    "cyan": [100, 255, 255],
    "purple": [128, 0, 255],
    "red": [255, 0, 0],
    "green": [128, 255, 0],
} # color of each agent in order

# AGENT_SIZES = [[7,7], [7,7], [7,7], [7,7], [7,7]]
AGENT_SIZES = [[2,4], [4,2], [2,4], [2,4], [4,2]]
# AGENT_SIZES = [[1,1], [1,1], [1,1], [1,1], [1,1]]
# AGENT_SIZES = [[3,3], [3,3], [3,3], [3,3], [3,3]]

def to_rgb_np(inputs):  # converts 16x16xn input to RGB
    """
    :param inputs: batch_size x height x width x n_agents
    :return: batch_size x height*scale_factor x width*scale_factor x 3
    """
    n_channels = 3
    n_agents = inputs.shape[3]
    batch_size = inputs.shape[0]
    if n_agents > 5:
        raise NotImplementedError("rgb input not implemented for more than 5 agents")
    rgb_im = np.zeros((batch_size, *inputs.shape[1:3], n_channels))
    colors = list(RGB_COLORS.values())[:n_agents]
    for i in range(n_agents):
        cur_agent = inputs[:, :, :, i, None]
        cur_im = np.tile(cur_agent, (1, 1, 1, n_channels))
        for c_idx, c in enumerate(colors[i]):
            cur_im[:, :, :, c_idx] += cur_im[:, :, :, c_idx] * c
        rgb_im += cur_im
    rgb_im /= 255  # normalize
    # save_image(rgb_im[:16]/10, "results/grid/08-19-20-exp/sample-im-3.png", padding=5, pad_value=10)
    return rgb_im


def get_max_agent_positions(n_agents, grid_n):
    max_positions = []
    for agent_idx in range(n_agents):
        max_x = grid_n-AGENT_SIZES[agent_idx][0]
        max_y = grid_n-AGENT_SIZES[agent_idx][1]
        max_positions.append(np.array([max_x, max_y]))
    return np.concatenate(max_positions)

def sample_single(n):
    '''
    :param n: grid size
    :return: np array of anchors, np array of positives

    step size is 1, agent is 1 pixel
    '''
    o_samples = np.mgrid[0:n:1, 0:n:1].reshape(2, -1).T
    actions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
    o_next = []
    os = []
    for o in o_samples:
      for action in actions:
        next_pos = (o + action) % n
        os.append(o)
        o_next.append(next_pos)
    os = np.array(os)
    o_next_samples = np.array(o_next)
    return os, o_next_samples

def log_sum_exp(arr):
    max_arr = torch.max(arr, dim=1, keepdim=True)[0]
    return max_arr + torch.log(torch.sum(torch.exp(arr - max_arr), dim=1))

def sample_double(n):
    '''
    :param n: grid size
    :return: np array of anchors, np array of positives

    step size is 1, agents are each 1 pixel
    '''
    xy = np.mgrid[0:n:1, 0:n:1, 0:n:1, 0:n:1].reshape(4, -1).T
    actions = [[0, 1, 0, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]]
    o_samples = xy
    o_next = []
    os = []
    for o in o_samples:
      for action in actions:
        next_pos = o.copy()
        next_pos = (o + action) % n
        os.append(o)
        o_next.append(next_pos)
    os = np.array(os)
    o_next_samples = np.array(o_next)
    return os, o_next_samples

def visualize_single(model, dataset, encoder, n, epoch=0):
    data, _ = dataset  # only look at anchors
    data = data[::4]
    batch_size = n  # process by row
    i = 0
    map = []
    while i < len(data):
        batch = from_numpy_to_var(np.transpose(data[i:i + batch_size], (0, 3, 1, 2)).astype('float32'))
        zs = model.encode(batch, vis=True).squeeze(1).detach().cpu().numpy()
        map.append(zs)
        i += batch_size
    map = np.array(map)
    return map

def from_numpy_to_var(npx, dtype='float32'):
    var = Variable(torch.from_numpy(npx.astype(dtype)))
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

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
                "lr"
                ]
    if kwargs['encoder'].startswith("attention") or kwargs['encoder'].startswith("cswm"):
        for key in ["activation", "normalization", "scope", "reg"]:
            if key in kwargs.keys():
                included.append(key)
    if kwargs['encoder'].startswith("cswm-key"):
        included.extend(["env", "n_agents", "n_keys", "n_traj", "len_traj", "key_alpha", 'ce_temp'])
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

def process_data_single(dataset, n):
  '''
  :param dataset: tuple of anchor, positive np arrays
  :param n: grid size
  :return: n x n x 1 onehot encodings of agent position
  '''
  o_pos, o_next_pos = dataset
  o = []
  o_next = []
  for pos, pos_next in zip(o_pos, o_next_pos):
    im_cur = np.zeros((n, n, 1))
    im_next = np.zeros((n, n, 1))
    x_cur, y_cur = pos
    x_next, y_next = pos_next
    im_cur[x_cur, y_cur ,0] = 10
    im_next[x_next, y_next, 0] = 10
    o.append(im_cur)
    o_next.append(im_next)
  return np.array(o), np.array(o_next)

def process_data_double(dataset, n):
  '''
  :param dataset: tuple of anchor, positive np arrays
  :param n: grid size
  :return: n x n x 2 onehot encodings of agent position
  '''
  o_pos, o_next_pos = dataset
  o = []
  o_next = []
  i = 0
  for pos, pos_next in zip(o_pos, o_next_pos):
    im_cur = np.zeros((n, n, 2))
    im_next = np.zeros((n, n, 2))
    x_cur0, y_cur0, x_cur1, y_cur1,= pos
    x_next0, y_next0, x_next1, y_next1 = pos_next
    im_cur[x_cur0, y_cur0, 0], im_cur[x_cur1, y_cur1, 1] = 10, 10
    im_next[x_next0, y_next0, 0], im_next[x_next1, y_next1, 1] = 10, 10
    o.append(im_cur)
    o_next.append(im_next)
    i+=1

  return np.array(o), np.array(o_next)

def process_single_pos(n_agents, pos, grid_n):
    '''
    :param n_agents: # agents
    :param pos: position
    :param n: grid size
    :return: single [n x n x n_agents] onehot image corresponding to input position
    '''
    im = np.zeros((grid_n, grid_n, n_agents))
    for i in range(n_agents):
        agent_dim = AGENT_SIZES[i]
        x_cur, y_cur = pos[2 * i], pos[2 * i + 1]
        im[x_cur, y_cur, i] = 10

        for x in range(agent_dim[0]):
            for y in range(agent_dim[1]):
                im[(x_cur + x) % grid_n, (y_cur + y) % grid_n, i] = 10
    return im

def sample_single_agent_pos(agent_idx, n_agents, grid_n, circular=False):
    max_positions = get_max_agent_positions(n_agents, grid_n)
    if circular:
        sample_pos = np.random.randint(grid_n, size=2)
    else:
        max_position_agent = max_positions[2*agent_idx:2*agent_idx+2]
        sample_pos_x = np.random.randint(max_position_agent[0])
        sample_pos_y = np.random.randint(max_position_agent[1])
        sample_pos = np.array((sample_pos_x, sample_pos_y))
    return sample_pos

def is_overlapping(agent_idx, seen, sample_pos):
    agent_dim = AGENT_SIZES[agent_idx]
    for x in range(agent_dim[0]):
        for y in range(agent_dim[1]):
            disp = np.array([x, y])
            if (tuple(disp + sample_pos)) in seen:
                return True
    return False

def get_unique_coordinates(n_agents, n_samples, grid_n, step_size, circular=True):
    # import pdb; pdb.set_trace()
    '''
    samples n_samples agent transitions such that no two agents overlap

    :param n_agents: # agents
    :param n_samples: number of sample coordinates
    :param grid_n: grid size
    :return: tuple of 2 np arrays: (anchors, positives)
    '''
    max_positions = get_max_agent_positions(n_agents, grid_n)
    pos = []
    pos_next = []
    single_agent_actions = np.concatenate((np.eye(2), -step_size*np.eye(2)))
    for _ in range(n_samples):
        seen = set()
        cur_sample = []
        for i in range(n_agents):
            sample_pos = sample_single_agent_pos(i, n_agents,grid_n, circular=circular)
            while is_overlapping(i, seen, sample_pos):
                sample_pos = sample_single_agent_pos(i, n_agents,grid_n, circular=circular)
            agent_dim = AGENT_SIZES[i]
            for x in range(agent_dim[0]):
                for y in range(agent_dim[1]):
                    disp = np.array([x,y])
                    seen.add(tuple(disp + sample_pos))
            cur_sample.append(sample_pos)
        pos.append(np.concatenate(cur_sample))

        rand_agent_to_move = np.random.choice(n_agents)
        rand_action = single_agent_actions[np.random.choice(4)]
        if circular:
            new_pos = (cur_sample[rand_agent_to_move] + rand_action) % grid_n
        else:
            # clamp next position
            new_pos = (cur_sample[rand_agent_to_move] + rand_action)
            new_pos = np.amax((np.zeros(2),
                    np.amin((new_pos,
                              max_positions[2*rand_agent_to_move:2*rand_agent_to_move+2])
                             , axis=0)), axis=0)
        agent_dim = AGENT_SIZES[rand_agent_to_move]
        for x in range(agent_dim[0]):
            for y in range(agent_dim[1]):
                disp = np.array([x, y])
                if (tuple(disp + new_pos)) in seen:
                    seen.remove(tuple(disp + new_pos))
        if not is_overlapping(rand_agent_to_move, seen, new_pos):
            cur_sample[rand_agent_to_move] = new_pos
        pos_next.append(np.concatenate(cur_sample))
    return np.array(pos).astype('uint8'), np.array(pos_next).astype('uint8')

def position_to_image(positions, n_agents, grid_n):
    '''
    Converts batch of agent xy positions to batch of [grid_n, grid_n, n_agents] images

    :param positions: batch of agent positions
    :param n_agents: # agents
    :param grid_n: grid size
    :return: [sample_size, grid_n, grid_n, n_agents] images
    '''
    if grid_n not in [16,64]:
        raise NotImplementedError("Grid size not supported: %d" % grid_n)
    n_samples = positions.shape[0]
    ims = np.zeros((n_samples, grid_n, grid_n, n_agents))
    for i in range(n_agents):
        agent_dim = AGENT_SIZES[i]
        x_cur, y_cur = positions[:, 2*i], positions[:, 2*i+1]
        ims[np.arange(n_samples), x_cur, y_cur, i] = np.ones(n_samples) * 10
        if grid_n in [16, 64]:
            for x in range(agent_dim[0]):
                for y in range(agent_dim[1]):
                    ims[np.arange(n_samples), (x_cur+x) % grid_n, (y_cur+y) % grid_n, i] = np.ones(n_samples)*10
    return ims

def image_to_position_thanard(im, agent_idx, grid_n):
    if grid_n in [16, 64]:
        up_left_pos = find_top_left(im[:,:,agent_idx])
        return up_left_pos
    else:
        raise NotImplementedError("Grid size not supported: %d" % grid_n)

def image_to_position(im, agent_idx, grid_n):
    '''
    :param im: A (grid_n, grid_n, n ) image where n is # agents
    :param agent_idx: idx of the agent to locate
    :param grid_n: size of grid
    :return: positon of the agent corresponding to the channel
    '''
    # if grid_n==16:
    #     return np.concatenate(np.where(im[:,:,agent_idx]!=0))
    # elif grid_n == 64:
    agent_grid = im[:,:,agent_idx].astype('uint8')
    if agent_grid[0,0] != 0 and agent_grid[(grid_n-1), (grid_n-1)] != 0:
        # if agent occupies upper left and bottom right corner, look at bottom right square for position
        last_col = agent_grid[:,grid_n-1]
        bottom_right_x = -1
        for i in np.arange(grid_n-1, -1, -1):
            if last_col[i] == 0:
                break
            bottom_right_x = i
        assert bottom_right_x != -1
        bottom_right_y = -1
        for i in np.arange(grid_n-1, -1, -1): # count backwards from the right side of grid
            if agent_grid[bottom_right_x, i] == 0:
                break
            bottom_right_y = i

        return np.array([bottom_right_x, bottom_right_y])

    first_row, last_row = agent_grid[0], agent_grid[grid_n-1]
    first_col, last_col = agent_grid[:,0], agent_grid[:, grid_n-1]

    if np.where(first_row!=0)[0].any() and np.where(last_row!=0)[0].any():
        # if agent occupies first and last row, looks at bottom area for position
        bottom_y = np.where(last_row!=0)[0][0]
        bottom_x = -1
        for i in np.arange(grid_n-1, -1, -1):
            if agent_grid[i,bottom_y] == 0:
                break
            bottom_x = i
        assert bottom_x != -1

        return np.array([bottom_x, bottom_y])

    elif np.where(first_col!=0)[0].any() and np.where(last_col!=0)[0].any():
        right_x = np.where(last_col!=0)[0][0]
        right_y = -1
        for i in np.arange(grid_n-1, -1, -1):
            if agent_grid[right_x, i] == 0:
                break
            right_y = i
        assert right_y != -1

        return np.array([right_x, right_y])
    else:
        positions = np.where(im[:,:,agent_idx]!=0)
        up_left_pos = np.array((positions[0][0], positions[1][0]))
        return up_left_pos
    # else:
    #     raise NotImplementedError("Grid size not supported: %d" % grid_n)

def sample_and_process_transitions(n_agents, n_samples, grid_n, allow_overlap=True, step_size=1, circular=True):
    '''
    Samples n_samples (anchor positive pairs), returns processed images
    :param n_agents: # agents
    :param n_samples: # sample transitions
    :param grid_n: grid width
    :param allow_overlap: True if agents move on top of each other
    :param step_size: # of steps per transition
    :return: tuple of 2 np arrays of processed images (anchors, positives)
                                                    each size [n_samples, grid_n, grid_n, n_agents]
    '''
    actions = np.eye(2 * n_agents)*step_size
    actions = np.concatenate((actions, -1 * actions))
    if allow_overlap:
        pos = np.random.randint(grid_n, size=(n_samples, 2 * n_agents))
        sample_actions = actions[np.random.randint(len(actions), size=n_samples)]
        ## Use below for multi-object movement ##
        # sample_actions = np.random.choice((0, -1, 1), pos.shape)
        pos_next = ((pos + sample_actions) % grid_n).astype('uint8')
    else:
        pos, pos_next = get_unique_coordinates(n_agents, n_samples, grid_n, step_size, circular=circular)
    im_cur = position_to_image(pos, n_agents, grid_n)
    im_next = position_to_image(pos_next, n_agents, grid_n)
    return im_cur, im_next

def reset(n_agents, grid_n, allow_agent_overlap=True, circular=True):
    if allow_agent_overlap:
        return np.random.randint(grid_n, size=(2 * n_agents))
    seen = set()
    cur_sample = []
    for i in range(n_agents):
        sample_pos = sample_single_agent_pos(i, n_agents, grid_n, circular=circular)
        while is_overlapping(i, seen, sample_pos):
            sample_pos = sample_single_agent_pos(i, n_agents, grid_n, circular=circular)
        agent_dim = AGENT_SIZES[i]
        for x in range(agent_dim[0]):
            for y in range(agent_dim[1]):
                disp = np.array([x, y])
                seen.add(tuple(disp + sample_pos))
        cur_sample.append(sample_pos)
    pos = np.concatenate(cur_sample).astype('uint8')
    return pos, seen

def get_trajectories_with_actions(n_agents, n_traj, len_traj, grid_n, allow_agent_overlap=True, circular=True, step_size=1):
    if not allow_agent_overlap:
        return get_trajectories_with_actions_no_overlap(n_agents, n_traj, len_traj, grid_n, step_size=step_size, circular=circular)
    max_positions = get_max_agent_positions(n_agents,grid_n)
    actions = np.eye(2 * n_agents) * step_size
    actions = np.concatenate((actions, -1 * actions)).astype('int')
    o_starts = np.random.randint(grid_n, size=(n_traj, 2 * n_agents))
    all_actions = []
    im_data = []
    for traj_idx in range(n_traj):
        os = []
        traj_actions = []
        o = o_starts[traj_idx]
        for t in range(len_traj):
            action = actions[np.random.randint(len(actions))]
            if not circular:
                # clamp next position
                next_pos = (o + action)
                next_pos = np.amax(np.zeros_like(max_positions),
                        (np.amin((next_pos, max_positions), axis=0)), axis=0)
            else:
                next_pos = (o+action) % grid_n
            os.append(o)
            traj_actions.append(action)
            o = next_pos.astype('uint8')
        traj_imgs = position_to_image(np.array(os), n_agents, grid_n)
        im_data.append(traj_imgs)
        all_actions.append(np.array(traj_actions))
    return np.array(im_data), np.array(all_actions)

def get_trajectories_with_actions_no_overlap(n_agents, n_traj, len_traj, grid_n, circular=True, step_size=1):
    max_positions = get_max_agent_positions(n_agents, grid_n)
    actions = np.eye(2 * n_agents) * step_size
    actions = np.concatenate((actions, -1 * actions)).astype('int')
    all_actions = []
    im_data = []
    for traj_idx in range(n_traj):
        cur_pos, seen = reset(n_agents, grid_n, allow_agent_overlap=False, circular=circular)
        os = []
        traj_actions = []
        for t in range(len_traj):
            action = actions[np.random.randint(len(actions))]
            os.append(cur_pos)
            traj_actions.append(action)

            if not circular:
                # clamp next position
                next_pos = (cur_pos + action)
                next_pos = np.amax((np.zeros_like(max_positions),
                                   np.amin((next_pos, max_positions), axis=0)), axis=0)
            else:
                next_pos = (cur_pos + action) % grid_n
            agent_moved = np.where(action != 0)[0][0] // 2
            agent_dim = AGENT_SIZES[agent_moved]

            # remove the current agent's occupied positions from seen
            agent_pos = cur_pos[2*agent_moved: 2*agent_moved+2]
            for x in range(agent_dim[0]):
                for y in range(agent_dim[1]):
                    disp = np.array([x, y])
                    seen.remove(tuple(disp + agent_pos))
            # check if action results in overlap with any other agent
            if not is_overlapping(agent_moved, seen, next_pos[2*agent_moved:2*agent_moved+2]):
                cur_pos = next_pos.astype('uint8')
            agent_new_pos = cur_pos[2 * agent_moved: 2 * agent_moved + 2]
            for x in range(agent_dim[0]):
                for y in range(agent_dim[1]):
                    disp = np.array([x, y])
                    seen.add(tuple(disp + agent_new_pos))

        traj_imgs = position_to_image(np.array(os), n_agents, grid_n)
        im_data.append(traj_imgs)
        all_actions.append(np.array(traj_actions))
    return np.array(im_data), np.array(all_actions)

def label_all_data_double(model, dataset, N):
  '''
  Labels all data in dataset for exactly 2 agents, where dataset consists of all possible states for size 16 grid
  model: CPC model
  data: dataset_single or dataset_double (with transitions)
  returns: all labels in order as np array size N**4
  '''
  assert N == 16, "not supported for grid_n != 16"
  model = model.cuda()
  num_factors = model.num_onehots
  data, _ = dataset # only look at anchors
  data = data[::8] # eliminate duplicate positions

  batch_size = 256
  i = 0
  data_size = len(data)
  all_labels = np.array([])
  all_labels_onehot = np.array([])
  while i < data_size:
    batch = from_numpy_to_var(np.transpose(data[i:i+batch_size], (0,3,1,2)).astype('float32'))
    zs = model.encode(batch,vis=True).cpu().numpy()
    z_labels = np.sum(np.array([zs[:,i] * model.z_dim ** (num_factors - i - 1) for i in range(num_factors)]), axis=0, dtype=int)
    if all_labels.any():
      all_labels = np.concatenate((all_labels, z_labels))
      all_labels_onehot = np.concatenate((all_labels_onehot, zs), axis=0)
    else:
      all_labels = z_labels
      all_labels_onehot = zs
    i += batch_size
  return all_labels, all_labels_onehot

def label_all_data_multi(model, dataset, N):
  '''
  Labels all data in dataset for > 2 agents
  '''
  model = model.cuda()
  num_factors = model.num_onehots
  anchors, positives = dataset # only look at anchors

  batch_size = 256
  i = 0
  data_size = len(anchors)
  all_labels_anchors, all_labels_positives = np.array([]), np.array([])
  all_labels_onehot_anchors, all_labels_onehot_positives = np.array([]), np.array([])
  while i < data_size:
    batch_anchors= from_numpy_to_var(np.transpose(anchors[i:i+batch_size], (0,3,1,2)).astype('float32'))
    zs_anchors = model.encode(batch_anchors,vis=True).cpu().numpy()
    z_labels_anchors = np.sum(
        np.array([zs_anchors[:, i] * model.z_dim ** (num_factors - i - 1) for i in range(num_factors)]), axis=0,
        dtype=int)

    batch_positives = from_numpy_to_var(np.transpose(positives[i:i + batch_size], (0, 3, 1, 2)).astype('float32'))
    zs_positives = model.encode(batch_positives, vis=True).cpu().numpy()
    z_labels_positives = np.sum(
        np.array([zs_positives[:, i] * model.z_dim ** (num_factors - i - 1) for i in range(num_factors)]), axis=0,
        dtype=int)

    if all_labels_anchors.any():
      all_labels_anchors = np.concatenate((all_labels_anchors, z_labels_anchors))
      all_labels_onehot_anchors = np.concatenate((all_labels_onehot_anchors, zs_anchors), axis=0)
      all_labels_positives = np.concatenate((all_labels_positives, z_labels_positives))
      all_labels_onehot_positives = np.concatenate((all_labels_onehot_positives, zs_positives), axis=0)
    else:
      all_labels_anchors = z_labels_anchors
      all_labels_onehot_anchors = zs_anchors
      all_labels_positives = z_labels_positives
      all_labels_onehot_positives = zs_positives
    i += batch_size
  return all_labels_anchors, all_labels_onehot_anchors, all_labels_positives, all_labels_onehot_positives

def label_to_idx(model, dataset, N, n_agents=2):
    '''
    Labels all data in dataset
    '''
    if n_agents == 2:
        all_labels, all_labels_onehot = label_all_data_double(model, dataset, N)
        label_to_idx = {}
        for idx, label in enumerate(all_labels):
            label_to_idx[label] = label_to_idx.get(label, []) + [idx]
        return label_to_idx, all_labels, all_labels_onehot
    elif n_agents >= 2:
        all_labels_anchors, all_labels_onehot_anchors, all_labels_positives, all_labels_onehot_positives = label_all_data_multi(model, dataset, N)
        label_to_idx_anchors = {}
        label_to_idx_positives = {}
        print("done labelling")
        sorted_anchor_labels = np.array(sorted(list(enumerate(all_labels_anchors)), key=lambda x: x[1]))
        sorted_positive_labels = np.array(sorted(list(enumerate(all_labels_positives)), key=lambda x: x[1]))
        start = 0
        end = 0
        while end < len(sorted_anchor_labels):
            label = sorted_anchor_labels[start][1]
            while end < len(sorted_anchor_labels) and sorted_anchor_labels[end][1] ==  sorted_anchor_labels[start][1]:
                end += 1
            label_to_idx_anchors[label] = sorted_anchor_labels[start:end]
            start = end
        start = 0
        end = 0
        while end < len(sorted_positive_labels):
            label = sorted_positive_labels[start][1]
            while end < len(sorted_positive_labels) and sorted_positive_labels[end][1] == sorted_positive_labels[start][1]:
                end += 1
            label_to_idx_positives[label] = sorted_positive_labels[start:end]
            start = end
        anchor_labels = (label_to_idx_anchors, all_labels_anchors, all_labels_onehot_anchors)
        positive_labels = (label_to_idx_positives, all_labels_positives, all_labels_onehot_positives)
        print("done label_to_idx")
        return anchor_labels, positive_labels

def process_batch_from_vp_grid(model, sample_ims, single=False):
    # images from vp model are tensors and rgb,
    # so they do not need to be preprocessed as in process_batch
    if single:
        return model.encode(sample_ims.unsqueeze(0), vis=True, rgb_input=True).squeeze(0).cpu().numpy()
    max_batch_size=128
    idx = 0
    z_labels = []
    while idx < len(sample_ims):
        zs = model.encode(sample_ims[idx:idx+max_batch_size], vis=True, rgb_input=True).cpu().numpy()
        z_labels.append(zs)
        idx += max_batch_size
    return np.concatenate(z_labels)

def process_batch(model, sample_ims, single=False):
    '''
    Computes and returns forward pass of CPC model for a batch of processed images
    :param model: CPC model
    :param sample_ims: batch of input images (any length)
    :return: np array of z outputs [sample_size, model.z_dim]
    '''
    if single:
        return model.encode(from_numpy_to_var(sample_ims).unsqueeze(0).permute(0,3,1,2), vis=True).squeeze(0).cpu().numpy()
    max_batch_size=128
    idx = 0
    z_labels = []
    while idx < len(sample_ims):
        zs = model.encode(from_numpy_to_var(sample_ims[idx:idx+max_batch_size]).permute(0, 3, 1, 2), vis=True).cpu().numpy()
        z_labels.append(zs)
        idx += max_batch_size
    return np.concatenate(z_labels)

def get_labels(model, im):
    '''
    returns labels for model.encode(im), converted from onehot to int
    '''
    z_labels = process_batch(model, im)
    zs = tensor_to_label_array(z_labels, model.num_onehots, model.z_dim)
    return zs

def get_labels_uncompressed(model, im):
    '''
    returns onehot labels model.encode(im)
    '''
    z_labels = process_batch(model, im)
    return z_labels

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
    final_label[num_factors-len(output):] = output
    return final_label.astype('uint8')

def tensor_to_label_array(idx_array, num_factors, z_dim):
    '''
    Convert a batch of idx_array to labels
    :param idx_array: batch_size x z_dim
    :param num_factors: the number of onehots
    :param z_dim:
    :return: labels: 1D array of size batch_size
    '''
    return (np.tile(np.power(z_dim, np.flip(np.arange(num_factors)))[None], (idx_array.shape[0],1)) * idx_array).sum(1)

def get_hamming_dists_double(all_labels_onehot, grid_n):
    '''

    :param all_labels_onhot: [dataset_size x 2]
    :return: [dataset_size]
    '''
    actions = [[0, 1, 0, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]]
    z = all_labels_onehot.reshape((grid_n, grid_n, grid_n, grid_n, 2))
    z_next = []
    for i in range(grid_n):
        for j in range(grid_n):
            for k in range(grid_n):
                for l in range(grid_n):
                    action = actions[np.random.randint(len(actions))]
                    next_idx = (np.array([i,j,k,l]) + action) % grid_n
                    z_next.append(z[tuple(next_idx)])
    z_next = np.array(z_next)
    z = all_labels_onehot
    distances = np.sum((z != z_next), axis=1)
    return distances

def get_hamming_dists_multi(all_labels_onehot_anchors, all_labels_onehot_positives):
    sample = np.arange(0, len(all_labels_onehot_anchors), 10)
    z, z_next = all_labels_onehot_anchors[sample], all_labels_onehot_positives[sample]
    distances = np.sum((z != z_next), axis=1)
    return distances

def get_hamming_dists_samples(model, n_agents, grid_n, n_batches=64, batch_size=512):
    distances = np.array([])
    for batch in range(n_batches):
        anchors, positives = sample_and_process_transitions(n_agents, batch_size, grid_n)
        anchors = from_numpy_to_var(np.transpose(anchors, (0,3,1,2)))
        positives = from_numpy_to_var(np.transpose(positives, (0, 3, 1, 2)))
        z = model.encode(anchors, vis=True).cpu().numpy()
        z_next = model.encode(positives, vis=True).cpu().numpy()
        distances_batch = np.sum((z != z_next), axis=1)
        if not distances.any():
            distances = distances_batch
        else:
            distances = np.concatenate((distances, distances_batch))
    print("hamming dist shape", distances.shape)
    return distances

def addjust_to_middle(labels, N):
  for dim in range(len(labels.shape)):
    for i in range(N-1):
      if labels.take(i, axis=dim).sum() == 0 and labels.take(i+1, axis=dim).sum() == 0:
          labels = np.concatenate([labels.take(range(i+1, N), axis=dim),
                                   labels.take(range(i+1), axis=dim)],
                                  axis=dim)
  return labels

def find_top_left(array2d):
    assert array2d.shape[0] == array2d.shape[1]
    assert len(array2d.shape) == 2
    grid_n = array2d.shape[0]
    shift_by = np.zeros(2)
    for dim in range(2):
        print(dim)
        for i in range(grid_n-1):
            if array2d.take(i, axis=dim).sum() == 0 and array2d.take(i+1, axis=dim).sum() == 0:
                array2d = np.concatenate([array2d.take(range(i+1, grid_n), axis=dim),
                                          array2d.take(range(i+1), axis=dim)],
                                         axis=dim)
                shift_by[dim] = i+1
                break
    positions = np.where(array2d>0)
    position_x = (positions[0][0] + shift_by[0]) % grid_n
    position_y = (positions[1][0] + shift_by[1]) % grid_n
    return np.array([position_x, position_y]).astype('int')