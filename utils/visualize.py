import matplotlib.pyplot as plt
from envs.gridworld import AGENT_SIZES, RGB_COLORS
import numpy as np
import os
from envs.key_door import *
from envs.gridworld import *

from utils.gen_utils import *
from model import get_discrete_representation
from torchvision.utils import save_image

def visualize_representations(env, model):
    # 1. visualize clustering if number of agents is 1 or 2
    # 2. visualize factorization with histogram
    # 3. visualize hamming distance graphs
    figs = None
    if env.name == 'gridworld':
        if env.n_agents == 1:
            figs = visualize_clusters_online_single(env, model)
        elif env.n_agents == 2:
            figs = visualize_clusters_online_double(env, model)
    if env.name == 'key-wall' or env.name == 'key-corridor':
        figs = visualize_single_agent_and_key(env, model)
    return figs

def visualize_clusters_online_single(env, model):
    assert env.n_agents == 1
    num_factors = model.num_onehots
    rows = np.arange(0, env.grid_n, env.grid_n//16)
    cols = np.arange(0, env.grid_n, env.grid_n//16)
    cluster_map = []
    for c in cols:
        batch_pos = np.transpose(np.stack((np.repeat(c, len(rows)), rows)), (1,0))
        im_torch = np_to_var(position_to_image(batch_pos, env.n_agents, env.grid_n)).permute(0,3,1,2)
        zs = model.encode(im_torch, vis=True).cpu().numpy()
        z_labels = np.sum(np.array([zs[:, i] * model.z_dim ** (num_factors - i - 1) for i in range(num_factors)]),
                          axis=0, dtype=int)
        cluster_map.append(z_labels)
    cluster_map = np.array(cluster_map)
    print("cluster map")
    print(cluster_map)
    fig = plt.figure()
    plt.imshow(cluster_map, cmap = 'gist_rainbow')
    return fig

def visualize_clusters_online_double_fix_one_agent(model, n_agents, grid_n, fixed_agent=0):
    fig = plt.figure()
    rows = np.arange(0, grid_n, grid_n // 16)
    cols = np.arange(0, grid_n, grid_n // 16)
    fixed_poses = np.arange(0, grid_n, grid_n // 4)
    n_subplots = 4
    plot_idx = 1
    num_factors = model.num_onehots

    onehots_0 = []
    onehots_1 = []

    for idx, fixed_row in enumerate(fixed_poses):
        for fixed_col in fixed_poses:
            fixed_pos = np.tile(np.array((fixed_row, fixed_col)), (len(cols), 1))
            cluster_map = []
            oh0_map = []
            oh1_map = []
            for c in cols:
                batch_pos = np.transpose(np.stack((np.repeat(c, len(rows)), rows)), (1, 0))
                batch_pos = np.hstack((batch_pos, fixed_pos)) if fixed_agent == 0 else np.hstack((fixed_pos, batch_pos))
                im_torch = np_to_var(position_to_image(batch_pos, n_agents, grid_n)).permute(0, 3, 1, 2)
                zs = model.encode(im_torch, vis=True).cpu().numpy()
                oh0_map.append(zs[:,0])
                oh1_map.append(zs[:,1])
                z_labels = np.sum(
                    np.array([zs[:, i] * model.z_dim ** (num_factors - i - 1) for i in range(num_factors)]),
                    axis=0, dtype=int)
                cluster_map.append(z_labels)
            cluster_map = np.array(cluster_map)
            ax = fig.add_subplot(n_subplots, n_subplots, plot_idx)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                           labelleft=False)
            print("cluster map %d fixing agent %d" % (idx, fixed_agent))
            print(cluster_map)
            plt.imshow(cluster_map, cmap='gist_rainbow')
            onehots_0.append(np.array(oh0_map))
            onehots_1.append(np.array(oh1_map))
            plot_idx += 1

    onehot_0_fig = plot_one_onehot_2agents(onehots_0, n_subplots**2)
    onehot_1_fig = plot_one_onehot_2agents(onehots_1, n_subplots**2)

    return fig, onehot_0_fig, onehot_1_fig

def visualize_clusters_online_double(env, model):
    plot_0, onehot_0_fig_0, onehot_1_fig_0 = visualize_clusters_online_double_fix_one_agent(model, env.n_agents, env.grid_n, fixed_agent=0)
    plot_1, onehot_0_fig_1, onehot_1_fig_1 = visualize_clusters_online_double_fix_one_agent(model, env.n_agents, env.grid_n, fixed_agent=1)

    return plot_0, onehot_0_fig_0, onehot_1_fig_0, plot_1, onehot_0_fig_1, onehot_1_fig_1

def position_to_image(positions, n_agents, grid_n):
    '''
    Converts batch of agent xy positions to batch of rgb images

    :param positions: batch of agent positions
    :param n_agents: # agents
    :param grid_n: grid size
    :return: [sample_size, grid_n, grid_n, n_agents] images
    '''
    if grid_n not in [16,64]:
        raise NotImplementedError("Grid size not supported: %d" % grid_n)
    n_samples = positions.shape[0]
    ims = np.zeros((n_samples, grid_n, grid_n, 3))
    for i in range(n_agents):
        agent_dim = AGENT_SIZES[i]
        x_cur, y_cur = positions[:, 2*i], positions[:, 2*i+1]
        if grid_n in [16, 64]:
            for x in range(agent_dim[0]):
                for y in range(agent_dim[1]):
                    ims[np.arange(n_samples),
                        (x_cur+x) % grid_n,
                        (y_cur+y) % grid_n] += np.tile(np.array(list(RGB_COLORS.values())[i]), (n_samples,1))
    return ims

def visualize_attn_map(amaps): # amap: B X 1 X W X H
    fig = plt.figure()
    n_subplots = len(amaps) # should be 16
    assert n_subplots == 16, "number of attention map samples should be 16"
    for i in range(len(amaps)):
        activations = amaps[i][0].cpu().detach().numpy()
        ax = fig.add_subplot(4, 4, i+1)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                       labelleft=False)
        plt.imshow(activations, cmap='Greys')
    return fig

def plot_one_onehot_2agents(onehot_labels, n):
    fig = plt.figure()
    plot_idx = 1
    n_subplots = 4
    for labels in onehot_labels:
        grid = labels.reshape(n,n)
        ax = fig.add_subplot(n_subplots, n_subplots, plot_idx)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False,
                       labelleft=False)
        plt.imshow(grid, cmap='gist_rainbow')
        plot_idx+=1
    return fig

def sample_trajectory(env, len_traj=400, choose_agent_i=0):
    '''
    Samples 1 trajectory length len_traj, moving agent choose_agent_i, fixing all other agents
    :param n_agents: total # agents
    :param n: grid width
    :param n_samples: number of trajectories to samples
    :param choose_agent_i: agent to move
    :return: np array of positions along trajectory [len_traj, 2*n_agents]
    '''
    actions = np.eye(env.n_agents*2)
    actions = np.concatenate((actions, -1 * actions))
    actions = np.concatenate((actions[choose_agent_i*2: choose_agent_i*2+2],
                              actions[env.n_agents*2+choose_agent_i*2: env.n_agents*2+choose_agent_i*2+2]))
    os = []
    env.reset()
    for i in range(len_traj):
        action = actions[np.random.randint(len(actions))]
        env.step(action)
        os.append(env.get_obs())
    os = np.array(os)
    return os.transpose((0, 3, 1, 2))

def get_factorization_hist(env, model, len_traj=600, n_traj=10):
    if env.name == 'gridworld':
        return test_factorization_fix_agents(env, model, len_traj, n_traj)
    elif env.name == 'key-wall' or env.name == 'key-corridor':
        return test_factorization_single_agent_key(env, model)

def test_factorization_fix_agents(env, model, len_traj=600, n_traj=10):
    '''
    Samples random trajectories moving one agent at a time, returns two lists of histograms of hamming distances and onehot changes
    by moving each agent. Histogram lists should be written to tensorboard

    :param model: CPC model
    :param n_agents: # agents
    :param grid_n: grid width
    :param len_traj: length of trajectories to sample
    :param n_traj: number of trajectories to sample
    :return: tuple of two lists of histogram figs (plt.fig), each length n_agents
                1. list of histograms of hamming distances
                2. list of histograms of onehot changes, moving one agent at a time
    '''
    hamming_hist_lst = []
    onehot_hist_lst = []
    for i in range(env.n_agents):
        dist_onehots = []
        for n in range(n_traj):
            dist_onehots = []
            x = sample_trajectory(env, len_traj=len_traj, choose_agent_i=i)
            # y = position_to_image(x, env.n_agents, env.grid_n)

            zs = get_discrete_representation(model, x)
            dist_hamming = np.sum(zs[1:] - zs[:-1] != 0, axis=1)
            prev_z = zs[0]
            for zlabel in zs[1:]:
                for k in range(model.num_onehots):
                    if zlabel[k] != prev_z[k]:
                        dist_onehots.append(k)
                prev_z = zlabel

        hammings_hist = plt.figure()
        plt.hist(dist_hamming)
        plt.ylabel("hamming distance distribution moving only agent %d" % i)
        hamming_hist_lst.append(hammings_hist)

        onehots_hist = plt.figure()
        print("dist_onehots", dist_onehots)
        plt.hist(dist_onehots, bins=np.arange(model.num_onehots + 1))
        plt.ylabel("onehot changes only moving agent %d" % i)
        onehot_hist_lst.append(onehots_hist)

    return hamming_hist_lst, onehot_hist_lst

def save_plots(savepath, losses, est_lbs):
    plt.plot(losses)
    plt.ylabel('C_loss')
    loss_path = os.path.join(savepath, "loss.png")
    plt.savefig(loss_path)

    plt.plot(est_lbs)
    plt.ylabel('estimated lowerbound')
    lb_path = os.path.join(savepath, "est_lowerbounds.png")
    plt.savefig(lb_path)

    plt.close('all')

def visualize_node(node, all_labels, dataset, grid_n, savepath):
    # visualize some samples
    idx = np.where(all_labels == node)[0]
    anchors = dataset[0][::8]
    node_samples = anchors[idx][:64]
    np.random.shuffle(node_samples)
    node_samples = np.concatenate((node_samples, np.zeros((len(node_samples), grid_n, grid_n, 1))), axis=3)
    samples_tensor = np_to_var(node_samples).permute(0,3,1,2)
    save_image(samples_tensor, os.path.join(savepath, "node_%d_samples.png" % node), padding=16)

def get_2agents_density(node, labels, dataset):
    idx = np.where(labels == node)[0]
    anchors = dataset[0][::8][idx]
    sum_positions = anchors.sum(axis=0)
    agent_0_pos, agent_1_pos = sum_positions[:,:, 0], sum_positions[:,:,1]
    return agent_0_pos/agent_0_pos.max(), agent_1_pos/agent_1_pos.max()

def visualize_density_failed_2agents(cur_pos, cur_node, node_to_go, labels, dataset, savepath, epoch):
    # visualize where execute_plan fails
    agent_0_dist, agent_1_dist = get_2agents_density(cur_node, labels, dataset)
    agent_0_dist_next, agent_1_dist_next = get_2agents_density(node_to_go, labels, dataset)
    agent_0_cur_pos = np.zeros(agent_0_dist.shape)
    agent_0_cur_pos[cur_pos[0], cur_pos[1]] = 1
    agent_1_cur_pos = np.zeros(agent_1_dist.shape)
    agent_1_cur_pos[cur_pos[2], cur_pos[3]] = 1
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    im0 = np.stack([agent_0_dist, agent_0_dist_next, agent_0_cur_pos], axis=2)
    plt.imshow(im0)
    ax = fig.add_subplot(1, 2, 2)
    im1 = np.stack([agent_1_dist, agent_1_dist_next, agent_1_cur_pos], axis=2)
    plt.imshow(im1)

    fname = "epoch_%d_failed_to_leave_node_%d" % (epoch, cur_node)
    plt.savefig(os.path.join(savepath, fname))


def test_factorization_single_agent_key(env, model, n_traj=10, len_traj=200):
    '''
    move agent with and with key, count onehot changes for
        1. Any agent movement with or without key, not changing key state within trajectory
        2. Fixing agent position, placing/taking away key
    '''

    onehot_hist_lst = []

    # 1. -------------- test factorization of agent
    dist_onehots_a = []
    for traj in range(n_traj):
        env.reset()
        traj_with_key = env.sample_random_trajectory(len_traj, interact_with_key=False)
        zs = get_discrete_representation(model, traj_with_key)

        prev_z = zs[0]
        for zlabel in zs[1:]:
            for k in range(model.num_onehots):
                if zlabel[k] != prev_z[k]:
                    dist_onehots_a.append(k)
            prev_z = zlabel

        env.remove_all_keys()
        traj_no_key = env.sample_random_trajectory(len_traj, interact_with_key=False)
        zs = get_discrete_representation(model, traj_no_key)
        prev_z = zs[0]
        for zlabel in zs[1:]:
            for k in range(model.num_onehots):
                if zlabel[k] != prev_z[k]:
                    dist_onehots_a.append(k)
            prev_z = zlabel

    onehots_hist = plt.figure()
    print("dist_onehots for moving agent", dist_onehots_a)
    plt.hist(dist_onehots_a, bins=np.arange(model.num_onehots + 1))
    plt.ylabel("onehot changes only moving agent")
    onehot_hist_lst.append(onehots_hist)

    # 2. ----------------- test factorization of key(s)

    for key_idx in range(env.n_keys):
        dist_onehots_k = []
        for i in range(len_traj):
            env.reset()
            z_key = get_discrete_representation(model, env.get_obs(), single=True)
            env.remove_key(key_idx, 0)
            z_no_key = get_discrete_representation(model, env.get_obs(), single=True)
            for k in range(model.num_onehots):
                if z_no_key[k] != z_key[k]:
                    dist_onehots_k.append(k)
        onehots_hist = plt.figure()
        print("dist_onehots for changing key %d" % key_idx, dist_onehots_k)
        plt.hist(dist_onehots_k, bins=np.arange(model.num_onehots + 1))
        plt.ylabel("onehot changes only placing/removing key %d (fixing agent)" % key_idx)
        onehot_hist_lst.append(onehots_hist)
    return onehot_hist_lst


def visualize_single_agent_and_key(env, model):
    # try to place agent at each position, with and without key on the grid
    assert isinstance(env, KeyWall) or isinstance(env, KeyCorridor)
    map_key = np.full((GRID_N, GRID_N), -1)
    map_no_key = np.full((GRID_N, GRID_N), -1)

    env.reset()
    for x in range(GRID_N):
        for y in range(GRID_N):
            pos = (x,y)
            obs = env.get_obs()
            if env.try_place_agent(pos):
                if model.encoder_form == 'cswm-key-gt':
                    z = model.encode((np_to_var(obs).unsqueeze(0).permute(0, 3, 1, 2), 1), vis=True)
                else:
                    z = model.encode(np_to_var(obs).unsqueeze(0).permute(0, 3, 1, 2), vis=True)
                z_label = tensor_to_label(z[0], model.num_onehots, model.z_dim)
                map_key[pos] = z_label

    env.remove_all_keys()
    for x in range(GRID_N):
        for y in range(GRID_N):
            for y in range(GRID_N):
                pos = (x, y)
                obs = env.get_obs()
                if env.try_place_agent(pos):
                    if model.encoder_form == 'cswm-key-gt':
                        z = model.encode((np_to_var(obs).unsqueeze(0).permute(0, 3, 1, 2), 0), vis=True)
                    else:
                        z = model.encode(np_to_var(obs).unsqueeze(0).permute(0, 3, 1, 2), vis=True)
                    z_label = tensor_to_label(z[0], model.num_onehots, model.z_dim)
                    map_no_key[pos] = z_label

    print("map with key")
    print(map_key)
    print()
    print("map no key")
    print(map_no_key)

    fig0 = plt.figure()
    plt.imshow(map_key, cmap='gist_rainbow')
    fig1 = plt.figure()
    plt.imshow(map_no_key, cmap='gist_rainbow')

    return fig0, fig1
