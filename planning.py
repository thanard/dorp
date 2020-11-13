import scipy
import networkx as nx
from networkx import NetworkXException
from utils.gen_utils import *
from utils.dataset import *
from model import get_discrete_representation

def create_graph_from_sample_transitions_full(model, env, n_batches=4, batch_size=256, len_traj=400):
    '''
    Creates graphs from sampled transition data
    :param model: CPC model
    :param grid_n: grid width
    :param n_agents: # agents
    :param n_batches: # batches of transtion data to process
    :param batch_size: size of each batch of transition data
    :return: tuple (explicit nx graph, list of onehot nx graphs)
    '''
    graph = nx.Graph()
    for i in range(n_batches):
        if i % 1 == 0:
            print("processing batch %d" % i)
        data = get_sample_transitions(env, batch_size, len_traj)
        im_cur, im_next = data[:,0], data[:,1]
        zs = model.encode(np_to_var(im_cur).permute(0,3,1,2), vis=True).cpu().numpy()
        z_next = model.encode(np_to_var(im_next).permute(0,3,1,2), vis=True).cpu().numpy()
        for idx, z in enumerate(zs):
            z_label = tensor_to_label(z, model.num_onehots, model.z_dim)
            z_pos = z_next[idx]
            z_next_label = tensor_to_label(z_pos, model.num_onehots, model.z_dim)
            # add node to explicit full graph
            if not graph.has_node(z_label):
                graph.add_node(z_label)

            if z_label != z_next_label and not graph.has_edge(z_label, z_next_label):
                graph.add_edge(z_label, z_next_label)

    print("number of nodes in full explicit graph", len(graph))
    return graph

def create_graph_from_sample_transitions_factorized(model, env, n_batches=4, batch_size=256, len_traj=400):
    onehot_graphs = [nx.Graph() for _ in range(model.num_onehots)]
    for i in range(n_batches):
        if i % 1 == 0:
            print("processing batch %d" % i)
        data = get_sample_transitions(env, batch_size, len_traj)
        im_cur, im_next = data[:,0], data[:,1]
        zs = get_discrete_representation(model, im_cur)
        # zs = model.encode(np_to_var(im_cur).permute(0,3,1,2), vis=True).cpu().numpy()
        z_next = get_discrete_representation(model, im_next)
        # z_next = model.encode(np_to_var(im_next).permute(0,3,1,2), vis=True).cpu().numpy()
        for idx, z in enumerate(zs):
            z_pos = z_next[idx]
            # add node to onehot graphs (for factorized planning)
            for onehot_idx, onehot_label in enumerate(z):
                subgraph = onehot_graphs[onehot_idx]
                onehot_z_pos = z_pos[onehot_idx]
                if not subgraph.has_node(onehot_label):
                    subgraph.add_node(onehot_label)
                if onehot_label != onehot_z_pos and not subgraph.has_edge(onehot_label, onehot_z_pos):
                    subgraph.add_edge(onehot_label, onehot_z_pos)
    for i in range(len(onehot_graphs)):
        print("number of nodes in graph for onehot %d" % i, len(onehot_graphs[i]))
    return onehot_graphs

def create_graph_from_sample_transitions_grouped_onehots(model,
                                                          env,
                                                          groups,
                                                          n_batches=4,
                                                          batch_size=256,
                                                          len_traj=400):
    assert sum(groups) == model.num_onehots
    onehot_graphs = [nx.Graph() for _ in range(len(groups))]
    for i in range(n_batches):
        if i % 1 == 0:
            print("processing batch %d" % i)
        data = get_sample_transitions(env, batch_size, len_traj)
        im_cur, im_next = data[:, 0], data[:, 1]
        zs = model.encode(np_to_var(im_cur).permute(0, 3, 1, 2), vis=True).cpu().numpy()
        z_next = model.encode(np_to_var(im_next).permute(0, 3, 1, 2), vis=True).cpu().numpy()

        for idx, z in enumerate(zs):
            zs_grouped = tensor_to_label_grouped(z, model.z_dim, groups)
            z_pos = z_next[idx]
            z_pos_grouped = tensor_to_label_grouped(z_pos, model.z_dim, groups)

            for onehot_idx, (z_onehot, z_next_onehot) in enumerate(zip(zs_grouped, z_pos_grouped)):
                subgraph = onehot_graphs[onehot_idx]
                if not subgraph.has_node(z_onehot):
                    subgraph.add_node(z_onehot)
                if z_onehot != z_next_onehot and not subgraph.has_edge(z_onehot, z_next_onehot):
                    subgraph.add_edge(z_onehot, z_next_onehot)

    for i in range(len(onehot_graphs)):
        print("number of nodes in graph for onehot %d" % i, len(onehot_graphs[i]))
    return onehot_graphs

def get_plan(graph, start, goal):
    '''
        returns shortest path from start to goal in path if it exists, otherwise returns None
        '''
    try:
        path = nx.shortest_path(graph, source=start, target=goal)
        return path
    except Exception as e:
        print("shortest path failed: ", e)
        return None

def plan_to_goal_and_execute_full_graph(actor, env, full_graph, oracle=False):
    model = actor.cpc_model
    full_graph_copy = full_graph.copy()
    zstart = tensor_to_label(get_discrete_representation(model, env.get_obs(), single=True),
                             model.num_onehots,
                             model.z_dim)
    zgoal = tensor_to_label(get_discrete_representation(actor.cpc_model, env.goal_im, single=True),
                            model.num_onehots,
                            model.z_dim)
    plan = get_plan(full_graph_copy, zstart, zgoal)
    prev_node = plan[0]
    node = plan[0]
    reached_node = True
    while plan:
        for node in plan[1:]:
            reached_node = actor.act(env, node, oracle=oracle)
            if not reached_node:
                break
            prev_node = node
        if not reached_node:
            # remove edge and replan
            full_graph_copy.remove_edge(prev_node, node)
            cur_z = tensor_to_label(get_discrete_representation(actor.cpc_model, env.get_obs(), single=True),
                                    model.num_onehots,
                                    model.z_dim)
            prev_node = cur_z
            plan = get_plan(full_graph_copy, cur_z, zgoal)
        else:
            return actor.move_to_goal(env, oracle=oracle)
    return False

def plan_to_goal_and_execute_factorized(actor, env, onehot_graphs, oracle=False):
    model = actor.cpc_model
    # fully factorized planning
    plan_success = [False for _ in range(model.num_onehots)]
    for idx, graph in enumerate(onehot_graphs):
        onehot_graph_copy = onehot_graphs[idx].copy()
        zstart = get_discrete_representation(model, env.get_obs(), single=True)[idx]
        zgoal = get_discrete_representation(actor.cpc_model, env.goal_im, single=True)[idx]
        onehot_plan = get_plan(onehot_graph_copy, zstart, zgoal)
        prev_node = onehot_plan[0]
        node = onehot_plan[0]
        reached_node = True
        while onehot_plan:
            print("plan for onehot %d: " % idx, onehot_plan)
            for node in onehot_plan[1:]:
                reached_node = actor.act(env, node, onehot_idx=idx, oracle=oracle)
                if not reached_node:
                    print("failed to reach node", node)
                    break
                prev_node = node
                print("reached node", node)
            if not reached_node:
                # remove edge and replan
                onehot_graph_copy.remove_edge(prev_node, node)
                cur_z = get_discrete_representation(actor.cpc_model, env.get_obs(), single=True)[idx]
                prev_node = cur_z
                onehot_plan = get_plan(onehot_graph_copy, cur_z, zgoal)
            else:
                plan_success[idx] = True
                break
    if np.sum(plan_success) == len(plan_success):
        return actor.move_to_goal(env, oracle=oracle)
    print("plan failed\n")
    return False

def plan_to_goal_and_execute_grouped(actor, env, grouped_graphs, onehot_groups, oracle=False):
    model = actor.cpc_model
    plan_success = [False for _ in range(len(onehot_groups))]

    for idx, graph in enumerate(grouped_graphs):
        grouped_graph_copy = grouped_graphs[idx].copy()
        zstart = tensor_to_label_grouped(get_discrete_representation(model, env.get_obs(), single=True),
                                         model.z_dim, onehot_groups)[idx]
        zgoal = tensor_to_label_grouped(get_discrete_representation(actor.cpc_model, env.goal_im, single=True),
                                        model.z_dim, onehot_groups)[idx]
        plan = get_plan(grouped_graph_copy, zstart, zgoal)
        prev_node = plan[0]
        node = plan[0]
        reached_node = True
        while plan:
            for node in plan[1:]:
                reached_node = actor.act(env, node, onehot_idx=idx, groups=onehot_groups, oracle=oracle)
                if not reached_node:
                    break
                prev_node = node
            if not reached_node:
                # remove edge and replan
                grouped_graph_copy.remove_edge(prev_node, node)
                cur_z = tensor_to_label_grouped(get_discrete_representation(
                    actor.cpc_model,
                    env.get_obs(),
                    single=True), model.z_dim, onehot_groups)[idx]
                prev_node = cur_z
                plan = get_plan(grouped_graph_copy, cur_z, zgoal)
            else:
                plan_success[idx] = True
    if np.sum(plan_success) == len(plan_success):
        return actor.move_to_goal(env, oracle=oracle)
    return False

def get_planning_success_rate(actor, env, n_trials, factorized=False, onehot_groups=None, oracle=False):
    success = 0
    if onehot_groups:
        grouped_graphs = create_graph_from_sample_transitions_grouped_onehots(actor.cpc_model, env, onehot_groups)
        for trial in range(n_trials):
            print("Trial %d of %d" % (trial, n_trials))
            env.reset()
            if plan_to_goal_and_execute_grouped(actor, env, grouped_graphs, onehot_groups, oracle=oracle):
                print("success")
                success += 1
    elif factorized:
        onehot_graphs = create_graph_from_sample_transitions_factorized(actor.cpc_model, env)
        for trial in range(n_trials):
            print("Trial %d of %d" % (trial, n_trials))
            env.reset()
            if plan_to_goal_and_execute_factorized(actor, env, onehot_graphs, oracle=oracle):
                print("success\n")
                success += 1
    else:
        full_graph = create_graph_from_sample_transitions_full(actor.cpc_model, env)
        for trial in range(n_trials):
            print("Trial %d of %d" % (trial, n_trials))
            env.reset()
            if plan_to_goal_and_execute_full_graph(actor, env, full_graph, oracle=oracle):
                print("success")
                success += 1
    rate = float(success)/n_trials
    print("success rate: %.3f" % rate)
    return rate




