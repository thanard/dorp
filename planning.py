import networkx as nx
import copy
from utils.gen_utils import *
from utils.dataset import *
from model import get_discrete_representation


# def create_graph_from_sample_transitions_full(model, env, n_traj=50, len_traj=1200, dataset=None):
#     '''
#     Creates graphs from sampled transition data
#     :param model: CPC model
#     :param grid_n: grid width
#     :param n_agents: # agents
#     :param n_batches: # batches of transtion data to process
#     :param batch_size: size of each batch of transition data
#     :return: tuple (explicit nx graph, list of onehot nx graphs)
#     '''
#     print("creating graph....")
#     graph = nx.Graph()
#     if dataset:
#         data = dataset
#         n_traj = len(dataset)
#         len_traj = len(dataset[0])
#     else:
#         data = get_sample_transitions(env, n_traj, len_traj)
#     for idx in range(n_traj):
#         traj = data[idx]
#         zs = get_discrete_representation(model, traj)
#         for i in range(len_traj-1):
#             z_cur = tensor_to_label(zs[i], model.num_onehots, model.z_dim)
#             z_next = tensor_to_label(zs[i + 1], model.num_onehots, model.z_dim)
#             if not graph.has_node(z_cur):
#                 graph.add_node(z_cur)
#             if z_next != z_cur and not graph.has_edge(z_cur, z_next):
#                 graph.add_edge(z_cur, z_next)
#
#     print("number of nodes in full explicit graph", len(graph))
#     print("nodes", graph.nodes())
#     print("edges", graph.edges())
#     return graph
#
# def create_graph_from_sample_transitions_factorized(model, env, n_traj=50, len_traj=400, dataset=None):
#     print("creating graphs....")
#     onehot_graphs = [nx.Graph() for _ in range(model.num_onehots)]
#     if dataset:
#         data = dataset
#         n_traj = len(dataset)
#         len_traj = len(dataset[0])
#     else:
#         data = get_sample_transitions(env, n_traj, len_traj)
#     for idx in range(n_traj):
#         traj = data[idx]
#         zs = get_discrete_representation(model, traj)
#         for i in range(len_traj - 1):
#             z_cur = zs[i]
#             z_next = zs[i + 1]
#
#             for onehot_idx, onehot_label in enumerate(z_cur):
#                 subgraph = onehot_graphs[onehot_idx]
#                 onehot_z_pos = z_next[onehot_idx]
#                 if not subgraph.has_node(onehot_label):
#                     subgraph.add_node(onehot_label)
#                 if onehot_label != onehot_z_pos and not subgraph.has_edge(onehot_label, onehot_z_pos):
#                     subgraph.add_edge(onehot_label, onehot_z_pos)
#     for i in range(len(onehot_graphs)):
#         print("number of nodes in graph for onehot %d" % i, len(onehot_graphs[i]))
#     return onehot_graphs
#
# def create_graph_from_sample_transitions_grouped_onehots(model,
#                                                           env,
#                                                           groups,
#                                                           n_traj=50,
#                                                           len_traj=400,
#                                                          dataset=None):
#     print("creating graphs....")
#     assert sum(groups) == model.num_onehots
#     onehot_graphs = [nx.Graph() for _ in range(len(groups))]
#     if dataset:
#         data = dataset
#         n_traj = len(dataset)
#         len_traj = len(dataset[0])
#     else:
#         data = get_sample_transitions(env, n_traj, len_traj)
#     for idx in range(n_traj):
#         traj = data[idx]
#         zs = get_discrete_representation(model, traj)
#         for i in range(len_traj - 1):
#             z_grouped = tensor_to_label_grouped(zs[i], model.z_dim, groups)
#             z_next_grouped = tensor_to_label_grouped(zs[i+1], model.z_dim, groups)
#
#             for onehot_idx, (z_onehot, z_next_onehot) in enumerate(zip(z_grouped, z_next_grouped)):
#                 subgraph = onehot_graphs[onehot_idx]
#                 if not subgraph.has_node(z_onehot):
#                     subgraph.add_node(z_onehot)
#                 if z_onehot != z_next_onehot and not subgraph.has_edge(z_onehot, z_next_onehot):
#                     subgraph.add_edge(z_onehot, z_next_onehot)
#
#     for i in range(len(onehot_graphs)):
#         print("number of nodes in graph for onehot group %d" % i, len(onehot_graphs[i]))
#     return onehot_graphs

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

def plan_and_execute_full_graph(actor, env, graph, n_trials, oracle=False):
    model = actor.cpc_model
    import time
    start = time.time()
    envs = [copy.deepcopy(env) for i in range(n_trials)]
    start_ims = []
    goal_ims = []
    del env
    for trial in range(n_trials):
        env = envs[trial]
        env.seed(trial)
        env.reset()
        start_ims.append(env.get_obs().transpose(2, 0, 1))
        goal_ims.append(env.goal_im.transpose(2, 0, 1))
    print("Data collection: ", time.time() - start)
    start = time.time()

    all_ims = np.concatenate([start_ims, goal_ims])
    zs_np = get_discrete_representation(model,
                                        all_ims)
    import pdb
    pdb.set_trace()
    nodes = np.ravel_multi_index
    # zgoal = tensor_to_label(get_discrete_representation(model,
    #                                                     goal_ims,
    #                                                     single=True),
    #                         model.num_onehots,
    #                         model.z_dim)
    print("Embedding: ", time.time() - start)
    start = time.time()

    plan = graph.get_plan(zstart, zgoal)
    print("Planning: ", time.time() - start)
    start = time.time()

def plan_to_goal_and_execute_full_graph(actor, env, full_graph, oracle=False):
    model = actor.cpc_model
    full_graph_copy = full_graph.copy()
    zstart = tensor_to_label(get_discrete_representation(model, env.get_obs().transpose(2, 0, 1), single=True),
                             model.num_onehots,
                             model.z_dim)
    zgoal = tensor_to_label(get_discrete_representation(actor.cpc_model, env.goal_im.transpose(2, 0, 1), single=True),
                            model.num_onehots,
                            model.z_dim)
    total_steps = 0
    plan = get_plan(full_graph_copy, zstart, zgoal)
    if plan:
        prev_node = plan[0]
        node = plan[0]
        reached_node = True

    while plan:
        print("plan from %d to %d: " % (zstart, zgoal), plan)
        for node in plan[1:]:
            reached_node, steps  = actor.act(env, node, oracle=oracle)
            total_steps += steps
            if not reached_node:
                break
            prev_node = node
            print("reached node", node)
        if not reached_node:
            # remove edge and replan
            print("Couldn't reach node %d" % node)
            print("Trying to replan....")
            full_graph_copy.remove_edge(prev_node, node)
            cur_z = tensor_to_label(get_discrete_representation(actor.cpc_model, env.get_obs(), single=True),
                                    model.num_onehots,
                                    model.z_dim)
            prev_node = cur_z
            plan = get_plan(full_graph_copy, cur_z, zgoal)
        else:
            print("Trying to reach goal..")
            reached_goal, steps = actor.move_to_goal(env, oracle=oracle)
            total_steps = total_steps + steps if reached_goal else 0
            if reached_goal:
                return True, total_steps
            elif len(plan) > 1:
                print("Failed to reach goal, trying to replan...")
                # remove edge from second-last node to goal, replan from second-last node
                full_graph_copy.remove_edge(plan[-2], zgoal)
                prev_node = plan[-2]
                plan = get_plan(full_graph_copy, plan[-2], zgoal)
            else:
                break
    print("plan failed\n")
    return False, 0

def plan_to_goal_and_execute_factorized(actor, env, onehot_graphs, oracle=False):
    model = actor.cpc_model
    # fully factorized planning
    plan_success = [False for _ in range(model.num_onehots)]
    total_steps = 0
    for idx, graph in enumerate(onehot_graphs):
        onehot_graph_copy = onehot_graphs[idx].copy()
        zstart = get_discrete_representation(model, env.get_obs().transpose(2, 0, 1), single=True)[idx]
        zgoal = get_discrete_representation(actor.cpc_model, env.goal_im.transpose(2, 0, 1), single=True)[idx]
        onehot_plan = get_plan(onehot_graph_copy, zstart, zgoal)
        if onehot_plan:
            prev_node = onehot_plan[0]
            node = onehot_plan[0]
            reached_node = True
        while onehot_plan:
            print("plan for onehot %d: " % idx, onehot_plan)
            for node in onehot_plan[1:]:
                reached_node, steps = actor.act(env, node, onehot_idx=idx, oracle=oracle)
                total_steps += steps
                if not reached_node:
                    print("failed to reach node", node)
                    break
                prev_node = node
                print("reached node", node)
            if not reached_node:
                print("Couldn't reach node %d" % node)
                print("Trying to replan....")
                # remove edge and replan
                onehot_graph_copy.remove_edge(prev_node, node)
                cur_z = get_discrete_representation(actor.cpc_model, env.get_obs().transpose(2, 0, 1), single=True)[idx]
                prev_node = cur_z
                onehot_plan = get_plan(onehot_graph_copy, cur_z, zgoal)
            else:
                plan_success[idx] = True
                break
    if np.sum(plan_success) == len(plan_success):
        reached_goal, steps = actor.move_to_goal(env, oracle=oracle)
        total_steps = total_steps+steps if reached_goal else 0
        return reached_goal, total_steps
    print("plan failed\n")
    return False, 0

def plan_to_goal_and_execute_grouped(actor, env, grouped_graphs, onehot_groups, oracle=False):
    model = actor.cpc_model
    plan_success = [False for _ in range(len(onehot_groups))]
    total_steps = 0
    for idx, graph in enumerate(grouped_graphs):
        grouped_graph_copy = grouped_graphs[idx].copy()
        zstart = tensor_to_label_grouped(get_discrete_representation(model, env.get_obs(), single=True),
                                         model.z_dim, onehot_groups)[idx]
        zgoal = tensor_to_label_grouped(get_discrete_representation(actor.cpc_model, env.goal_im, single=True),
                                        model.z_dim, onehot_groups)[idx]
        plan = get_plan(grouped_graph_copy, zstart, zgoal)
        if plan:
            prev_node = plan[0]
            node = plan[0]
            reached_node = True
        while plan:
            print("plan from %d to %d:" % (zstart, zgoal), plan)
            for node in plan[1:]:
                reached_node, steps = actor.act(env, node, onehot_idx=idx, groups=onehot_groups, oracle=oracle)
                total_steps += steps
                if not reached_node:
                    break
                prev_node = node
            if not reached_node:
                print("Couldn't reach node %d" % node)
                print("Trying to replan....")
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
        reached_goal, steps = actor.move_to_goal(env, oracle=oracle)
        total_steps = total_steps + steps if reached_goal else 0
        return reached_goal, total_steps
    print("plan failed\n")
    return False, 0

def get_planning_success_rate(actor,
                              env,
                              graph,
                              n_trials,
                              onehot_groups=None,
                              factorized=False,
                              oracle=False):
    if onehot_groups:
        success_rate = 0
        pass
    elif factorized:
        success_rate = 0
        pass
    else:
        successes = 0
        for trial in range(n_trials):
            print("Trial %d of %d" % (trial, n_trials))
            env.reset()
            success, total_steps = plan_to_goal_and_execute_full_graph(actor, env, graph, oracle=oracle)
            if success:
                print("success")
                print("total steps: ", total_steps)
                successes += 1
        success_rate = float(successes) / n_trials
        # success_rate = plan_and_execute_full_graph(actor, env, graph, n_trials, oracle)
    return success_rate