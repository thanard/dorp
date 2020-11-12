import argparse
import json
import matplotlib
matplotlib.use('agg')
from utils.gen_utils import *
from model import CPC, init_weights_func
from agent.visual_foresight import CEM_actor
from train import train
from planning import *
from torch.utils.tensorboard import SummaryWriter

def plan_to_goal_and_execute(actor, env, factorized=False, onehot_groups=None):
    model = actor.cpc_model
    if factorized:
        if onehot_groups:
            grouped_graphs = create_graph_from_sample_transitions_grouped_onehots(model, env, onehot_groups)
            plan_success = [False for _ in range(len(onehot_groups))]

            for idx, graph in enumerate(grouped_graphs):
                grouped_graph_copy = grouped_graphs[idx].copy()
                plan = get_plan(model, grouped_graph_copy, env)
                prev_node = plan[0]
                node = plan[0]
                reached_node = True
                while plan:
                    for node in plan[1:]:
                        reached_node = actor.act(env, node, onehot_idx=idx, groups=onehot_groups)
                        if not reached_node:
                            break
                    if not reached_node:
                        # remove edge and replan
                        grouped_graph_copy.remove((prev_node, node))
                        plan = get_plan(model, grouped_graph_copy, env)
                    else:
                        plan_success[idx] = True
            if np.sum(plan_success) == len(plan_success):
                return actor.move_to_goal(env)
            return False
        else:
            onehot_graphs = create_graph_from_sample_transitions_factorized(model, env)
            # fully factorized planning
            plan_success = [False for _ in range(len(model.num_onehots))]
            for idx, graph in enumerate(onehot_graphs):
                onehot_graph_copy = onehot_graphs[idx].copy()
                onehot_plan = get_plan(model, onehot_graph_copy, env)
                prev_node = onehot_plan[0]
                node = onehot_plan[0]
                reached_node = True
                while onehot_plan:
                    for node in onehot_plan[1:]:
                        reached_node = actor.act(env, node, onehot_idx=idx)
                        if not reached_node:
                            break
                    if not reached_node:
                        # remove edge and replan
                        onehot_graph_copy.remove((prev_node, node))
                        onehot_plan = get_plan(model, onehot_graph_copy, env)
                    else:
                        plan_success[idx] = True
            if np.sum(plan_success) == len(plan_success):
                return actor.move_to_goal(env)
            return False

    else:
        full_graph = create_graph_from_sample_transitions_full(model, env)
        full_graph_copy = full_graph.copy()
        plan = get_plan(model, full_graph_copy, env)
        prev_node = plan[0]
        node = plan[0]
        reached_node = True
        while plan:
            for node in plan[1:]:
                reached_node = actor.act(env, node)
                if not reached_node:
                    break
                prev_node = node
            if not reached_node:
                # remove edge and replan
                full_graph_copy.remove((prev_node, node))
                plan = get_plan(model, full_graph_copy, env)
            else:
                return actor.move_to_goal(env)
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='gridworld')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--grid_n', type=int, default=16)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--model_dir', type=str, default="")

# Model params:
parser.add_argument('--n_epochs', type=int, default=800)
parser.add_argument('--encoder', type=str, default='cswm')
parser.add_argument('--n_agents', type=int, default=1)
parser.add_argument('--z_dim', type=int, default=32)
parser.add_argument('--W', type=int, default=3)
parser.add_argument('--temp', type=float, default=.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_filters', type=int, default=32)
parser.add_argument('--num_onehots', type=int, default=0, help="if set to 0, model will use num_onehots=n_agents")
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--separate_W', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--loss', type=str, default='ce') # either ce or hinge for different contrastive loss forms
parser.add_argument('--normalization', type=str, default="batchnorm")
parser.add_argument('--ce_temp', type=float, default=1.)
parser.add_argument('--scale_weights', type=float, default=1.)

parser.add_argument('--savepath', type=str, default='results/grid')
parser.add_argument('--vis_freq', type=int, default=5) # visualize every n epochs
parser.add_argument('--plan_freq', type=int, default=50) # plan evaluation every n epochs
parser.add_argument('--h_distance', type=int, default=8)
parser.add_argument('--baseline', type=str, default='', help="htm")
parser.add_argument('--reset_rate', type=int, default=1)
parser.add_argument('--n_traj', type=int, default=20000)
parser.add_argument('--len_traj', type=int, default=2)

args = parser.parse_args()

env = get_env(args.env, args)
torch.manual_seed(args.seed)
savepath = setup_savepath(vars(args))
if not os.path.exists(savepath):
    os.makedirs(savepath)

print("savepath ", savepath)
save_python_cmd(savepath, vars(args), "main.py")

writer = SummaryWriter(savepath)
if args.num_onehots == 0:
    args.num_onehots = args.n_agents

mode = 'continuous' if args.baseline == 'htm' else 'double_encoder'
cpc_params = {
              'encoder': args.encoder,
              'z_dim': args.z_dim,
              'batch_size': args.batch_size,
              'in_channels': args.n_agents,
              'mode': mode,
              'input_dim': args.grid_n,
              'W_form': args.W,
              'num_filters': args.num_filters,
              'num_onehots': args.num_onehots,
              'temp': args.temp,
              'separate_W': args.separate_W,
              'num_layers': args.num_layers,
              'normalization': args.normalization,
              }

all_params = cpc_params.copy()
all_params['n_epochs'] = args.n_epochs
all_params['grid_width'] = args.grid_n
params_path = os.path.join(savepath, "%s-params.json" % args.env)
with open(params_path, 'w') as fp:
    json.dump(all_params, fp, indent=2, sort_keys=True)

model = CPC(**cpc_params)
model_path = os.path.join(args.model_dir, "%s-params.json" % args.env)
if args.model_dir and os.path.exists(model_path):
    print("### Model Loaded ###")
    model.load_state_dict(torch.load(model_path))
else:
    model.apply(init_weights_func(args.scale_weights))

n_epochs = args.n_epochs
print(model)

actor = CEM_actor()
actor.load_cpc_model(model=model)

train(env,
      actor,
      n_epochs,
      savepath,
      vis_freq=args.vis_freq,
      plan_freq=args.plan_freq,
      writer=writer,
      loss_form='ce',
      lr=1e-3,
      ce_temp=1.0,
      baseline='',
      n_traj=args.n_traj,
      len_traj=args.len_traj)
