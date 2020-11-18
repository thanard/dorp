import argparse
import json
from planning import *
from agent.visual_foresight import CEM_actor
from model import *
from envs import *
from utils.gen_utils import *
from utils.dataset import *

ONEHOT_GROUPS = (3,2)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)  # need to specify model path
parser.add_argument('--n_agents', type=int, default=1)
parser.add_argument('--n_trials', type=int, default=50)
parser.add_argument('--grid_n', type=int, default=16)
parser.add_argument('--env', type=str, default='gridworld')
parser.add_argument('--vp_modeldir', type=str, default='')
parser.add_argument('--n_traj', type=int, default=1000)
parser.add_argument('--len_traj', type=int, default=7)
parser.add_argument('--action_repeat', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('-group_onehots', action='store_true')
parser.add_argument('-gt', action='store_true')
parser.add_argument('-factorized', action='store_true')

# params for graph construction
parser.add_argument('--n_traj_for_graph', type=int, default=50)
parser.add_argument('--len_traj_for_graph', type=int, default=1200)
parser.add_argument('--dataset_path', type=str, default='')

args = parser.parse_args()

np.random.seed(args.seed)

actor = CEM_actor()
actor.load_cpc_model(args.n_agents, cpc_modeldir=args.model_dir)
actor.load_vp_model(args.vp_modeldir, args.action_repeat * args.len_traj, args.n_traj)

env = get_env(args.env, args)

onehot_groups = ONEHOT_GROUPS if args.group_onehots else None

dataset = None
if args.dataset_path:
    dataset = np.load(args.dataset_path)

rate = get_planning_success_rate(actor,
                                 env,
                                 args.n_trials,
                                 factorized=args.factorized,
                                 onehot_groups=onehot_groups,
                                 oracle=args.gt,
                                 n_traj=args.n_traj_for_graph,
                                 len_traj=args.len_traj_for_graph,
                                 dataset=dataset)

print("Success rate for %d trials: " % args.n_trials, rate)