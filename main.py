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
from pathlib import Path
from transition import Transition

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='gridworld')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--grid_n', type=int, default=16)
parser.add_argument('--step_size', type=int, default=1)

# Model params:
parser.add_argument('--n_epochs', type=int, default=1000)
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
parser.add_argument('--baseline', type=str, default='', help="htm")
parser.add_argument('--reset_rate', type=int, default=1)
parser.add_argument('--n_traj', type=int, default=30000)
parser.add_argument('--len_traj', type=int, default=2)

parser.add_argument('--dataset_path', type=str, default="") # specify path if using offline dataset

parser.add_argument('--random_step_size', action="store_true")
parser.add_argument('--switching_factor_freq', type=int, default=1)
# Planning
parser.add_argument('--enable_factor', action="store_true")
parser.add_argument('--enable_full', action="store_true")

args = parser.parse_args()

env = get_env(args.env, args)
torch.manual_seed(args.seed)
savepath = Path(setup_savepath(vars(args)))
savepath.mkdir(parents=True, exist_ok=True)

print("savepath ", savepath)
save_python_cmd(str(savepath), vars(args), "main.py")

writer = SummaryWriter(savepath)
if args.num_onehots == 0:
    args.num_onehots = args.n_agents

mode = 'continuous' if args.baseline == 'htm' else 'double_encoder'
cpc_params = {
              'encoder': args.encoder,
              'z_dim': args.z_dim,
              'batch_size': args.batch_size,
              'in_channels': 3, # inputs are rgb
              'mode': mode,
              'input_dim': env.grid_n,
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
params_path = savepath / "model-hparams.json"
with params_path.open('w') as fp:
    json.dump(all_params, fp, indent=2, sort_keys=True)

model = CPC(**cpc_params)
model_path = savepath / 'cpc-model'
if model_path.exists():
    print("### Model Loaded ###")
    model.load_state_dict(torch.load(model_path))
else:
    print("### Initialize Model ###")
    model.apply(init_weights_func(args.scale_weights))

n_epochs = args.n_epochs
print(model)

actor = CEM_actor()
actor.load_cpc_model(args.n_agents, model=model)

graph = Transition(args.enable_factor, args.enable_full, args.z_dim, args.num_onehots)

dataset = {'train': {'act': None, 'obs': None}, 'valid': {'act': None, 'obs': None}}
datapath = {'train': {'obs': Path(args.dataset_path, 'train_observations.npy'),
                      'act': Path(args.dataset_path, 'train_actions.npy')},
            'valid': {'obs': Path(args.dataset_path, 'valid_observations.npy'),
                      'act': Path(args.dataset_path, 'valid_actions.npy')}}
if args.dataset_path:
    # precollect dataset of transitions:
    for type in dataset.keys():
        for key, path in datapath[type].items():
            if path.exists():
                print("### Data %s %s Loaded ###" % (type, key))
                dataset[type][key] = np.load(path)
                if dataset[type][key].dtype == np.uint16:
                    print("Updating the types")
                    dataset[type][key] = dataset[type][key].astype(np.int64)
                    np.save(path, dataset[type][key])
            else:
                print("### Data %s %s Not Exist ###" % (type, key))

train(env,
      actor,
      n_epochs,
      savepath,
      step_size=args.step_size,
      vis_freq=args.vis_freq,
      plan_freq=args.plan_freq,
      writer=writer,
      loss_form=args.loss,
      lr=args.lr,
      ce_temp=args.ce_temp,
      baseline='',
      n_traj=args.n_traj,
      len_traj=args.len_traj,
      dataset=dataset,
      datapath=datapath,
      graph=graph,
      kwargs={"random_step_size": args.random_step_size,
              "switching_factor_freq": args.switching_factor_freq})
