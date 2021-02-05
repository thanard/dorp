from agent import Agent
from train import train
import numpy as np
from model import get_discrete_representation
from utils.gen_utils import *
from video_prediction.setup_predictor import setup_predictor
from video_prediction.vpred_model_interface import VPred_Model_Interface
from model import CPC
import json

class CEM_actor(Agent):
    def __init__(self):
        super(CEM_actor, self).__init__()

    def load_vp_model(self, vp_modeldir, sequence_length=14, num_action_sequences=1000):
        self.setup_vp(vp_modeldir, sequence_length, num_action_sequences)


    def load_cpc_model(self, n_agents, cpc_modeldir='', model=None):
        if model:
            self.cpc_model = model
        else:
            with open(os.path.join(cpc_modeldir, "model-hparams.json"), 'r') as f:
                cpc_params = json.load(f)

            grid_n = cpc_params.pop('grid_width')
            n_epochs = cpc_params.pop('n_epochs')
            model = CPC(**cpc_params).cuda()
            print("loading model.....")
            modelpath = os.path.join(cpc_modeldir, "%d-agents-model" % n_agents)
            model.load_state_dict(torch.load(modelpath))
            model.eval()
            self.cpc_model = model

    def setup_vp(self, result_folder, sequence_length, num_action_sequences):
        '''
        sets up video prediction model saved in results_folder
        '''
        with open(os.path.join(result_folder, 'dataset_hparams.json')) as f:
            dataset_hparams = json.load(f)
        with open(os.path.join(result_folder, 'model_hparams.json')) as f:
            model_hparams = json.load(f)
        assert dataset_hparams['env'] == 'gridworld'

        configuration = {
            'json_dir': result_folder,
            'pred_model': VPred_Model_Interface,
            'pretrained_model': result_folder,  # 'filepath of a pretrained model to resume training from.' ,
            'sequence_length': sequence_length,  # 'sequence length to load, including context frames.' ,
            'context_frames': model_hparams['context_frames'],  # of frames before predictions.' ,
            'batch_size': num_action_sequences,
            'sdim': dataset_hparams['sdim'],
            'adim': dataset_hparams['adim'],
            'orig_size': dataset_hparams['input_dims'],
            'ncam': 1,
            'no_pix_distrib': ''
        }

        predictor = setup_predictor({}, configuration, gpu_id=0, ngpu=1)
        self.vp_model = predictor
        self.vp_dataset_hparams = dataset_hparams

    def predict_sequence(self, first_image, actions):
        scale_fac = [self.vp_dataset_hparams['input_dims'][0] // self.vp_dataset_hparams['grid_n'],
                     self.vp_dataset_hparams['input_dims'][1] // self.vp_dataset_hparams['grid_n']]
        # scale up to 32x32x3
        first_image = np.repeat(np.repeat(first_image,
                                          scale_fac[0], axis=1),
                                scale_fac[1], axis=2)
        first_image = first_image[None, None]  # [1, 1, 1, 32, 32, 32]
        states = np.zeros((1, 1, self.vp_dataset_hparams['sdim']))

        gen_images, _, gen_state = self.vp_model(input_images=first_image,
                                             input_state=states,
                                             input_actions=actions)
        gen_images = gen_images[:, :, 0]
        # Scale down to 16x16x3:
        gen_images = gen_images[:, :, ::2, ::2, :]
        return gen_images

    def reset(self):
        """If episodic memory is required."""
        pass

    def train(self, training=True):
        """Set training or eval mode."""
        pass

    def update(self, buffer, logger, step, kwargs):
        # """Main function of the agent that performs learning."""
        pass

    def sample_action_sequences(self, env, n_traj, len_traj, action_repeat):
        action_seqs = []
        for traj in range(n_traj):
            traj_actions = []
            i = 0
            while i < len_traj:
                action = env.sample_action()
                for _ in range(action_repeat):
                    traj_actions.append(action)
                i+=1
            action_seqs.append(np.array(traj_actions))
        return np.array(action_seqs)

    def move_to_goal(self,
                     env,
                     n_traj=1000,
                     len_traj=14,
                     action_repeat=1,
                     oracle=False,
                     max_attempts=3):
        steps = 0
        attempt = 0
        while attempt <  max_attempts:
            print("Attempting to reached goal ", attempt + 1)
            print("cur state before goal", env.get_state())
            goal_im = env.goal_im / 255
            goal_im = np.tile(goal_im, (n_traj * (len_traj * action_repeat), 1, 1, 1))
            cur_im = env.get_obs()  # 16x16xn
            action_seqs = self.sample_action_sequences(env, n_traj, len_traj, action_repeat)
            if oracle:
                pred_seqs = self.step_sequence_batch(env, action_seqs) / 255
            else:
                pred_seqs = self.predict_sequence(cur_im[None]/255, action_seqs)
            pred_seqs = pred_seqs[:, :len_traj * action_repeat]
            all_pred_ims = pred_seqs.reshape(n_traj * (len_traj * action_repeat), -1)
            goal_im = goal_im.reshape(n_traj * (len_traj * action_repeat), -1)
            im_norms = np.sum((goal_im - all_pred_ims) ** 2, axis=1)
            min_traj_idx = np.argmin(im_norms) // (len_traj*action_repeat)
            min_step_idx = np.argmin(im_norms) % (len_traj*action_repeat)
            best_action_seq = action_seqs[min_traj_idx, :min_step_idx + 1]
            env.step_sequence(best_action_seq)
            steps += len(best_action_seq)
            if env.reached_goal():
                return True, steps
            attempt+=1
        print("didn't reach goal (planned goal pos, true goal pos)", env.get_state(), env.goal_state)
        return False, 0

    def act(self,
            env,
            target_node,
            n_traj=1000,
            len_traj=7,
            action_repeat=2,
            onehot_idx=-1,
            groups=(),
            oracle=False):
        steps = 0
        obs = env.get_obs()
        obs_torch = single_im_to_torch(obs).permute(0, 3, 1, 2)
        node_label_start = self.cpc_model.encode(obs_torch, vis=True).squeeze(0).cpu().numpy()
        action_seqs = self.sample_action_sequences(env, n_traj, len_traj, action_repeat)

        if oracle:
            pred_seqs = self.step_sequence_batch(env, action_seqs)
        else:
            pred_seqs = self.predict_sequence(obs[None]/255, action_seqs)*255

        pred_seqs = pred_seqs[:, :len_traj * action_repeat]
        all_pred_ims = pred_seqs.reshape(n_traj * (len_traj * action_repeat), env.grid_n, env.grid_n, 3).transpose(0, 3, 1, 2)
        pred_zs = get_discrete_representation(self.cpc_model, all_pred_ims)
        filters = []

        if groups:
            # grouped onehots (semi-factorized planning)
            assert onehot_idx != -1
            node_label_start = tensor_to_label_grouped(node_label_start, self.cpc_model.z_dim, groups)
            pred_zs = tensor_to_label_grouped(pred_zs, self.cpc_model.z_dim, groups).reshape(n_traj, len_traj * action_repeat, -1)
            for idx in range(len(groups)):
                if idx != onehot_idx:
                    filters.append(pred_zs[:, -1, idx] == node_label_start[idx])
            filters.append(pred_zs[:, -1, onehot_idx] == target_node)
            filters = np.logical_and.reduce(filters)
        elif onehot_idx != -1:
            # fully factorized planning
            pred_zs = pred_zs.reshape(n_traj, len_traj * action_repeat, -1)
            for idx in range(self.cpc_model.num_onehots):
                if idx != onehot_idx:
                    filters.append(pred_zs[:, -1, idx] == node_label_start[idx])
            filters.append(pred_zs[:, -1, onehot_idx] == target_node)
            filters = np.logical_and.reduce(filters)
        else:
            # non factorized planning (full graph)
            pred_zs = tensor_to_label_array(pred_zs, self.cpc_model.num_onehots, self.cpc_model.z_dim)
            pred_zs = pred_zs.reshape(n_traj, len_traj * action_repeat, -1)
            filters = (pred_zs[:, -1, 0] == target_node)

        possible_seqs = pred_zs[filters][:, :, onehot_idx]  # check for action sequences ending with the node
        if possible_seqs.any():
            # choose best sequence based on max number of desired node within sequence
            possible_action_seqs = action_seqs[filters]
            best_seq_idx = np.argmax(np.sum(possible_seqs == target_node, axis=1))
            best_action_seq = possible_action_seqs[best_seq_idx]

            env.step_sequence(best_action_seq) # step through action sequence
            print("env state", env.get_state())
            new_obs = single_im_to_torch(env.get_obs()).permute(0, 3, 1, 2)
            new_node = self.cpc_model.encode(new_obs, vis=True).squeeze(0).cpu().numpy()
            steps += len(best_action_seq)
            # new_node = new_node[onehot_idx]
            if groups and tensor_to_label_grouped(new_node, self.cpc_model.z_dim, groups)[onehot_idx] != target_node:
                print("new node does not match: (new node, true node)",
                      tensor_to_label_grouped(new_node, self.cpc_model.z_dim, groups)[onehot_idx],
                      target_node)
                print("cur state:", env.get_state())
            elif onehot_idx != -1 and new_node[onehot_idx] != target_node:
                print("new node does not match: (new node, true node)", new_node[onehot_idx], target_node)
                print("cur state:", env.get_state())
            elif onehot_idx == -1 and tensor_to_label(new_node, self.cpc_model.num_onehots, self.cpc_model.z_dim) != target_node:
                print("new node does not match: (new node, true node)", new_node, target_node)
                print("cur state:", env.get_state())
                # save_image(from_numpy_to_var(pred_seqs[filters][best_seq_idx]).permute(0, 3, 1, 2),
                #            "results/grid/plan_test/pred_seq.png", padding=2, pad_value=10)
                # save_image(model.encoder.to_rgb(cur_im), "results/grid/plan_test/real_im.png", padding=2, pad_value=10)
        else:
            return False, steps
        return True, steps

    def step_sequence_batch(self, env, action_seqs):
        '''
        Only used for GT dynamics model, not used for final evaluation
        :return:
        '''
        cur_state = env.get_state()
        new_poses = []
        for actions in action_seqs:
            env.reset(cur_state)
            traj_ims = []
            for action in actions:
                env.step(action)
                new_pos_im = env.get_obs()
                traj_ims.append(new_pos_im)
            new_poses.append(np.array(traj_ims))
        new_poses = np.stack(new_poses)
        env.reset(cur_state)
        return new_poses


