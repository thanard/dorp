from video_prediction.control.policy.cem_controllers.cem_base_controller import CEMBaseController
import imp
import numpy as np
from video_prediction.pred_util import get_context, rollout_predictions
from ..visualizer.construct_html import save_gifs, save_html, save_img, fill_template, img_entry_html
from ..visualizer.plot_helper import plot_score_hist
from collections import OrderedDict
from utils import from_numpy_to_var

def set_requires_grad(model, is_true):
    for param in model.parameters():
        param.requires_grad = is_true

class ClassifierController(CEMBaseController):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams, gpu_id, ngpu):
        """

        :param ag_params: agent parameters
        :param policyparams: policy parameters
        :param gpu_id: starting gpu id
        :param ngpu: number of gpus
        """
        CEMBaseController.__init__(self, ag_params, policyparams)

        params = imp.load_source('params', ag_params['conf_dir'] + '/conf.py')
        net_conf = params.configuration
        net_conf['batch_size'] = min(net_conf['batch_size'], self._hp.num_samples)
        if ngpu > 1:
            vpred_ngpu = ngpu - 1
        else: vpred_ngpu = ngpu

        self._predictor = net_conf['setup_predictor'](ag_params, net_conf, gpu_id, vpred_ngpu, self._logger)
        self._classifier = None # setup classifier here. one suggestion is to use 2 gpus, one for the video prediction and one for the classifier. thus ngpu = 2 and gpu 1 goes to classifier

        self._vpred_bsize = net_conf['batch_size']

        self._seqlen = net_conf['sequence_length']
        self._net_context = net_conf['context_frames']
        self._hp.T -= self._net_context
        self._hp.start_planning = self._net_context            # skip steps so there are enough context frames
        self._n_pred = self._seqlen - self._net_context
        assert self._n_pred > 0, "context_frames must be larger than sequence_length"

        self._img_height, self._img_width = net_conf['orig_size']

        self._images = None
        self._expert_images = None
        self._expert_score = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None

        # hard coded because these experiments only have one camera (and don't want to accidentally support multiple)
        self._n_cam = 1

        if self._hp.cost_type == "classifier":
            import json
            import torch
            from models import CPC, VAE
        # Use relative part here
        # Still hacky
            vpath = "out/local/remove_dec_w_mask_cont/var/vae-2-last-5"
            cpath = "out/local/cpc_on_remove_dec_w_mask_cont_vae/var/cpc-1-last-5"
            jpath = "out/local/e-cpc-z-dim-100-batch-size-64-N-1-vae-w-0.00-vae-b-10-k-1-pretrain-0-load-v-c-a/params.json"
            with open(jpath) as json_file:
                data = json.load(json_file)
            e_arch = data['e_arch']
            d_arch = data['d_arch']
            z_dim = data['z_dim']
            conditional = data['conditional']
            c_type = data['c_type']
            c_arch = data['c_arch']
            freeze_enc = data['freeze_enc']
            self.model = VAE(e_arch,
                        d_arch,
                        z_dim,
                        conditional)
            self.model.load_state_dict(torch.load(vpath))
            self.c_model = CPC(c_type, c_arch, e_arch, z_dim, self.model.encoder, conditional, freeze_enc)
            self.c_model.load_state_dict(torch.load(cpath))
            all_models = [self.c_model, self.model]
            if torch.cuda.is_available():
                for m in all_models:
                    m.cuda()
            for m in all_models:
                if m not in all_models:
                    set_requires_grad(m, False)
                    m.eval()
        elif self._hp.cost_type == "gt":
            img_size = (64, 64)
            self.mask_x = np.tile(np.linspace(-1, 1, img_size[1])[None], (img_size[0], 1))[None]
            self.mask_y = - np.tile(np.linspace(-1, 1, img_size[0])[:, None], (1, img_size[1]))[None]

    def get_block_only(self, o):
        return o[:, 1]

    def get_cm(self, o, mask):
        return np.sum((o * mask).reshape(o.shape[0], -1), axis=1, keepdims=True)/ np.sum(o.reshape(o.shape[0], -1), axis=1, keepdims=True)

    def reset(self):
        self._expert_score = None
        self._images = None
        self._expert_images = None
        self._goal_image = None
        self._start_image = None
        self._verbose_worker = None
        return super(ClassifierController, self).reset()

    def _default_hparams(self):
        default_dict = {
            'score_fn': 'dot_prod',
            'finalweight': 100,
            'nce_conf_path': '',
            'nce_restore_path': '',
            'nce_batch_size': 200,
            'state_append': None,
            'compare_to_expert': False,
            'verbose_img_height': 128,
            'verbose_frac_display': 0.,
            'cost_type': 'mse', # mse or classifier or gt
            'score_type': ''
        }
        parent_params = super(ClassifierController, self)._default_hparams()

        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def evaluate_rollouts(self, actions, cem_itr):
        context_actions = [self._actions_taken[-(self._net_context - 1 - i)][None][None] for i in range(self._net_context)]
        context_actions = np.repeat(np.concatenate(context_actions, axis=1), self._hp.num_samples, axis=0)
        padded_actions = np.concatenate((context_actions, actions), axis=1)

        last_frames, last_states = get_context(self._net_context, self._t,
                                               self._state, self._images, self._hp)

        gen_images = rollout_predictions(self._predictor,
                                         self._vpred_bsize,
                                         padded_actions,
                                         last_frames,
                                         last_states,
                                         logger=self._logger)[0]

        gen_images = np.concatenate(gen_images, 0)[:, :self._hp.T] * 255.                    # by default images are 0-1
        # (N_SAMPLES x T x 1 x H x W x C)

        """
        Use classifier to score gen_images given self._goal_image
        raw_scores = [N_SAMPLES, T]
        """
        num_imgs = self._hp.num_samples * self._hp.T
        nruns = max(1, num_imgs // self._vpred_bsize)
        _gen_imgs = gen_images[:, :, 0].reshape(self._hp.num_samples * self._hp.T, 64, 64, 3).transpose(0, 3, 1, 2)/255.
        _goal_img = self._goal_image
        bsize = self._vpred_bsize
        assert num_imgs % bsize == 0
        if self._hp.cost_type == 'classifier':
            raw_scores = []
            for run in range(nruns):
                c = self._contexts.repeat(bsize, 1, 1, 1)
                o_pred = self.model(from_numpy_to_var(_gen_imgs[bsize*run: bsize*(run+1)]), c)[0]
                goal_pred = self.model(from_numpy_to_var(np.tile(_goal_img, (bsize, 1, 1, 1))), c)[0]
                rs = -self.c_model(
                    o_pred,
                    goal_pred,
                    c
                ).detach().cpu().numpy()
                raw_scores.append(rs)
            raw_scores = np.concatenate(raw_scores, 0).reshape(self._hp.num_samples, self._hp.T)
        elif self._hp.cost_type == 'mse':
            raw_scores = np.sqrt(((_gen_imgs - np.tile(_goal_img, (num_imgs, 1, 1, 1)))**2).mean(axis=(1, 2, 3))).reshape(self._hp.num_samples, self._hp.T)
        elif self._hp.cost_type == "gt":
            c = self._contexts.cpu().detach().numpy()
            o_x = self.get_cm(self.get_block_only(_gen_imgs - c), self.mask_x)
            o_y = self.get_cm(self.get_block_only(_gen_imgs - c), self.mask_y)

            goal_x = self.get_cm(self.get_block_only(_goal_img - c), self.mask_x)
            goal_y = self.get_cm(self.get_block_only(_goal_img - c), self.mask_y)
            raw_scores = np.sqrt(np.sum((o_x - goal_x)**2, 1) + np.sum((o_y - goal_y)**2, 1)).reshape(self._hp.num_samples, self._hp.T)
        # Compute final score
        scores = self._weight_scores(raw_scores)

        if self._verbose_condition(cem_itr):
            verbose_folder = "planning_{}_itr_{}".format(self._t, cem_itr)
            content_dict = OrderedDict()
            # visualize_indices = scores.argsort()[:max(10, int(actions.shape[0] * self._hp.verbose_frac_display))]
            ordered_scores = scores.argsort()
            visualize_indices = [ordered_scores[0], ordered_scores[1], ordered_scores[-2], ordered_scores[-1]]

            # start image and predictions (alternate by camera)
            for c in range(self._n_cam):
                name = 'cam_{}_start'.format(c)
                save_path = save_img(self._verbose_worker, verbose_folder, name, (self._images[-1, c] * 255).astype(np.uint8))
                content_dict[name] = [save_path for _ in visualize_indices]

                name = 'cam_{}_goal'.format(c)
                goal_image = np.transpose((self._goal_image[0] * 255).astype(np.uint8), (1, 2, 0))
                save_path = save_img(self._verbose_worker, verbose_folder, name, goal_image)
                content_dict[name] = [save_path for _ in visualize_indices]

                name = 'cam_{}_final'.format(c)
                content_dict[name] = []
                for g_i in visualize_indices:
                    final_image = (gen_images[g_i, -1, c]).astype(np.uint8)
                    save_path = save_img(self._verbose_worker, verbose_folder, name + '_%d' % g_i, final_image)
                    content_dict[name].append(save_path)

                verbose_images = [gen_images[g_i, :, c].astype(np.uint8) for g_i in visualize_indices]
                row_name = 'cam_{}_pred_images'.format(c)
                content_dict[row_name] = save_gifs(self._verbose_worker, verbose_folder,
                                                   row_name, verbose_images)

            # scores
            content_dict['scores'] = scores[visualize_indices]
            if self._hp.cost_type == "gt":
                content_dict['o_x'] = o_x.squeeze()[visualize_indices]
                content_dict['o_y'] = o_y.squeeze()[visualize_indices]
                content_dict['goal_x'] = [goal_x.squeeze() for v in visualize_indices]
                content_dict['goal_y'] = [goal_y.squeeze() for v in visualize_indices]


            hist = plot_score_hist(scores, tick_value=self._expert_score)
            hist_path = save_img(self._verbose_worker, verbose_folder, "score_histogram", hist)
            extra_entry = img_entry_html(hist_path, height=hist.shape[0], caption="score histogram")

            html_page = fill_template(cem_itr, self._t, content_dict, img_height=self._hp.verbose_img_height,
                                      extra_html=extra_entry)
            save_html(self._verbose_worker, "{}/plan.html".format(verbose_folder), html_page)

        return scores

    def _weight_scores(self, raw_scores):
        if self._hp.score_type == 'exp-neg' and self._hp.cost_type == 'classifier':
            return self._log_sum_exp(raw_scores.copy())
        if self._hp.finalweight >= 0:
            scores = raw_scores.copy()
            scores[:, -1] *= self._hp.finalweight
            scores = np.sum(scores, axis=1) / sum([1. for _ in range(self._n_pred - 1)] + [self._hp.finalweight])
        else:
            scores = raw_scores[:, -1].copy()
        return scores
        # return raw_scores.copy()[:, 2]

    def _eval_embedding_cost(self, goal_embed, input_embed):
        if self._hp.score_fn == 'dot_prod':
            # - log prob ignoring constant term (denominator)
            return -np.matmul(goal_embed[None], np.swapaxes(input_embed, 2, 1))[:, 0]
        raise NotImplementedError

    def _log_sum_exp(self, arr):
        max_arr = np.max(arr, axis=1)
        return max_arr + np.log(np.sum(np.exp(arr - max_arr[:,None]), axis=1))

    def act(self, t=None, goal_image=None, images=None, state=None, contexts=None, verbose_worker=None):
        """
        t: current timestep (starting from 0)
        goal_image: goal image to be fed to classifier (assert correct dtype, etc. yourself)
        images: [T_past,ncam,im_height,im_width,3] RGB tensor of images (should be float32 in range 0-1)
        state: [T, sdim] tensor of state
        verbose_worker: pointer to queue for asynchronous file saver (for fast gif saving)
        """
        self._start_image = images[-1].astype(np.float32)
        self._goal_image = goal_image
        self._images = images
        self._verbose_worker = verbose_worker
        self._contexts = contexts

        if self._hp.compare_to_expert:
            self._expert_images = goal_image[1:self._n_pred + 1] * 255

        return super(ClassifierController, self).act(t, state)
