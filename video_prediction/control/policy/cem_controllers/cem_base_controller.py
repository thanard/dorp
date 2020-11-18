import numpy as np
from video_prediction.logger import Logger
from video_prediction.control.policy.cem_controllers.samplers.cem_sampler import CEMSampler
from video_prediction.control.policy.policy import Policy


class CEMBaseController(Policy):
    """
    Cross Entropy Method Stochastic Optimizer
    """
    def __init__(self, ag_params, policyparams):
        """
        :param ag_params:
        :param policyparams:
        """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.agentparams = ag_params

        if self._hp.logging_dir:
            self._logger = Logger(self._hp.logging_dir, 'cem{}log.txt'.format(self.agentparams['gpu_id']))
        else:
            self._logger = Logger(printout=True)

        self._logger.log('init CEM controller')

        self._t_since_replan = None
        self._t = None
        self._n_iter = self._hp.iterations

        #action dimensions:
        self._adim = self.agentparams['adim']
        self._sdim = self.agentparams['sdim']                             # state dimension

        self._sampler = None
        self._best_indices, self._best_actions = None, None

        self._state = None
        self._actions_taken = []

    def _default_hparams(self):
        default_dict = {
            'verbose': True,
            'verbose_every_iter': False,
            'logging_dir': '',
            'replan_interval': 0,
            'sampler': CEMSampler,
            'T': 15,                       # planning horizon
            'iterations': 3,
            'num_samples': 200,
            'selection_frac': 0., # specifcy which fraction of best samples to use to compute mean and var for next CEM iteration
            'start_planning': 0,
            'default_k': 10
        }

        parent_params = super(CEMBaseController, self)._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def _override_defaults(self, policyparams):
        for name, value in policyparams.get('sampler', CEMSampler).get_default_hparams().items():
            if name in self._hp:
                print('Warning default value for {} already set!'.format(name))
                self._hp.set_hparam(name, value)
            else:
                self._hp.add_hparam(name, value)

        return super(CEMBaseController, self)._override_defaults(policyparams)

    def reset(self):
        self._actions_taken = []
        self._best_indices = None
        self._best_actions = None
        self._t_since_replan = None

        self._sampler = self._hp.sampler(self._hp, self._adim, self._sdim)
        self.plan_stat = {} #planning statistics

    def perform_CEM(self, state):
        self._logger.log('starting cem at t{}...'.format(self._t))
        self._logger.log('------------------------------------------------')

        K = self._hp.default_k
        if self._hp.selection_frac:
            K = max(int(self._hp.selection_frac * self._hp.num_samples), self._hp.default_k)
        actions = self._sampler.sample_initial_actions(self._t, self._hp.num_samples, state[-1])
        for itr in range(self._n_iter):
            self._logger.log('------------')
            self._logger.log('iteration: ', itr)

            scores = self.evaluate_rollouts(actions, itr)
            assert scores.shape == (actions.shape[0],), "score shape should be (n_actions,)"

            self._best_indices = scores.argsort()[:K]
            self._best_actions = actions[self._best_indices]

            self.plan_stat['scores_itr{}'.format(itr)] = scores
            if itr < self._n_iter - 1:
                actions = self._sampler.sample_next_actions(self._hp.num_samples, self._best_actions)

        self._t_since_replan = 0

    def evaluate_rollouts(self, actions, cem_itr):
        raise NotImplementedError

    def _verbose_condition(self, cem_itr):
        if self._hp.verbose:
            if self._hp.verbose_every_iter or cem_itr == self._n_iter - 1:
                return True
        return False

    def act(self, t=None, state=None):
        """
        Return a random action for a state.
        Args:
            t: the current controller's Time step
        """
        self._state = state
        self._t = t

        if t < self._hp.start_planning:
            action = np.zeros(self.agentparams['adim'])
        else:
            if self._hp.replan_interval:
                if self._t_since_replan is None or self._t_since_replan + 1 >= self._hp.replan_interval:
                    self.perform_CEM(state)
                else:
                    self._t_since_replan += 1
            else:
                self.perform_CEM(state)
            action = self._best_actions[0, self._t_since_replan]

        assert action.shape == (self.agentparams['adim'],), "action shape does not match adim!"

        self._logger.log('time {}, action - {}'.format(t, action))

        if self._best_actions is not None:
            action_plan_slice = self._best_actions[:, min(self._t_since_replan + 1, self._hp.T - 1):]
            self._sampler.log_best_action(action, action_plan_slice)
        else:
            self._sampler.log_best_action(action, None)

        self._actions_taken.append(action)
        return action
        # return {'actions':action, 'plan_stat':self.plan_stat}