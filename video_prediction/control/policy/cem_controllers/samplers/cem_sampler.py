import  numpy as np
"""
Implements a custom sampler for the CEM controller
"""

class CEMSampler(object):
    def __init__(self, hp, adim, sdim, **kwargs):
        self._hp = hp
        self._adim, self._sdim = adim, sdim
        self._chosen_actions = []
        self._best_action_plans = []

    def sample_initial_actions(self, t, nsamples, current_state):
        """
        Samples initial actions for CEM iterations
        :param nsamples: number of samples
        :param current_state: Current state of robot
        :return: action samples in (B, T, adim) format
        """
        # TODO Hack for complex block domain
        return np.random.uniform(-1, 1, (nsamples, self._hp.T, 2))
        # T = self._hp.T
        # x = np.random.randn(nsamples, T, 2) * 0.05
        # if self._adim == 2:
        #     return x
        # elif self._adim == 9:
        #     a = np.zeros((nsamples, T, self._adim))
        #     a[:, :, 3:5] = x
        #     return a

    def sample_next_actions(self, n_samples, best_actions):
        """
        Samples actions for CEM iterations, given BEST last actions
        :param nsamples: number of samples
        :param best_actions: number of samples (K, T, adim)
        :return: action samples in (B, T, adim) format
        """
        mu = best_actions.mean(0).reshape(1, self._hp.T, self._adim)
        std = best_actions.std(0).reshape(1, self._hp.T, self._adim)
        return np.random.randn(n_samples, self._hp.T, self._adim)*std + mu

    def log_best_action(self, action, best_action_plans):
        """
        Some sampling distributions may change given the taken action

        :param action: action executed
        :param best_action_plans: batch of next planned actions (after this timestep) ordered in ascending cost
        :return: None
        """
        self._chosen_actions.append(action.copy())
        self._best_action_plans.append(best_action_plans)

    @staticmethod
    def get_default_hparams():
        hparams_dict = {
        }
        return hparams_dict
