import abc


class Agent(object):
    def __init__(self):
        self.cpc_model = None
        self.vp_model = None

    @abc.abstractmethod
    def reset(self):
        """If episodic memory is required."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Set training or eval mode."""
        pass

    @abc.abstractmethod
    def update(self, buffer, logger, step, kwargs):
        """Main function of the agent that performs learning."""
        pass

    @abc.abstractmethod
    def act(self,
            env,
            target_node,
            n_traj=1000,
            len_traj=6,
            action_repeat=2,
            onehot_idx=-1,
            gt=False):
        """Infer action given an observation."""
        pass
