import abc


class Agent(object):
    def reset(self):
        """If episodic memory is required."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Set training or eval mode."""
        pass

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""
        pass

    @abc.abstractmethod
    def act(self, obs):
        """Infer action given an observation."""
        pass
