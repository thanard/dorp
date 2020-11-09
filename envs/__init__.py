import abc


class GoalEnv(object):
    """Goal-based Environment"""
    def __init__(self):
        self.current = None
        self.goal = None

    @abc.abstractmethod
    def reset(self):
        """Resets the state of the environment and the goal, returning an initial observation (images)."""
        pass

    @abc.abstractmethod
    def seed(self, seed):
        """Set the random seed."""
        pass

    @abc.abstractmethod
    def render(self):
        """Return current image observation."""
        pass

    @abc.abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics.

        :param action:
        :return:
        """
        pass
