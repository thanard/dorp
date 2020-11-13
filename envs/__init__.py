import abc


class GoalEnv(object):
    """Goal-based Environment"""
    def __init__(self):
        self.state = None
        self.goal_im = None

    @abc.abstractmethod
    def reset(self):
        """Resets the state of the environment and the goal, returning an initial observation (images)."""
        pass

    @abc.abstractmethod
    def seed(self, seed):
        """Set the random seed."""
        pass

    @abc.abstractmethod
    def obs(self):
        """Return current image observation."""
        pass

    @abc.abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics.

        :param action:
        :return:
        """
        pass

    @abc.abstractmethod
    def sample_action(self):
        """Return a uniformly sampled action from the action space"""
        pass

    @abc.abstractmethod
    def reached_goal(self):
        """Return True if the state of the environment matches the goal state"""
        pass
