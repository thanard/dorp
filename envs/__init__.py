import abc


class GoalEnv(object):
    """Goal-based Environment"""
    def __init__(self):
        self.goal_im = None
        self.name = None
        self.n_agents = 1

    @abc.abstractmethod
    def get_state(self):
        """Resets the state of the environment and the goal, returning an initial observation (images)."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Resets the state of the environment and the goal, returning an initial observation (images)."""
        pass

    @abc.abstractmethod
    def seed(self, seed):
        """Set the random seed."""
        pass

    @abc.abstractmethod
    def get_obs(self):
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
    def step_sequence(self, action_seq):
        """Step through an action sequence.

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
        """Return True if the state of the environment matches the goal state (or is within a certain distance to goal"""
        pass
