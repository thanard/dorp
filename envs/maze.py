import argparse
# import safety_gym
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time

from safety_gym.envs.engine import Engine
from PIL import Image
from imageio import imwrite
from envs import GoalEnv
from utils.dataset import *
import os


CAMERA_FREE = 0
CAMERA_FIXED = 1

class Maze(GoalEnv):
    def __init__(self, num_steps=20000, step_size=50):
        super(Maze, self).__init__()
        self.goal_im = None
        self.step_size = step_size
        self.grid_n = 64
        self.maze_env = SafetyGymMaze('point')

        self.maze_env.reset()
        self.maze_env.render(mode='rgb_array', camera_id=2)
        # env.render(camera_id=2)
        self.maze_env.viewer_setup()
        self.maze_env.set_num_steps(num_steps)
        self.state = self.maze_env.robot_pos
        self.name = 'maze'

    def reset(self):
        """Resets the state of the environment and the goal, returning an initial observation (images)."""
        self.maze_env.reset()
        self.goal_im = self.get_obs()
        self.goal_state = self.maze_env.robot_pos

        self.maze_env.reset()
        return self.get_obs()

    def seed(self, seed):
        """Set the random seed."""
        np.random.seed(seed)

    def get_state(self):
        return self.maze_env.robot_pos

    def get_obs(self):
        """Return current image observation."""
        img = self.maze_env.render(mode='rgb_array', camera_id=2)
        img = np.array(Image.fromarray(img).resize((64, 64), resample=Image.NEAREST))
        return img

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        :param action:
        :return:
        """
        for i in range(self.step_size):
            self.maze_env.step(action[i])

    def step_sequence(self, action_seq):

        for action in action_seq:
            self.step(action)

    def sample_action(self):
        """Return a uniformly sampled action from the action space"""
        action = []
        for i in range(self.step_size):
            action.append(self.maze_env.action_space.sample())
        return action

    def reached_goal(self):
        """Return True if the state of the environment matches the goal state"""
        pass

class SafetyGymMaze(Engine):
    def __init__(self, agent, robot_xy = None):
        surrounding_walls = set()
        for i in (np.linspace(0, 1, 15) * 7 - 3.5):
            surrounding_walls.add((i, 3.5))
            surrounding_walls.add((i, -3.5))
            surrounding_walls.add((-3.5, i))
            surrounding_walls.add((3.5, i))

        self.wall_locs = [(2, -2.5), (1.5, -2), (1, -1.5),
                          (2, -1),
                          (-1, 0), (-0.5, 0), (0, 0), (0.5, 0), (1, 0), (0, -0.5), (0, -1),
                          (0.5, 2), (1, 2), (1.5, 2), (1.5,1.5), (2, 2), (2.5, 2),
                          (-1.5, 2), (-2, 1.5), (-2, 1), (-2.5, 1), (-2.5, 0.5), (-1, 2),
                          (-2.5, -1), (-2.5, -1.5), (-2.5, -2), (-2.5, -2.5), (-2.5, -3),
                          (-0.5, -3), (-1, -3), (-0.5, -2.5)]  # simple maze wall locations

        self.wall_locs.extend(surrounding_walls)

        config = {
            'robot_base': 'xmls/%s.xml' % agent,
            'task': 'none',
            # 'observe_goal_lidar': True,
            'observe_walls': True,
            # 'observe_hazards': True,
            'observation_flatten': True,
            # 'observe_box_lidar': True,
            'constrain_hazards': True,
            'walls_locations': self.wall_locs,
            'walls_num': len(self.wall_locs),
            'walls_size': 0.25,
            'observe_qpos': True,
        }

        if robot_xy:
            config['robot_locations'] = robot_xy

        super(SafetyGymMaze, self).__init__(config)
        # register(id='SimpleMaze-v0',
        #          entry_point='safety_gym.envs.mujoco:Engine',
        #          kwargs={'config': config})

    def detect_wall_contact(self, pos):
        for wall in self.wall_locs:
            wall = np.array(wall)
            pos = np.array(pos)
            if np.all(np.abs(np.array(wall-pos)) < (0.25 + 1e-1)):
                return True
        return False

    def viewer_setup(self):
        print("cam", self.viewer.cam.type)

    def set_num_steps(self, num_steps):
        self.world.num_steps = num_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample trajectories from safety gym')
    parser.add_argument('--n_traj', type=int, default=200, help='number of simulations')
    parser.add_argument('--len_traj', type=int, default=150)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--savepath', type=str, default="datasets/")

    args = parser.parse_args()
    env = Maze(step_size=args.step_size)
    env.reset()

    data_dir = os.path.join(args.savepath, 'maze')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    start = time.time()
    env.reset()
    all_data = []
    for n in range(args.n_traj):
        print("Collecting traj %d...... " % n)
        env.reset()
        traj_ims = []
        for t in range(args.len_traj):
            action = env.sample_action()
            env.step(action)
            traj_ims.append(env.get_obs())
        all_data.append(np.array(traj_ims))
    all_data = np.array(all_data)

    data = get_sample_transitions(env, args.n_traj, args.len_traj)

    fname = 'maze_env_randact_traj_length_%d_n_traj_%d_stepsize_%d.npy' % \
                    (args.len_traj, args.n_traj, args.step_size)
    data_path = os.path.join(data_dir, fname)
    np.save(data_path, data)
    total = time.time() - start

    print("Finished collecting trajectories. Took {} seconds.".format(total))