from envs import GoalEnv
import numpy as np

RGB_COLORS = {
    "yellow": [255, 255, 50],
    "cyan": [100, 255, 255],
    "purple": [128, 0, 255],
    "red": [255, 0, 0],
    "green": [128, 255, 0],
    "blue" : [0, 0, 255],
    "orange" : [255, 165, 0],
    "teal" : [0, 50, 50],
    "bluegrey" : [98, 111, 115],
    "grey": [80, 80, 80],
} # color of each agent in order

# AGENT_SIZES = [[7,7], [7,7], [7,7], [7,7], [7,7]]
# AGENT_SIZES = [[2,4], [4,2], [2,4], [2,4], [4,2]]
# AGENT_SIZES = [[1,1], [1,1], [1,1], [1,1], [1,1]]
# AGENT_SIZES = [[3,3], [3,3], [3,3], [3,3], [3,3]]
AGENT_SIZES = [[1,2], [2,1], [1,2], [2,1], [1,2], [2,1], [1,2], [2,1], [1,2], [2,1]]

class GridWorld(GoalEnv):

    def __init__(self, n_agents=1, grid_n=16, step_size=1):
        super(GridWorld, self).__init__()
        self.name = 'gridworld'
        self.grid_n = grid_n
        self.n_agents = n_agents
        self.step_size=1

        self.actions = np.eye(2*n_agents)
        self.actions = np.concatenate((self.actions, -1*self.actions)).astype('uint8')

    def reset(self, state=None):
        if type(state) is np.ndarray:
            self.state = state
            return state
        self.state = self.get_reset_position()
        self.goal_state = self.get_reset_position()
        self.render_goal()
        return self.state

    def get_reset_position(self):
        seen = set()
        cur_sample = []
        for i in range(self.n_agents):
            sample_pos = self.sample_single_agent_pos(i)
            while self.is_overlapping(i, sample_pos, seen):
                sample_pos = self.sample_single_agent_pos(i)
            agent_dim = AGENT_SIZES[i]
            disps = np.mgrid[0:agent_dim[0], 0:agent_dim[1]].T.reshape(-1, 2)
            pos_to_add = disps + np.tile(sample_pos, (agent_dim[0] * agent_dim[1], 1))
            pos_to_add = set(list(map(tuple, pos_to_add)))
            seen = seen | pos_to_add
            # agent_dim = AGENT_SIZES[i]
            # for x in range(agent_dim[0]):
            #     for y in range(agent_dim[1]):
            #         disp = np.array([x, y])
            #         seen.add(tuple(disp + sample_pos))
            cur_sample.append(sample_pos)
        pos = np.concatenate(cur_sample).astype('uint8')
        return pos

    def seed(self, seed):
        """Set the random seed."""
        np.random.seed(seed)

    def render(self):
        """Return current image observation as RGB image"""
        colors = list(RGB_COLORS.values())[:self.n_agents]
        im = np.zeros((self.grid_n, self.grid_n, 3))
        for i in range(self.n_agents):
            agent_dim = AGENT_SIZES[i]
            x_cur, y_cur = self.state[2 * i], self.state[2 * i + 1]
            im[x_cur:x_cur+agent_dim[0], y_cur:y_cur+agent_dim[1]] += \
                np.tile(np.tile(colors[i], (agent_dim[1], 1)), (agent_dim[0], 1, 1))
            # for x in range(agent_dim[0]):
            #     for y in range(agent_dim[1]):
            #         im[(x_cur + x) % self.grid_n, (y_cur + y) % self.grid_n] += colors[i]
        return im

    def render_goal(self):
        cur_state = self.state
        self.reset(self.goal_state)
        self.goal_im = self.render()
        self.state = cur_state
        return self.goal_im

    def step(self, action):
        max_positions = self.get_max_agent_positions()
        cur_pos = self.state
        seen = set()
        for i in range(self.n_agents):
            agent_pos = self.state[2 * i:2 * i + 2]
            agent_dim = AGENT_SIZES[i]
            disps = np.mgrid[0:agent_dim[0], 0:agent_dim[1]].T.reshape(-1, 2)
            pos_to_add = disps + np.tile(agent_pos, (agent_dim[0] * agent_dim[1], 1))
            pos_to_add = set(list(map(tuple, pos_to_add)))
            seen = seen | pos_to_add
            # for x in range(agent_dim[0]):
            #     for y in range(agent_dim[1]):
            #         disp = np.array([x, y])
            #         seen.add(tuple(disp + agent_pos))
        # clamp next position
        next_pos = (self.state + action)
        next_pos = np.amax((np.zeros_like(max_positions),
                            np.amin((next_pos, max_positions), axis=0)), axis=0)

        agent_moved = np.where(action != 0)[0][0] // 2
        agent_dim = AGENT_SIZES[agent_moved]
        agent_pos = self.state[2 * agent_moved: 2 * agent_moved + 2]
        # remove the current agent's occupied positions from seen

        disps = np.mgrid[0:agent_dim[0], 0:agent_dim[1]].T.reshape(-1,2)
        pos_to_remove = disps + np.tile(agent_pos, (agent_dim[0]*agent_dim[1], 1))
        pos_to_remove = set(list(map(tuple, pos_to_remove)))
        seen = seen - pos_to_remove
        # for x in range(agent_dim[0]):
        #     for y in range(agent_dim[1]):
        #         disp = np.array([x, y])
        #         seen.remove(tuple(disp + agent_pos))
        # check if action results in overlap with any other agent
        if not self.is_overlapping(agent_moved, next_pos[2 * agent_moved:2 * agent_moved + 2], seen):
            cur_pos = next_pos.astype('uint8')

        # agent_new_pos = cur_pos[2 * agent_moved: 2 * agent_moved + 2]
        # pos_to_add = disps + np.tile(agent_new_pos, (agent_dim[0] * agent_dim[1], 1))
        # pos_to_add = set(list(map(tuple, pos_to_add)))
        # seen = seen | pos_to_add
        # for x in range(agent_dim[0]):
        #     for y in range(agent_dim[1]):
        #         disp = np.array([x, y])
        #         seen.add(tuple(disp + agent_new_pos))
        self.state = cur_pos
        return cur_pos

    def step_sequence(self, action_seq):
        for action in action_seq:
            cur_pos = self.step(action)
        return self.state

    def sample_action(self):
        action_idx = np.random.choice(len(self.actions))
        return self.actions[action_idx]

    def reached_goal(self):
        return np.array_equal(self.goal_state, self.state)

    def get_max_agent_positions(self):
        '''
        Returns the max possible x and y values for each agent based on AGENT_SIZES
        Only used when self.circular=False
        '''
        max_positions = []
        for agent_idx in range(self.n_agents):
            max_x = self.grid_n - AGENT_SIZES[agent_idx][0]
            max_y = self.grid_n - AGENT_SIZES[agent_idx][1]
            max_positions.append(np.array([max_x, max_y]))
        return np.concatenate(max_positions)

    def sample_single_agent_pos(self, agent_idx):
        max_positions = self.get_max_agent_positions()
        max_position_agent = max_positions[2 * agent_idx:2 * agent_idx + 2]
        sample_pos_x = np.random.randint(max_position_agent[0])
        sample_pos_y = np.random.randint(max_position_agent[1])
        sample_pos = np.array((sample_pos_x, sample_pos_y))
        return sample_pos

    def is_overlapping(self, agent_idx, sample_pos, seen):
        '''
        agent_idx: index of the agent to check for overlaps
        sample pos: candidate position for this agent
        seen: set of positions for other agents
        Returns True if the agent at agent_idx overlaps with any other agent whose position is present in 'seen' (set)
        '''
        agent_dim = AGENT_SIZES[agent_idx]
        disps = np.mgrid[0:agent_dim[0], 0:agent_dim[1]].T.reshape(-1, 2)
        pos_to_check = disps + np.tile(sample_pos, (agent_dim[0] * agent_dim[1], 1))
        pos_to_check = set(list(map(tuple, pos_to_check)))
        if seen.intersection(pos_to_check):
            return True
        # for x in range(agent_dim[0]):
        #     for y in range(agent_dim[1]):
        #         disp = np.array([x, y])
        #         if (tuple(disp + sample_pos)) in seen:
        #             return True
        return False

    def image_to_position_thanard(self, im, agent_idx):
        up_left_pos = self.find_top_left(im[:, :, agent_idx])
        return up_left_pos

    def image_to_position(self, im, agent_idx):
        '''
        :param im: A (grid_n, grid_n, n ) image where n is # agents
        :param agent_idx: idx of the agent to locate
        :param grid_n: size of grid
        :return: positon of the agent corresponding to the channel
        '''

        agent_grid = im[:, :, agent_idx].astype('uint8')
        if agent_grid[0, 0] != 0 and agent_grid[(self.grid_n - 1), (self.grid_n - 1)] != 0:
            # if agent occupies upper left and bottom right corner, look at bottom right square for position
            last_col = agent_grid[:, self.grid_n - 1]
            bottom_right_x = -1
            for i in np.arange(self.grid_n - 1, -1, -1):
                if last_col[i] == 0:
                    break
                bottom_right_x = i
            assert bottom_right_x != -1
            bottom_right_y = -1
            for i in np.arange(self.grid_n - 1, -1, -1):  # count backwards from the right side of grid
                if agent_grid[bottom_right_x, i] == 0:
                    break
                bottom_right_y = i

            return np.array([bottom_right_x, bottom_right_y])

        first_row, last_row = agent_grid[0], agent_grid[self.grid_n - 1]
        first_col, last_col = agent_grid[:, 0], agent_grid[:, self.grid_n - 1]

        if np.where(first_row != 0)[0].any() and np.where(last_row != 0)[0].any():
            # if agent occupies first and last row, looks at bottom area for position
            bottom_y = np.where(last_row != 0)[0][0]
            bottom_x = -1
            for i in np.arange(self.grid_n - 1, -1, -1):
                if agent_grid[i, bottom_y] == 0:
                    break
                bottom_x = i
            assert bottom_x != -1

            return np.array([bottom_x, bottom_y])

        elif np.where(first_col != 0)[0].any() and np.where(last_col != 0)[0].any():
            right_x = np.where(last_col != 0)[0][0]
            right_y = -1
            for i in np.arange(self.grid_n - 1, -1, -1):
                if agent_grid[right_x, i] == 0:
                    break
                right_y = i
            assert right_y != -1

            return np.array([right_x, right_y])
        else:
            positions = np.where(im[:, :, agent_idx] != 0)
            up_left_pos = np.array((positions[0][0], positions[1][0]))
            return up_left_pos

    def find_top_left(self, array2d):
        assert array2d.shape[0] == array2d.shape[1]
        assert len(array2d.shape) == 2
        grid_n = array2d.shape[0]
        shift_by = np.zeros(2)
        for dim in range(2):
            print(dim)
            for i in range(grid_n - 1):
                if array2d.take(i, axis=dim).sum() == 0 and array2d.take(i + 1, axis=dim).sum() == 0:
                    array2d = np.concatenate([array2d.take(range(i + 1, grid_n), axis=dim),
                                              array2d.take(range(i + 1), axis=dim)],
                                             axis=dim)
                    shift_by[dim] = i + 1
                    break
        positions = np.where(array2d > 0)
        position_x = (positions[0][0] + shift_by[0]) % grid_n
        position_y = (positions[1][0] + shift_by[1]) % grid_n
        return np.array([position_x, position_y]).astype('int')