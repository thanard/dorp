from envs import GoalEnv
import numpy as np
import numpy.lib.arraysetops as aso
from torchvision.utils import save_image
from utils.gen_utils import *

GRID_N = 16

BLACK = [0, 0, 0]
GREY = [80, 80, 80]
RED = [255, 0 ,0]
CYAN = [100, 255, 255]
GREEN = [128, 255, 0]
YELLOW = [255, 255, 50]
PURPLE = [128, 0, 255]
BLUE = [0, 0, 255]
ORANGE = [255, 165, 0]
TEAL = [0, 50, 50]
BLUEGREY = [98, 111, 115]

OBJ_TO_IDX = {
    'empty': 0,
    'agent0': 1,
    'agent1': 2,
    'agent2': 3,
    'wall': 4,
    'door0': 5,
    'door1': 6,
    'door2': 7,
    'key0': 8,
    'key1': 9,
    'key2': 10,
}

DOORS = {OBJ_TO_IDX['door0'], OBJ_TO_IDX['door1'], OBJ_TO_IDX['door2']}
KEYS = {OBJ_TO_IDX['key0'], OBJ_TO_IDX['key1'], OBJ_TO_IDX['key2']}

COLORS= [BLACK, #0
         RED, #1
         RED, #2
         RED, #3
         GREY, #4
         YELLOW,#5
         BLUE,#6
         TEAL,#7
         ORANGE,#8
         PURPLE,#9
         GREEN,#10
         ]

class KeyDoor(GoalEnv):
    """Goal-based Environment"""

    '''
        empty grid, no walls, keys, or obstacles
        no wrapping (agent cannot move off the grid)
        '''

    def __init__(self, n_agents):
        super(KeyDoor, self).__init__()
        self.name = 'key'
        self.grid_n = GRID_N
        self.n_agents = n_agents
        self.n_keys = 0
        self.step_size = 1

        self.actions = np.concatenate((np.eye(2), -1 * np.eye(2))).astype('int')
        self.grid = np.zeros((GRID_N, GRID_N)).astype('uint8')
        self.agent_pos = np.zeros(2)
        self.carrying = [set([]) for _ in range(self.n_agents)]
        self.state = [self.agent_pos, to_list(self.carrying)]# state

    def process_grid(self):
        '''
        updates grid to be onehot encoding of objects at grid position
        :return:
        '''
        for i in range(self.n_agents):
            self.grid[self.get_one_agent_pos(self.agent_pos, i)] = OBJ_TO_IDX['agent' + str(i)]
        self.grid = self.grid.astype('uint8')

    def reset(self, state=None):
        '''
        all agents are reset to random position
        :return:
        '''
        self.goal_im, self.goal_state = self.sample_and_process_goal()
        self.grid = np.zeros((GRID_N, GRID_N)).astype('uint8')
        agent_position = np.random.randint(0, GRID_N, 2)
        self.agent_pos = agent_position
        self.process_grid()
        self.state = [self.agent_pos, to_list(self.carrying)]
        return agent_position


    def sample_action(self):
        action = self.actions[np.random.randint(len(self.actions))]
        return action

    def detect_border_contact(self, action):
        if (self.agent_pos + action >= GRID_N).any() or \
                (self.agent_pos + action < 0).any():
            return True
        return False

    def step(self, action):
        if (self.agent_pos + action >= GRID_N).any() or (self.agent_pos + action < 0).any():
            return self.agent_pos
        self.agent_pos += action
        self.state = [self.agent_pos, to_list(self.carrying)]
        return self.state

    def step_sequence(self, action_seq):
        for action in action_seq:
            self.step(action)
        return self.state

    def get_one_agent_pos(self, pos, agent_idx):
        return tuple(pos[2 * agent_idx: 2 * agent_idx + 2])

    def get_obj(self, pos):
        return self.grid[tuple(pos)]

    def get_obs(self):
        im = self.render()
        return im

    def render(self):
        rgb_im = np.zeros((GRID_N, GRID_N, 3))
        for row in range(GRID_N):
            for col in range(GRID_N):
                rgb_im[row, col] = COLORS[self.grid[row, col]]
        return rgb_im.astype('float32')

    def sample_random_trajectory(self, len_traj):
        traj_data = []
        for i in range(len_traj):
            action = self.sample_action()
            self.step(action)
            traj_data.append(self.get_obs())
        return np.array(traj_data)

    def seed(self, seed):
        """Set the random seed."""
        np.random.seed(seed)

    def reached_goal(self):
        goal_carrying = np.array(self.goal_state[1][0])
        carrying = np.array(list(self.carrying[0]))
        if np.array_equal(self.agent_pos, self.goal_state[0]) \
                and len(aso.setdiff1d(goal_carrying, carrying)) == 0:
            return True
        return False

    def sample_and_process_goal(self):
        self.grid = np.zeros((GRID_N, GRID_N)).astype('uint8')
        agent_position = np.random.randint(0, GRID_N, 2)
        self.agent_pos = agent_position
        self.process_grid()
        self.state = [self.agent_pos, to_list(self.carrying)]

        goal_im = self.get_obs()
        label = [self.agent_pos, to_list(self.carrying)]
        return goal_im, label


class KeyCorridor(KeyDoor):
    '''
        - adds walls to minigrid to make 4 rooms plus a hallway
        - up to 3 keys
        - n_keys == n_doors, keys are always on left side, locked doors always on right side
    '''
    def __init__(self, n_agents, n_keys):
        super().__init__(n_agents)
        self.n_keys = n_keys
        self.n_doors = n_keys
        self.wall_poses = set([])
        self.key_poses = []
        self.door_poses = []
        self.name = 'key-corridor'

    def setup_walls(self):
        for i in range(GRID_N):
            self.wall_poses.add((i, 5))
            self.wall_poses.add((i, 10))
        for i in range(6):
            self.wall_poses.add((5, i))
            self.wall_poses.add((5, i+10))
            self.wall_poses.add((10, i))
            self.wall_poses.add((10, i + 10))
        openings = {(3, 5), (4,5), (6, 5), (7,5), (11, 5), (12,5)}
        self.wall_poses = self.wall_poses - openings
        openings_if_no_door = [[(0, 10), (1,10)], [(7, 10), (8, 10)], [(13, 10), (14,10)]]
        for i in range(self.n_keys, 3): # max 3 keys
            for open_pos in openings_if_no_door[i]:
                self.wall_poses = self.wall_poses - {open_pos}

    def reset_keys_and_doors(self):
        self.key_poses = []
        self.door_poses = []
        self.carrying = [set([]) for _ in range(self.n_agents)]
        all_key_pos = ((15, 0), (0,0), (7, 0))
        all_door_pos = ([(0, 10), (1,10)], [(7, 10), (8, 10)], [(13, 10), (14,10)])
        for key_idx in range(self.n_keys):
            self.key_poses.append(all_key_pos[key_idx])
            self.door_poses.append(all_door_pos[key_idx])
            for door in all_door_pos[key_idx]:
                if door in self.wall_poses:
                    self.wall_poses.remove(door)

    def reset(self, state=None):
        if isinstance(state, list):
            return self.reset_to_state(state)
        self.grid = np.zeros((GRID_N, GRID_N))
        self.reset_keys_and_doors()
        self.setup_walls()
        self.agent_pos = self.sample_agent_positions()
        self.process_grid()
        self.state = [self.agent_pos, to_list(self.carrying)]
        self.goal_im, self.goal_state = self.sample_and_process_goal()
        return self.agent_pos

    def reset_random_key_config(self):
        # Resets with random number of keys on the grid
        self.grid = np.zeros((GRID_N, GRID_N))
        self.reset_keys_and_doors()
        self.setup_walls()
        self.agent_pos = self.sample_agent_positions()
        self.process_grid()
        n_keys_on_grid = np.random.choice(self.n_keys)
        for key_idx in range(n_keys_on_grid):
            self.remove_key(key_idx, 0) # give agent 0 key if removed from grid
        return self.agent_pos

    def sample_and_process_goal(self):
        cur_state = self.state
        self.grid = np.zeros((GRID_N, GRID_N))
        self.reset_keys_and_doors()
        self.setup_walls()
        self.agent_pos = self.sample_agent_positions_no_keys()
        self.process_grid()
        self.remove_all_keys()
        goal_im = self.get_obs()
        label = (self.agent_pos, to_list(self.carrying))
        self.reset_to_state(cur_state)
        return goal_im, label

    def randomize_agent_pos(self):
        self.grid[self.agent_pos[0], self.agent_pos[1]] = OBJ_TO_IDX['empty']
        new_pos = self.sample_agent_positions()
        self.grid[new_pos[0], new_pos[1]] = OBJ_TO_IDX['agent0']
        self.agent_pos=new_pos

    def get_one_key_pos(self, key_idx):
        return tuple(self.key_poses[key_idx])

    def get_one_door_pos(self, door_idx):
        return tuple(self.door_poses[door_idx])

    def get_all_door_poses(self):
        all_door_poses = set()
        for door in self.door_poses:
            for door_pos in door:
                all_door_poses.add(door_pos)
        return all_door_poses

    def process_grid(self):
        assert self.n_keys == self.n_doors
        for i in range(self.n_agents):
            self.grid[self.get_one_agent_pos(self.agent_pos, i)] = OBJ_TO_IDX['agent'+str(i)]
        for wall in self.wall_poses:
            self.grid[wall] = OBJ_TO_IDX['wall']
        for i in range(self.n_keys):
            self.grid[self.get_one_key_pos(i)] = OBJ_TO_IDX['key'+str(i)]
        for i in range(self.n_doors):
            door_poses = self.get_one_door_pos(i)
            for door in door_poses:
                self.grid[door]= OBJ_TO_IDX['door'+str(i)]
        self.grid = self.grid.astype('uint8')

    def remove_all_keys(self):
        '''
        removes all keys and doors from the environment
        **Only use this method for visualization or sampling goal images
        **Make sure to call env.reset() after calling this
        '''
        for key_idx in range(self.n_keys):
            self.carrying[0].add(key_idx)
            key_pos = self.key_poses[key_idx]
            self.grid[key_pos] = OBJ_TO_IDX['empty']
            self.key_poses[key_idx] = ()
            self.remove_door(key_idx)

    def sample_agent_positions_no_keys(self):
        possible_positions = np.mgrid[0:GRID_N, 0:GRID_N].reshape(2, -1).T
        possible_positions = set(tuple(map(tuple, possible_positions)))
        possible_positions = possible_positions - self.wall_poses
        possible_positions = possible_positions - set(self.key_poses)
        possible_positions = possible_positions - self.get_all_door_poses()

        possible_positions = np.array(list(possible_positions))
        agent_positions = possible_positions[np.random.randint(len(possible_positions), size=self.n_agents)]
        return agent_positions.flatten().astype('uint8')

    def try_place_agent(self, pos):
        '''
        pos: array size 2*n_agents
        Tries to place the agents at specified (x,y) position
        Returns false if this is not possible
        **Only use this method for visualization
        '''
        # remove all current agents
        for i in range(self.n_agents):
            agent_pos = self.get_one_agent_pos(self.agent_pos, i)
            self.grid[agent_pos] = OBJ_TO_IDX['empty']

        # try to set all agent positions:
        for i in range(self.n_agents):
            new_agent_pos = self.get_one_agent_pos(pos, i)
            obj = self.get_obj(new_agent_pos)
            if obj == OBJ_TO_IDX['wall'] or obj in DOORS:
                return False
            self.grid[new_agent_pos] = OBJ_TO_IDX['agent' + str(i)]
        self.agent_pos = pos
        return True

    def sample_agent_positions(self):
        possible_positions = np.mgrid[0:GRID_N, 0:GRID_N].reshape(2,-1).T
        possible_positions = set(tuple(map(tuple, possible_positions)))
        possible_positions = possible_positions - self.wall_poses
        possible_positions = possible_positions - set(self.key_poses)
        possible_positions = possible_positions - self.get_all_door_poses()

        # do not reset in room with locked door
        room0 = set(tuple(map(tuple, np.mgrid[0:5, 10:GRID_N].reshape(2,-1).T)))
        room1 = set(tuple(map(tuple, np.mgrid[6:10, 10:GRID_N].reshape(2, -1).T)))
        room2 = set(tuple(map(tuple, np.mgrid[11:GRID_N, 10:GRID_N].reshape(2, -1).T)))
        rooms = [room0, room1, room2]
        for i in range(self.n_keys):
            possible_positions = possible_positions - rooms[i]
        possible_positions = np.array(list(possible_positions))
        agent_positions = possible_positions[np.random.randint(len(possible_positions), size=self.n_agents)]
        return agent_positions.flatten().astype('uint8')

    def detect_border_contact(self, action):
        # detect wall contact
        new_pos = self.agent_pos + action
        for i, wall in enumerate(self.wall_poses):
            if tuple(self.get_one_agent_pos(new_pos, i)) == wall:
                return True

        if (self.agent_pos + action >= GRID_N).any() or \
            (self.agent_pos + action < 0).any():
            return True
        return False

    def remove_door(self, door_idx):
        door_poses = self.door_poses[door_idx]
        for door_pos in door_poses:
            self.grid[door_pos] = OBJ_TO_IDX['empty']
        self.door_poses[door_idx] = []

    def remove_key(self, key_idx, agent_idx):
        self.carrying[agent_idx].add(key_idx)
        key_pos = self.key_poses[key_idx]
        self.key_poses[key_idx] = ()
        self.grid[key_pos] = OBJ_TO_IDX['empty']
        self.remove_door(key_idx)

    def pickup_key(self, agent_idx, key_pos):
        # when the key is no longer on the grid, set its position to ()
        # adds this key to agent's list of carried items
        # changes agent position to key's position
        # removes key from grid
        key_idx = -1
        for i, key in enumerate(self.key_poses):
            # print("key_pos, key", key_pos, key)
            if key_pos == key:
                key_idx = i
                break

        assert key_idx != -1
        self.remove_key(key_idx, agent_idx)
        self.agent_pos = key_pos

    def step(self, action, interact_with_key=True):
        agent_idx = np.where(action != 0)[0][0] // 2
        old_pos = self.agent_pos
        new_pos = self.agent_pos+action
        if self.detect_border_contact(action):
            return self.agent_pos
        new_pos_xy = self.get_one_agent_pos(new_pos, agent_idx)
        obj = self.get_obj(new_pos_xy)
        # print("obj", obj)
        if obj == OBJ_TO_IDX['wall'] or obj in DOORS:
            return self.agent_pos
        elif obj in KEYS:
            if interact_with_key:
                self.pickup_key(agent_idx, new_pos_xy)
            else:
                return self.agent_pos
        else:
            self.agent_pos = new_pos

        self.grid[self.get_one_agent_pos(old_pos, agent_idx)] = OBJ_TO_IDX['empty']
        self.grid[self.get_one_agent_pos(self.agent_pos, agent_idx)] = OBJ_TO_IDX['agent' + str(agent_idx)]
        return self.agent_pos

    def reset_to_state(self, state):
        # state is in the form [agent_pos, env.carrying]
        self.grid = np.zeros((GRID_N, GRID_N))
        self.reset_keys_and_doors()
        self.setup_walls()

        new_pos, carrying = state[0], state[1]
        if self.try_place_agent(new_pos):
            for agent_idx, agent_carrying in enumerate(carrying):
                if len(agent_carrying) > 0:
                    for key_idx in agent_carrying:
                        self.remove_key(key_idx, agent_idx)
                self.process_grid()
                self.state = state
            return state
        return None

    def sample_random_trajectory(self, len_traj, interact_with_key=True):
        traj_data = []
        for i in range(len_traj):
            action = self.sample_action()
            self.step(action, interact_with_key=interact_with_key)
            traj_data.append(self.get_obs())
        return np.array(traj_data)

class KeyWall(KeyCorridor):
    '''
    single key, wall down the middle of grid, with single door
    '''
    def __init__(self, n_agents):
        super().__init__(n_agents, 1)
        self.name = 'key-wall'

    def setup_walls(self):
        for i in range(GRID_N):
            self.wall_poses.add((i,8))

    def reset_keys_and_doors(self):
        self.key_poses = []
        self.door_poses = []
        self.carrying = [set([]) for _ in range(self.n_agents)]
        all_key_pos = [(0, 0)]
        all_door_pos = [[(8, 8), (7,8), (6,8)]]
        for key_idx in range(self.n_keys):
            self.key_poses.append(all_key_pos[key_idx])
            self.door_poses.append(all_door_pos[key_idx])
            for door in all_door_pos[key_idx]:
                if door in self.wall_poses:
                    self.wall_poses.remove(door)

    def sample_agent_positions(self):
        # do not reset on right side
        possible_positions = np.mgrid[0:GRID_N, 0:GRID_N].reshape(2, -1).T
        possible_positions = set(tuple(map(tuple, possible_positions)))
        possible_positions = possible_positions - self.wall_poses
        possible_positions = possible_positions - set(self.key_poses)
        possible_positions = possible_positions - self.get_all_door_poses()

        # do not reset in room with locked door
        room0 = set(tuple(map(tuple, np.mgrid[0:GRID_N, 7:GRID_N].reshape(2, -1).T)))
        rooms = [room0]
        for i in range(self.n_keys):
            possible_positions = possible_positions - rooms[i]
        possible_positions = np.array(list(possible_positions))
        agent_positions = possible_positions[np.random.randint(len(possible_positions), size=self.n_agents)]
        return agent_positions.flatten().astype('uint8')

class KeyWall2Keys(KeyCorridor):
    '''
        double key, wall down the middle of grid, with 2 doors, 2 rooms
        '''

    def __init__(self, n_agents):
        super().__init__(n_agents, 2)

    def setup_walls(self):
        for i in range(GRID_N):
            self.wall_poses.add((i, 8))
        for i in range(8, GRID_N):
            self.wall_poses.add((8,i))

    def reset_keys_and_doors(self):
        self.key_poses = []
        self.door_poses = []
        self.carrying = [set([]) for _ in range(self.n_agents)]
        all_key_pos = [(0, 0), (GRID_N-1, 0)]
        all_door_pos = [[(4, 8), (5, 8)], [(11,8), (12,8)]]
        for key_idx in range(self.n_keys):
            self.key_poses.append(all_key_pos[key_idx])
            self.door_poses.append(all_door_pos[key_idx])
            for door in all_door_pos[key_idx]:
                if door in self.wall_poses:
                    self.wall_poses.remove(door)

    def sample_agent_positions(self):
        # do not reset on right side
        possible_positions = np.mgrid[0:GRID_N, 0:GRID_N].reshape(2, -1).T
        possible_positions = set(tuple(map(tuple, possible_positions)))
        possible_positions = possible_positions - self.wall_poses
        possible_positions = possible_positions - set(self.key_poses)
        possible_positions = possible_positions - self.get_all_door_poses()

        # do not reset in room with locked door
        room0 = set(tuple(map(tuple, np.mgrid[0:GRID_N, 7:GRID_N].reshape(2, -1).T)))
        rooms = [room0]
        for i in range(len(rooms)):
            possible_positions = possible_positions - rooms[i]
        possible_positions = np.array(list(possible_positions))
        agent_positions = possible_positions[np.random.randint(len(possible_positions), size=self.n_agents)]
        return agent_positions.flatten().astype('uint8')


class KeyWallRandom(KeyCorridor):
    '''
    single key, wall down the middle of grid, with single door
    '''
    def __init__(self, n_agents):
        super().__init__(n_agents, 1)
        self.name = 'key-wall-random'

    def setup_walls(self):
        for i in range(GRID_N):
            self.wall_poses.add((i,8))

    def reset_keys_and_doors(self):
        self.key_poses = []
        self.door_poses = []
        self.carrying = [set([]) for _ in range(self.n_agents)]
        key_pos_x = np.random.randint(GRID_N)
        key_pos_y = np.random.randint(GRID_N//2)
        all_key_pos = [(key_pos_x, key_pos_y)]
        all_door_pos = [[(8, 8), (7,8), (6,8)]]
        for key_idx in range(self.n_keys):
            self.key_poses.append(all_key_pos[key_idx])
            self.door_poses.append(all_door_pos[key_idx])
            for door in all_door_pos[key_idx]:
                if door in self.wall_poses:
                    self.wall_poses.remove(door)

    def sample_agent_positions(self):
        # do not reset on right side
        possible_positions = np.mgrid[0:GRID_N, 0:GRID_N].reshape(2, -1).T
        possible_positions = set(tuple(map(tuple, possible_positions)))
        possible_positions = possible_positions - self.wall_poses
        possible_positions = possible_positions - set(self.key_poses)
        possible_positions = possible_positions - self.get_all_door_poses()

        # do not reset in room with locked door
        room0 = set(tuple(map(tuple, np.mgrid[0:GRID_N, 7:GRID_N].reshape(2, -1).T)))
        rooms = [room0]
        for i in range(self.n_keys):
            possible_positions = possible_positions - rooms[i]
        possible_positions = np.array(list(possible_positions))
        agent_positions = possible_positions[np.random.randint(len(possible_positions), size=self.n_agents)]
        return agent_positions.flatten().astype('uint8')

class KeyCorridorRandom(KeyCorridor):
    def __init__(self, n_agents):
        super().__init__(n_agents, 1)
        self.name = 'key-corridor-random'

    def reset_keys_and_doors(self):
        self.key_poses = []
        self.door_poses = []
        self.carrying = [set([]) for _ in range(self.n_agents)]
        key_pos_x = np.random.randint(GRID_N)
        key_pos_y = np.random.randint(10)
        while (key_pos_x, key_pos_y) in self.wall_poses:
            key_pos_x = np.random.randint(GRID_N)
            key_pos_y = np.random.randint(10)

        all_key_pos = [(key_pos_x, key_pos_y)]
        all_door_pos = [[(0, 10), (1, 10)]]

        for key_idx in range(self.n_keys): # should be 1 key
            self.key_poses.append(all_key_pos[key_idx])
            self.door_poses.append(all_door_pos[key_idx])
            for door in all_door_pos[key_idx]:
                if door in self.wall_poses:
                    self.wall_poses.remove(door)


class KeyWallSequential(KeyCorridor):
    '''
    Same as KeyWall2Keys, but one key is inside a locked room
    '''
    def __init__(self, n_agents):
        super().__init__(n_agents, 2)
        self.name = 'key-wall-seq'

    def setup_walls(self):
        for i in range(GRID_N):
            self.wall_poses.add((i, 8))
        for i in range(8, GRID_N):
            self.wall_poses.add((8,i))

    def reset_keys_and_doors(self):
        self.key_poses = []
        self.door_poses = []
        self.carrying = [set([]) for _ in range(self.n_agents)]
        all_key_pos = [(0, 0), (3, 13)]
        all_door_pos = [[(4, 8), (5, 8)], [(11,8), (12,8)]]
        for key_idx in range(self.n_keys):
            self.key_poses.append(all_key_pos[key_idx])
            self.door_poses.append(all_door_pos[key_idx])
            for door in all_door_pos[key_idx]:
                if door in self.wall_poses:
                    self.wall_poses.remove(door)

    def sample_agent_positions(self):
        # do not reset on right side
        possible_positions = np.mgrid[0:GRID_N, 0:GRID_N].reshape(2, -1).T
        possible_positions = set(tuple(map(tuple, possible_positions)))
        possible_positions = possible_positions - self.wall_poses
        possible_positions = possible_positions - set(self.key_poses)
        possible_positions = possible_positions - self.get_all_door_poses()

        # do not reset in room with locked door
        room0 = set(tuple(map(tuple, np.mgrid[0:GRID_N, 7:GRID_N].reshape(2, -1).T)))
        rooms = [room0]
        for i in range(len(rooms)):
            possible_positions = possible_positions - rooms[i]
        possible_positions = np.array(list(possible_positions))
        agent_positions = possible_positions[np.random.randint(len(possible_positions), size=self.n_agents)]
        return agent_positions.flatten().astype('uint8')

class KeyCorridorSequential(KeyCorridor):
    '''
    Same as KeyCorridor with 3 keys, but one key is inside a locked room
    '''
    def __init__(self, n_agents):
        super().__init__(n_agents, 3)
        self.name = 'key-corridor-seq'

    def reset_keys_and_doors(self):
        self.key_poses = []
        self.door_poses = []
        self.carrying = [set([]) for _ in range(self.n_agents)]
        all_key_pos = ((15, 0), (1, 14), (7, 14))
        all_door_pos = ([(0, 10), (1, 10)], [(7, 10), (8, 10)], [(13, 10), (14, 10)])
        for key_idx in range(self.n_keys):
            self.key_poses.append(all_key_pos[key_idx])
            self.door_poses.append(all_door_pos[key_idx])
            for door in all_door_pos[key_idx]:
                if door in self.wall_poses:
                    self.wall_poses.remove(door)


def to_list(carrying_lst):
    ret = []
    for i, carrying in enumerate(carrying_lst):
        ret.append(list(carrying))
    return ret


if __name__ == "__main__":
    env = KeyWall(1)
    env.reset()
    import pdb; pdb.set_trace()