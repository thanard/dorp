from envs import GoalEnv
from pyrep.errors import IKError, ConfigurationPathError
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.const import RenderMode

import numpy as np
import time
import pathlib


GROCERY_NAMES = [
    'Cylinder0',
    'Cylinder1',
    'Cylinder2',
    'Cylinder3',
    'Cylinder4',
]

class PushEnv(GoalEnv):
    def __init__(self, n_objects, headless=True):
        super(PushEnv, self).__init__()
        self.name = 'pushenv'

        # Turn on only one camera
        self._pyrep = PyRep()
        cur_dir = pathlib.Path(__file__).parent.absolute()
        self._pyrep.launch(str(cur_dir / 'scene_one_camera_low_poly_arm.ttt'),
                           headless=headless, responsive_ui=True)
        self._pyrep.import_model(str(cur_dir / 'rearrange_objects.ttm'))
        # self._pyrep.set_simulation_timestep(dt=0.1)
        self._pyrep.start()

        self._camera = VisionSensor('cam_front')
        self._camera.set_explicit_handling(1)
        self._camera.set_resolution([64, 64])
        self._camera.set_render_mode(RenderMode.OPENGL)

        self._gripper_mask = Shape('gripper_mask_small')
        self._objects = [Shape(name.replace(' ', '_'))
                          for name in GROCERY_NAMES]
        for obj in range(5-n_objects):
            obj = self._objects.pop(-1)
            obj.remove()

        self._setup_robot()
        self._setup_action_space()
        self._setup_state_space()

    def get_state(self):
        return [g.get_position() for g in self._objects]

    def seed(self, seed):
        np.random.seed(0)

    def get_obs(self):
        self._camera.handle_explicitly()
        return self._camera.capture_rgb()

    def num_actions(self):
        assert len(self.actions) == 324
        return len(self.actions)

    def reset(self):
        # todo: set goals
        # Reset Objects
        for obj in self._objects:
            x, y = None, None
            while x is None or self._check_gripper_ee_collision(x, y, min_dist=self.min_distance):
                x = np.random.uniform(low=self.obj_boundary[0, 0], high=self.obj_boundary[0, 1], size=(1,))
                y = np.random.uniform(low=self.obj_boundary[1, 0], high=self.obj_boundary[1, 1], size=(1,))
            obj.set_pose([x, y, self.obj_height, 0, 0, 0, 1])
        # for i in self.obj_boundary:
        # self._boundary.clear()
        # [self._boundary.sample(g, min_distance=self.min_distance) for g in self._objects]
        return self.get_obs()

    def sample_action(self):
        positions = [None, None]
        quat = None
        idx = None
        while idx is None or self._check_gripper_collision(positions[0], quat) or not self._check_gripper_collision(positions[1], quat):
            idx = np.random.choice(self.num_actions())
            positions = [self.before_ee_positions[idx], self.after_ee_positions[idx]]
            quat = self.fb_quat if idx < self.num_actions()/2 else self.lr_quat
        return idx, self._find_gripper_collision(positions[1], quat)

    def sample_action_efficient(self):
        position = None
        quat = None
        idx = None
        sampled_object = np.random.choice(self._objects)
        x, y, _ = sampled_object.get_position()
        action_idces = self._get_overlap_action_idces(x, y)
        while idx is None or self._check_gripper_collision(position, quat):
            idx = np.random.choice(action_idces)
            position = self.before_ee_positions[idx]
            quat = self._get_quaternion(idx)
            assert self._check_gripper_collision(self.after_ee_positions[idx], quat)
            assert (self.after_ee_positions[idx] - self.before_ee_positions[idx]).abs().sum() == self.step_size
        return idx

    def _get_quaternion(self, action_idx, backup=False):
        if not backup:
            return self.fb_quat if action_idx < self.num_actions()/2 else self.lr_quat
        else:
            return self.fb_quat_backup if action_idx < self.num_actions()/2 else self.lr_quat_backup

    def step(self, action_idx):
        """
        :param action:
        :return:
        """
        # Before push
        before_joint = self.before_joints[action_idx]
        self.arm.set_joint_positions(before_joint)
        # self._pyrep.step_ui()

        # Push
        # after_joint = self.after_joints[action_idx]
        # linear_path = np.linspace(before_joint, after_joint, num=2).reshape(-1)
        # path = ArmConfigurationPath(self._robot.arm,
        #                             linear_path)
        try:
            path = self.arm.get_linear_path(
                self.after_ee_positions[action_idx],
                quaternion=self._get_quaternion(action_idx),
                ignore_collisions=True,
                steps=50
            )
        except ConfigurationPathError:
            print("Try backup quaternion")
            before_position = self.before_ee_positions[action_idx]
            new_quat = self._get_quaternion(action_idx, backup=True)
            before_joint = self._get_joint_from_pos(before_position, new_quat)
            self.arm.set_joint_positions(before_joint)
            # self._pyrep.step_ui()
            try:
                after_position = self.after_ee_positions[action_idx]
                path = self.arm.get_linear_path(
                    after_position,
                    quaternion=new_quat,
                    ignore_collisions=True,
                    steps=50
                )
            except ConfigurationPathError:
                print("return none")
                return None
        # import time
        # start_start = time.time()
        done = False
        while not done:
            # start = time.time()
            done = path.step()
            # print(" ----> path steps: ", time.time() - start)
            # start = time.time()
            self._pyrep.step()
            # print(" ----> pyrep steps: ", time.time() - start)
        # print(" --> inner steps: ", time.time() - start_start)

        # After push
        self.arm.set_joint_positions(self._original_joints)
        # self._pyrep.step_ui()
        return self.get_obs()

    def reached_goal(self):
        return

    def close(self):
        self._pyrep.stop()
        self._pyrep.shutdown()

    def _get_overlap_action_idces(self, x, y):
        """Get overlap indices from object position.
        Assume that object size is twice step_size.

        :param x:
        :param y:
        :return:
        """
        x_idx = int((- self.action_boundary[0, 0] + x) // self.step_size) - 1
        y_idx = int((- self.action_boundary[0, 1] + y) // self.step_size) - 1
        assert all((0 <= x_idx, y_idx < 8))
        action_idces = np.array([
            y_idx * 9 + x_idx,
            (y_idx+1)*9 + x_idx,
            y_idx * 9 + x_idx + 1,
            (y_idx+1)*9 + x_idx])
        action_idces.repeat(4).reshape((4, 4)) + np.arange(4) * 81
        return action_idces.reshape(-1)

    def _find_gripper_collision(self, position, quat):
        self._gripper_mask.set_position(position)
        self._gripper_mask.set_quaternion(quat)
        for i, obj in enumerate(self._objects):
            if self._gripper_mask.check_collision(obj):
                return i
        return None

    def _check_gripper_collision(self, position, quat):
        self._gripper_mask.set_position(position)
        self._gripper_mask.set_quaternion(quat)
        for obj in self._objects:
            if self._gripper_mask.check_collision(obj):
                return True
        return False

    def _check_gripper_ee_collision(self, ee_x, ee_y, min_dist=0.04):
        for obj in self._objects:
            # min_x, max_x, min_y, max_y, _, _ = obj.get_bounding_box()
            obj_x, obj_y, _ = obj.get_position()
            if (obj_x - ee_x)**2 + (obj_y - ee_y)**2 < min_dist**2:
            # if obj_x + min_x < ee_x and ee_x < obj_x + max_x and \
            #     obj_y + min_y < ee_y and ee_y < obj_y + max_y:
                return True
        return False

    def _get_joint_from_pos(self, pos, quat):
        try:
            return self.arm.solve_ik_via_jacobian(
                pos, quaternion = quat
            )
        except IKError:
            return self.arm.solve_ik_via_sampling(
                pos, quaternion = quat, ignore_collisions=True
            )[0]

    def _vectorized_get_joint_from_pos(self, poses, quat):
        output = np.zeros((len(poses), 7)) # number of joints
        for i, pos in enumerate(poses):
            output[i] = self._get_joint_from_pos(pos, quat)
        return output

    def _visualize_actions(self):
        for joint in self.fb_joints:
            self.arm.set_joint_positions(joint)
            self._pyrep.step_ui()
            time.sleep(1)

    def _setup_robot(self):
        self.arm = Panda()
        # arm.max_velocity = 4.0
        self.arm.set_control_loop_enabled(True)
        self._original_joints = self.arm.get_joint_positions()

    def _setup_state_space(self):
        self.obj_boundary = np.array([[0, 0.30], [-0.15, 0.15]])
        self.min_distance = 0.101
        self.obj_height = 0.802

    def _setup_action_space(self):
        self.gripper_height = 0.81
        self.step_size = 0.05
        self.action_boundary = np.array(((-0.1, -0.25), (0.4, 0.25)))

        # Possible EE for forward and backward pushes
        fb_x, fb_y = np.meshgrid(
            np.linspace(-0.1, 0.40, 11), # xlim
            np.linspace(-0.2, 0.2, 9),  # ylim
            indexing='ij'
        )
        self.fb_quat = (0, 1, 0, 0)
        self.fb_quat_backup = (-1, 0, 0, 0)
        self.fb_positions = np.stack([fb_x, fb_y, np.ones((11, 9))*self.gripper_height],
                          axis=-1).reshape((-1, 3))
        self.fb_joints = self._vectorized_get_joint_from_pos(self.fb_positions, self.fb_quat).reshape((11, 9, 7))
        self.fb_positions = self.fb_positions.reshape((11, 9, 3))

        # Possible EE for left and right pushes
        lr_x, lr_y = np.meshgrid(
            np.linspace(-0.05, 0.35, 9), # xlim
            np.linspace(-0.25, 0.25, 11),  # ylim
            indexing='ij'
        )
        self.lr_quat = (np.sqrt(0.5), np.sqrt(0.5), 0, 0)
        self.lr_quat_backup = (-np.sqrt(0.5), np.sqrt(0.5), 0, 0)
        self.lr_positions = np.stack([lr_x, lr_y, np.ones((9, 11))*self.gripper_height],
                            axis=-1).reshape((-1, 3))
        self.lr_joints = self._vectorized_get_joint_from_pos(self.lr_positions, self.lr_quat).reshape((9, 11, 7))
        self.lr_positions = self.lr_positions.reshape((9, 11, 3))

        # All possible joint actions (9x9x4) -- order: x_dec, x_inc, y_dec, y_inc
        self.before_ee_positions = np.concatenate([
            self.fb_positions[2:].reshape((-1, 3)),
            self.fb_positions[:-2].reshape((-1, 3)),
            self.lr_positions[:, 2:].reshape((-1, 3)),
            self.lr_positions[:, :-2].reshape((-1, 3))
        ], axis=0)
        self.before_joints = np.concatenate([
            self.fb_joints[2:].reshape((-1, 7)),
            self.fb_joints[:-2].reshape((-1, 7)),
            self.lr_joints[:, 2:].reshape((-1, 7)),
            self.lr_joints[:, :-2].reshape((-1, 7))
        ], axis=0)
        self.after_ee_positions = np.concatenate([
            self.fb_positions[1:-1].reshape((-1, 3)),
            self.fb_positions[1:-1].reshape((-1, 3)),
            self.lr_positions[:, 1:-1].reshape((-1, 3)),
            self.lr_positions[:, 1:-1].reshape((-1, 3))
        ], axis=0)
        self.after_joints = np.concatenate([
            self.fb_joints[1:-1].reshape((-1, 7)),
            self.fb_joints[1:-1].reshape((-1, 7)),
            self.lr_joints[:, 1:-1].reshape((-1, 7)),
            self.lr_joints[:, 1:-1].reshape((-1, 7))
        ], axis=0)
        self.actions = np.concatenate([
            self.before_joints, self.after_joints
        ], axis=1)
