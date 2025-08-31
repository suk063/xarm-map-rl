import copy
import numpy as np

from envs.base_env import BaseEnv
from devices.camera import Camera

class VisualEnv(BaseEnv):
    """
    In this task the xArm6 robot end-effector performs object manipulation using camera input directly.
    It can be used for tasks like reaching, pick and place, etc.
    """

    def __init__(self, mode=0, simulated=False, has_gripper=True, imsize=(128,128), **kwargs):
        super().__init__(mode=mode, simulated=simulated, has_gripper=has_gripper, **kwargs)

        self.target_offset = np.array([0, 0, 0.15])
        self.xarm_init_position = np.array([2.09461150e-01, -1.24141867e-06,  1.44394929e-01])
        self.imsize = imsize

        self.snap_gripper = False
        if 'snap_gripper' in kwargs:
            self.snap_gripper = kwargs['snap_gripper']

        self.cam_kwargs = {}
        if 'cam_kwargs' in kwargs:
            self.cam_kwargs = kwargs['cam_kwargs']
        self._setup_camera()
        self._setup_observables()

    def _setup_observables(self):
        super()._setup_observables()

        # Add task related observables
        self.obs.update({
            "rgb": np.zeros((*self.imsize, 3), dtype=np.uint8),
            "depth_image": np.zeros(self.imsize, dtype=np.float32),
        })

    def _setup_camera(self):
        self.cam = Camera(debug=True, save_video=True, imsize=self.imsize)
        self.cam.flush()

    def _get_obs(self):

        # Get xArm related observations from base_env first
        super()._get_obs()

        # Get camera observations that are visually similar to robosuite images
        color_image, _ = self.cam.fetch_image(**self.cam_kwargs)

        self.obs["rgb"] = color_image
        # self.obs["depth_image"] = depth_image
        
        # Robosuite seems to always have quaternions with w < 0, so we ensure that here
        quat = self.obs["robot0_eef_quat"]
        if quat[0] > 0:
            self.obs["robot0_eef_quat"] = -1 * quat
        
        # Convert eef position from real robot base frame to robosuite origin frame
        real_eef_pos = self.obs["robot0_eef_pos"]
        robosuite_to_real_origin = np.array([-0.3, 0, 0.8])
        self.obs["robot0_eef_pos"] = real_eef_pos + robosuite_to_real_origin

        return copy.deepcopy(self.obs)

    def _check_success(self):
        return False
        # eef_pos = self.xarm.get_eef_position()

        # if self.use_camera:
        #     # self.target_pos = self.ext_cam2arm[:3, :3] @ self.cube_position_buffer.get() + self.ext_cam2arm[:3, -1]
        #     self.target_pos = self.cam.detect(arm_frame=True) + self.target_offset

        # if np.linalg.norm(eef_pos - self.target_pos) < 0.03:
        #     return True
        # else:
        #     return False

    def reset(self):
        """
        Resets xArm and get initial observation
        """
        super().reset(custom_init_position=self.xarm_init_position)
        return self._get_obs()

    def step(self, action, duration=None):
        gripper_action = None
        if self.mode == 0:
            # End-effector position control
            # action is defined as the relative eef movement 
            # action = 1 for x, y, z <==> displacement = 1 cm 
            # action = 1 for roll, pitch, yaw <==> rotation = 0.1 rad ~ 5.7 degrees

            # 6 DOF for position + orientation and 1 for gripper
            assert len(action) == 7
            x, y, z, roll, pitch, yaw = action[:6]
            x, y, z = x / 100., y / 100., z / 100.
            roll, pitch, yaw = roll / 10, pitch / 10, yaw / 10
            gripper_action = action[-1]

            self.xarm.set_eef_pose(x, y, z, roll, pitch, yaw, relative=True)

        elif self.mode == 4:
            # Joint velocity control

            # 6 DOF for joint velocity and 1 for gripper
            assert len(action) == 7
            joint_vel = action[:6]
            gripper_action = action[-1]

            if duration is None:
                duration = 0.05
            self.xarm.set_joint_velocity(joint_vel, duration=duration)

        # if gripper_action > 0:
        #     self.xarm.close_gripper()
        # else:
        #     self.xarm.open_gripper()

        if self.snap_gripper:
            if gripper_action > 0:
                gripper_action = 1.0
            else:
                gripper_action = -1.0
        # Make sure gripper_action is in [-1, 1]
        self.xarm.set_gripper_pos(gripper_action)

        done = self._check_success() 
        return self._get_obs(), 0, done, None

    def close(self):
        self.cam.close()
        self.xarm.close()


class VisualStateEnv(VisualEnv):
    """
    VisualStateEnv is a subclass of VisualEnv that uses environment state information along with visual observations.
    """

    def __init__(self, mode=0, simulated=False, has_gripper=True, imsize=(128,128), object_to_grip="red_cube_40mm", **kwargs):
        super().__init__(mode=mode, simulated=simulated, has_gripper=has_gripper, imsize=imsize, object_to_grip=object_to_grip, **kwargs)
        # Target position should be in robot base frame for correct relative calculation in object_to_target_pos.
        # Setting it to be similar to [0, 0, 0.15] in Maniskill.
        self.target_pos = np.array([0.52, 0, 0.15])
        self.object_color = "r" if "red" in object_to_grip else "g"

    def _setup_observables(self):
        super()._setup_observables()

        # Add task related observables
        self.obs.update({
            "cube_pos": np.zeros(3),
            "robot0_eef_pos_to_cube": np.zeros(3),
            "object_to_target_pos": np.zeros(3),
            "is_grasped": np.zeros(1, dtype=np.float32),
        })

    def _get_obs(self):
        super()._get_obs()

        cube_pos = self.cam.detect(arm_frame=True, color=self.object_color)
        if cube_pos is not None:
            self.obs['cube_pos'] = cube_pos
        self.obs['robot0_eef_pos_to_cube'] = self.obs['cube_pos'] - self.obs['robot0_eef_pos']
        self.obs['object_to_target_pos'] = self.target_pos - self.obs["cube_pos"]

        self.is_grasped = (np.linalg.norm(self.obs['robot0_eef_pos_to_cube']) < 0.05) and self.obs["robot0_is_gripper_closed"][0]
        self.obs["is_grasped"] = np.array([int(self.is_grasped)])

        return copy.deepcopy(self.obs)

    def _check_success(self):
        # return False
        eef_pos = self.xarm.get_eef_position()

        if np.linalg.norm(eef_pos - self.target_pos) < 0.03 and self.is_grasped:
            return True
        else:
            return False
        
    def reset(self):
        """
        Resets xArm and get initial observation
        """
        self.xarm.reset()
        return self._get_obs()
