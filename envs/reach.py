import copy
import numpy as np

from envs.base_env import BaseEnv
from devices.camera import camera_thread, load_extrinsics, Camera
import multiprocessing as mp

class Reach(BaseEnv):
    """
    In this task the xArm6 robot end-effector reaches a specific target position
    """

    def __init__(self, mode=0, target_src="random", simulated=False, has_gripper=True):
        assert isinstance(target_src, np.ndarray) or target_src in ["random", "camera"], \
               "target_src must be either 'random', 'camera', or a numpy array of shape (3,)"
        super().__init__(mode=mode, simulated=simulated, has_gripper=has_gripper)

        if isinstance(target_src, np.ndarray):
            self.use_camera = False
            self.target_pos = target_src
            print(f"[xArm Reach] Target position is {self.target_pos}")
        elif target_src == "camera":
            self.use_camera = True
            self._setup_camera(target_cube_offset=np.array([0, 0, 0.25]))
            print("[xArm Reach] Target position is 15cm above live cube position from camera")
        elif target_src == "random":
            self.use_camera = False
            # Robosuite limits: [-0.1, -0.1, 0.9], to [0.1, 0.1, 1.1], with robot base at [-0.56,  0, 0.912]
            # So xArm limits: min = [-0.1, -0.1, 0.9] - [-0.56,  0, 0.912] = [0.46, -0.1, -0.012]
            #                 max = [ 0.1,  0.1, 1.1] - [-0.56,  0, 0.912] = [0.66,  0.1,  0.188]
            self.target_pos = np.random.uniform(low=[0.46, -0.1, -0.012], high=[0.66, 0.1, 0.188])
            print(f"[xArm Reach] Target position is {self.target_pos}")
        self._setup_observables()

    def _setup_observables(self):
        super()._setup_observables()

        # Add task related observables
        self.obs.update({
            "target_pos": np.zeros(3),
            "target_to_robot0_eef_pos": np.zeros(3),
        })


    def _setup_camera_thread(self, target_cube_offset=np.array([0, 0, 0.05])):
        self.ext_cam2arm, _ = load_extrinsics()
        self.ext_cam2arm[:3, -1] += target_cube_offset
        # Start cube detection process
        mp.set_start_method('spawn')
        self.cube_position_buffer = mp.Queue()
        self.camera_stop_event = mp.Event()
        self.camera_process = mp.Process(target=camera_thread, args=(self.cube_position_buffer, self.camera_stop_event))
        self.camera_process.start()

    def _setup_camera(self, target_cube_offset=np.array([0, 0, 0.05])):
        self.cam = Camera(debug=True, save_video=True)
        self.cam.cam2arm, self.cam.arm2cam = load_extrinsics()
        self.cam.cam2arm[:3, -1] += target_cube_offset
        self.camera_process = None

    def _get_obs(self):

        # Get xArm related observations from base_env first
        super()._get_obs()

        # Update additional observations from devices and task
        if self.use_camera:
            # self.target_pos = self.ext_cam2arm[:3, :3] @ self.cube_position_buffer.get() + self.ext_cam2arm[:3, -1]
            self.target_pos = self.cam.detect(arm_frame=True)
        self.obs['target_pos'] = self.target_pos
        self.obs['target_to_robot0_eef_pos'] = self.obs['robot0_eef_pos'] - self.target_pos

        return copy.deepcopy(self.obs)


    def _check_success(self):
        eef_pos = self.xarm.get_eef_position()

        if self.use_camera:
            # self.target_pos = self.ext_cam2arm[:3, :3] @ self.cube_position_buffer.get() + self.ext_cam2arm[:3, -1]
            self.target_pos = self.cam.detect(arm_frame=True)

        if np.linalg.norm(eef_pos - self.target_pos) < 0.03:
            return True
        else:
            return False

    def reset(self):
        """
        Resets xArm and get initial observation
        """
        super().reset()
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

        if gripper_action > 0:
            self.xarm.close_gripper()
        else:
            self.xarm.open_gripper()

        done = self._check_success() 
        return self._get_obs(), 0, done, None

    def close(self):
        self.xarm.close()
        
        if self.use_camera and (self.camera_process is not None):
            self.camera_stop_event.set()
            self.camera_process.join()

