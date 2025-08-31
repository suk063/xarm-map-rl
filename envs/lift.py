import copy
import time
import threading
import multiprocessing as mp

import numpy as np
import robosuite.utils.transform_utils as T

from envs.base_env import BaseEnv
from devices.camera import camera_thread, load_extrinsics, Camera
from devices.fsr import fsr_thread


class Lift(BaseEnv):
    """
    In this task the xArm6 robot end-effector reaches for a green cube, 
    close gripper to grasp and lift it above a threshold height.
    """

    def __init__(self, mode=0, simulated=False, has_gripper=True, use_cam_thread=False, use_pose=False, object_to_grip="red_cube_40mm"):

        super().__init__(mode=mode, simulated=simulated, has_gripper=has_gripper, object_to_grip=object_to_grip)

        self.cam2arm, _ = load_extrinsics("extrinsics.npz")

        # print(self.cam2arm.shape, self.cam2arm)

        self.use_cam_thread = use_cam_thread
        self.use_pose = use_pose
        self._setup_observables()
        
        # self.obs["target_pos"] = np.array([ 0.56, 0,  0.038])  # np.array([0, 0, 0.95]) - np.array([-0.56, 0, 0.912])
        self.obs["target_pos"] = np.array([0.407, 0., -0.08])  # np.array([0, 0, 1]) - np.array([-0.407, 0, 1.083])

        if object_to_grip.startswith("green"):
            self.object_color = "g"
        elif object_to_grip.startswith("red"):
            self.object_color = "r"

    def _setup_observables(self):
        super()._setup_observables()

        # Add task related observables
        self.obs.update({
            "cube_pos": np.zeros(3),                                    # np.array([0.56, 0, -0.095]),
            "cube_quat": np.array([0., 0., 0., 1.]),                    # xyzw
            "cube_to_robot0_eef_pos": np.zeros(3),
            "cube_to_robot0_eef_quat": np.array([0., 0., 0., 1.]),      # xyzw
            "robot0_touch": np.zeros(2),
            "target_pos": np.zeros(3),                                  # [0., 0., 0.1285] above cube
        })

        # Initialize camera and touch sensors
        if self.use_cam_thread:
            self.camera_buffer_position = mp.Queue(1)
            self.camera_stop_event = mp.Event()
            self.camera_process = mp.Process(target=camera_thread, args=(self.camera_buffer_position, self.camera_stop_event))
            self.camera_process.start()
            time.sleep(5)
        else:
            self.cam = Camera(debug=True, save_video=True)
            self.cam.flush()
            self.cam.cam2arm, self.cam.arm2cam = load_extrinsics()
            
        print("Camera initialized")

        self.fsr_buffer = mp.Queue(1)
        self.fsr_stop_event = mp.Event()
        self.fsr_process = mp.Process(target=fsr_thread, args=(self.fsr_buffer, self.fsr_stop_event))
        self.fsr_process.start()

        # Wait for processes to start
        time.sleep(5)

        print("Fsr initialized")

    def _get_obs(self):

        # Get xArm related observations from base_env first
        super()._get_obs()

        if self.use_cam_thread and (not self.camera_buffer_position.empty()):
            # convert cube position from camera frame to robot frame
            cube_pos_cam = np.array(self.camera_buffer_position.get())
            self.obs['cube_pos'] = self.cam2arm[:3, :3] @ cube_pos_cam + self.cam2arm[:3, -1]

        elif not self.use_cam_thread:
            cube_pos = self.cam.detect(arm_frame=True, color=self.object_color)
            if cube_pos is not None:
                self.obs['cube_pos'] = cube_pos
            if self.use_pose:
                self.obs['cube_quat'] = self.cam.estimate_pose()

        self.obs['cube_to_robot0_eef_pos'] = self.obs['robot0_eef_pos'] - self.obs['cube_pos']
        self.obs["cube_to_target_pos"] = self.obs["target_pos"] - self.obs["cube_pos"]
        is_grasped = (np.linalg.norm(self.obs['cube_to_robot0_eef_pos']) < 0.05) and self.obs["robot0_is_gripper_closed"][0]
        self.obs["is_grasped"] = np.array([int(is_grasped)])

        # world_pose_in_gripper = T.pose_inv(T.pose2mat((self.obs["robot0_eef_pos"], self.obs["robot0_eef_quat"])))
        # cube_pose = T.pose2mat((self.obs["cube_pos"], self.obs["cube_quat"]))
        # rel_pose = T.pose_in_A_to_pose_in_B(cube_pose, world_pose_in_gripper)
        # rel_pos, rel_quat = T.mat2pose(rel_pose)
        self.obs['cube_to_robot0_eef_quat'] = T.quat_distance(self.obs['cube_quat'], self.obs['robot0_eef_quat'])  # This one line calculates the same quat as rel_quat above

        if not self.fsr_buffer.empty():
            self.obs['robot0_touch'] = np.array(self.fsr_buffer.get(timeout=1))

        return copy.deepcopy(self.obs)

    def _check_success(self):
        """
        Success if has touch signal and end-effector above some height
        """
        touch_obs = self.obs['robot0_touch']
        cube_height = self.obs['cube_pos'][2]

        return (touch_obs > 0.9).all() and cube_height - self.init_cube_height > 0.05
        
    def reset(self):
        """
        Resets xArm and get initial observation
        """
        super().reset()
        init_obs = self._get_obs()
        self.init_cube_height = init_obs['cube_pos'][2]
        return init_obs

    def step(self, action, speed=80):
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

            # print(x, y, z)
            self.xarm.set_eef_pose(x, y, z, roll, pitch, yaw, relative=True, speed=speed)

        elif self.mode == 4:
            # Joint velocity control

            # 6 DOF for joint velocity and 1 for gripper
            assert len(action) == 7
            joint_vel = action[:6]
            gripper_action = action[-1]

            self.xarm.set_joint_velocity(joint_vel, duration=0.05)
        
        # self.xarm.set_gripper_pos(gripper_action)
        if gripper_action > 0:
            self.xarm.close_gripper()
        else:
            self.xarm.open_gripper()

        done = self._check_success() 
        return self._get_obs(), 0, done, None

    def close(self):
        print("Closing Lift Env")
        if self.use_cam_thread:
            self.camera_stop_event.set()
            self.camera_process.join()
        else:
            self.cam.close()
        
        self.fsr_stop_event.set()
        self.fsr_process.join()

    def __del__(self):
        self.close()
        super().__del__()