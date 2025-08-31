import copy
import time
import threading
import multiprocessing as mp

import numpy as np
import robosuite.utils.transform_utils as T

from envs.base_env import BaseEnv
from devices.camera import camera_thread, load_extrinsics, Camera
from devices.fsr import fsr_thread


class PickPlace(BaseEnv):
    """
    In this task the xArm6 robot end-effector reaches for a green cube, 
    closes gripper to grasp and lift it, moves to a position above a bin, 
    and places the cube in it.
    """

    def __init__(self, mode=0, simulated=False, has_gripper=True, use_cam_thread=False, use_pose=False):

        super().__init__(mode=mode, simulated=simulated, has_gripper=has_gripper)

        self.cam2arm, _ = load_extrinsics("extrinsics.npz")

        # print(self.cam2arm.shape, self.cam2arm)

        self.use_cam_thread = use_cam_thread
        self.use_pose = use_pose
        self._setup_observables()
        
        self.curr_stage = 1
        # self.bin_size = (0.39, 0.49)                                    # Same as robosuite (check env.env.bin_size)
        self.bin_size = (0.15, 0.15)
        self.grasped = False

    def _setup_observables(self):
        super()._setup_observables()

        # Add task related observables
        self.obs.update({
            "cube_pos": np.zeros(3),                                    # [0.1, -0.25, 0.845] in robosuite, [0.6, -0.15, -0.067] in real
            # "bin_pos": np.array([0.6975, 0.2575, -0.112]),              # [0.1975, 0.1575, 0.8] in robosuite, [0.6975, 0.2575, -0.112] in real
            "bin_pos": np.array([0.4545, 0.1575, -0.183]),              # [0.0475, 0.1575, 0.90] in robosuite, [0.4545, 0.1575, -0.183] in real
            "cube_quat": np.array([0., 0., 0., 1.]),                    # xyzw
            "cube_to_robot0_eef_pos": np.zeros(3),
            "cube_to_robot0_eef_quat": np.array([0., 0., 0., 1.]),      # xyzw
            "cube_to_bin_pos": np.zeros(3),
            "bin_to_robot0_eef_pos": np.zeros(3),
            "robot0_touch": np.zeros(2),
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
            cube_pos = self.cam.detect(arm_frame=True)
            if cube_pos is not None:
                self.obs['cube_pos'] = cube_pos
            if self.use_pose:
                self.obs['cube_quat'] = self.cam.estimate_pose()

        world_pose_in_gripper = T.pose_inv(T.pose2mat((self.obs["robot0_eef_pos"], self.obs["robot0_eef_quat"])))
        # cube_pose = T.pose2mat((self.obs["cube_pos"], self.obs["cube_quat"]))
        # rel_pose = T.pose_in_A_to_pose_in_B(cube_pose, world_pose_in_gripper)
        # rel_pos, rel_quat = T.mat2pose(rel_pose)
        self.obs['cube_to_robot0_eef_quat'] = T.quat_distance(self.obs['cube_quat'], self.obs['robot0_eef_quat'])  # This one line calculates the same quat as rel_quat above

        self.obs['cube_to_robot0_eef_pos'] = self.obs['robot0_eef_pos'] - self.obs['cube_pos']
        self.obs['cube_to_bin_pos'] = self.obs['bin_pos'] - self.obs['cube_pos']

        is_grasped = (np.linalg.norm(self.obs['cube_to_robot0_eef_pos']) < 0.05) and self.obs["robot0_is_gripper_closed"][0]
        self.obs["is_grasped"] = np.array([int(is_grasped)])

        bin_pos = self.obs['bin_pos']
        bin_quat = np.array([0, 0, 0, 1], dtype=np.float64)
        bin_pose = T.pose2mat((bin_pos, bin_quat))
        rel_pose = T.pose_in_A_to_pose_in_B(bin_pose, world_pose_in_gripper)
        rel_pos, rel_quat = T.mat2pose(rel_pose)
        self.obs['bin_to_robot0_eef_pos'] = rel_pos

        # TODO: Check why this does not work
        # self.obs['bin_to_robot0_eef_pos'] = self.obs['robot0_eef_pos'] - self.obs['bin_pos']

        if not self.fsr_buffer.empty():
            self.obs['robot0_touch'] = np.array(self.fsr_buffer.get(timeout=1))
        
        # Print if grasped
        if not self.grasped and (self.obs['robot0_touch'] > 0.9).all():
            self.grasped = True
            print("Cube grasped!")
        elif self.grasped and (self.obs['robot0_touch'] < 0.1).all():
            self.grasped = False
            print("Cube released!")
            
        return copy.deepcopy(self.obs)

    def _check_success(self):
        """
        Success if has touch signal and end-effector above some height
        """
        touch_obs = self.obs['robot0_touch']

        if self.curr_stage == 3:
            # Check if arm has been lifted
            dist = np.linalg.norm(self.obs['bin_to_robot0_eef_pos']) 
            if (touch_obs < 0.1).all() and np.tanh(10.0 * dist) > 0.4:
                return True
    
    def cube_over_bin(self, cube_pos=None, bin_pos=None):
        if cube_pos is None:
            cube_pos = self.obs['cube_pos']
        if bin_pos is None:
            bin_pos = self.obs['bin_pos']

        return (cube_pos[0] > bin_pos[0] - self.bin_size[0]/2) and (cube_pos[0] < bin_pos[0] + self.bin_size[0]/2) and \
               (cube_pos[1] > bin_pos[1] - self.bin_size[1]/2) and (cube_pos[1] < bin_pos[1] + self.bin_size[1]/2) and \
               (cube_pos[2] > bin_pos[2]) #and cube_pos[2] < bin_pos[2] + 0.1
        #     print(f"Cube in bin: cube: {cube_pos}, bin: {bin_pos}, diff: {cube_pos - bin_pos}")
        # else:
        #     print(f"Cube NOT in bin: cube: {cube_pos}, bin: {bin_pos}, diff: {cube_pos - bin_pos}")

    def reset(self):
        """
        Resets xArm and get initial observation
        """
        super().reset()
        return self._get_obs()

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

        obs = self._get_obs()
        done = self._check_success() 

        # Stage 1: Reach cube
        if self.curr_stage == 1 and np.linalg.norm(obs['cube_to_robot0_eef_pos']) < 0.06:
            print("\nCube reached!\n")
            self.curr_stage += 1
        # Stage 2: Lift cube and bring it to bin
        if self.curr_stage == 2 and self.cube_over_bin():
            print("\nCube over target bin!\n")
            self.curr_stage += 1
        # Stage 3: Place cube in bin and lift arm

        return obs, 0, done, None

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