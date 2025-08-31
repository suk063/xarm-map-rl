import copy
import time
import threading
import multiprocessing as mp

import numpy as np

from envs.base_env import BaseEnv
from devices.camera import camera_thread, load_extrinsics, Camera
from devices.fsr import fsr_thread


class Stack(BaseEnv):
    """
    In this task the xArm6 robot end-effector reaches for a red cube, 
    closes gripper to grasp and lift it, moves to a position above a green cube, 
    and places the cube on top of it.
    """

    def __init__(self, mode=0, simulated=False, has_gripper=True):

        super().__init__(mode=mode, simulated=simulated, has_gripper=has_gripper, object_to_grip="red_box_raspi")

        self.cam2arm, _ = load_extrinsics("extrinsics.npz")

        # print(self.cam2arm.shape, self.cam2arm)

        self._setup_observables()
        
        self.curr_stage = 1
        self.tumble_check = False
        self.cubeB_size = (0.055, 0.055)

    def _setup_observables(self):
        super()._setup_observables()

        # Add task related observables
        self.obs.update({
            "cubeA_pos": np.zeros(3),                                   # [] in robosuite, [] in real
            "cubeB_pos": np.zeros(3),                                   # [] in robosuite, [] in real
            "robot0_eef_to_cubeA_pos": np.zeros(3),
            "robot0_eef_to_cubeB_pos": np.zeros(3),
            "cubeA_to_cubeB_pos": np.zeros(3),
            "robot0_touch": np.zeros(2),
        })

        # Initialize camera and touch sensors
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
        
        red_pos = self.cam.detect(arm_frame=True, color='r')
        green_pos = self.cam.detect(arm_frame=True, color="g")
        if red_pos is not None:
            self.obs['cubeA_pos'] = red_pos
        if green_pos is not None:
            self.obs['cubeB_pos'] = green_pos

        self.obs['robot0_eef_to_cubeA_pos'] = self.obs['cubeA_pos'] - self.obs['robot0_eef_pos']
        self.obs['robot0_eef_to_cubeB_pos'] = self.obs['cubeB_pos'] - self.obs['robot0_eef_pos']
        self.obs['cubeA_to_cubeB_pos'] = self.obs['cubeB_pos'] - self.obs['cubeA_pos']

        if not self.fsr_buffer.empty():
            self.obs['robot0_touch'] = np.array(self.fsr_buffer.get(timeout=1))

        return copy.deepcopy(self.obs)

    def _check_success(self):
        """
        Success if cubeA is on top of cubeB and arm is not grasping
        """
        if self.tumble_check:
            if self.wait_steps == 0:
                self.tumble_check = False
                return self.cubeA_over_cubeB()
            else:
                self.wait_steps -= 1
                return False

        if self.curr_stage == 3 and (self.obs['robot0_touch'] < 0.1).all():
            self.tumble_check = True
            self.wait_steps = 3

        return False
    
    def cubeA_over_cubeB(self, cubeA_pos=None, cubeB_pos=None):
        if cubeA_pos is None:
            cubeA_pos = self.obs['cubeA_pos']
        if cubeB_pos is None:
            cubeB_pos = self.obs['cubeB_pos']

        horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
        return horiz_dist < 0.055 and cubeA_pos[2] > cubeB_pos[2] + 0.02
        #     print(f"CubeA over CubeB: cubeA: {cubeA_pos}, cubeB: {cubeB_pos}, diff: {cubeA_pos - cubeB_pos}")
        # else:
        #     print(f"Cube NOT over CubeB: cubeA: {cubeA_pos}, cubeB: {cubeB_pos}, diff: {cubeA_pos - cubeB_pos}")

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

        self.regrasp_attempt = False
        # Stage 1: Reach cubeA
        if self.curr_stage == 1 and np.linalg.norm(obs['robot0_eef_to_cubeA_pos']) < 0.05:
            print("\nCubeA reached!\n")
            self.curr_stage += 1
        # Stage 2: Lift cubeA and bring it over cubeB
        if self.curr_stage == 2 and self.cubeA_over_cubeB():
            print("\nCubeA over CubeB!\n")
            self.curr_stage += 1
        # Stage 3: Stack cubeA on cubeB
        if self.curr_stage == 3 and not self.cubeA_over_cubeB():
            print("\nCubeA NOT over CubeB, retrying!\n")
            self.curr_stage = 1
            self.regrasp_attempt = True

        return obs, 0, done, None

    def close(self):
        print("Closing Lift Env")
        self.cam.close()
        
        self.fsr_stop_event.set()
        self.fsr_process.join()

    def __del__(self):
        self.close()
        super().__del__()