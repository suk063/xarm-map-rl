import copy 
import numpy as np
from devices.xarm6 import XArmControl

class BaseEnv:
    """
    Base environment for xArm6, only set up xArm6 control
    Sets up xArm6 control, devices like camera/touch sensors should be set up in each env
    """
    def __init__(self, mode=0, simulated=False, has_gripper=True, object_to_grip="red_cube_40mm", **kwargs):

        self.has_gripper = has_gripper
        if has_gripper:
            tcp_z_offset = 145
        else:
            tcp_z_offset = 0

        self.mode = mode
        self.xarm = XArmControl(
            ip="192.168.1.242", 
            mode=mode, 
            simulated=simulated,
            tcp_z_offset=tcp_z_offset,
            object_to_grip=object_to_grip
        )


    def _setup_observables(self):
        """
        Observables for xArm6 only
        """
        self.obs = {
            "robot0_joint_pos": np.zeros(6),
            "robot0_joint_pos_cos": np.zeros(6),
            "robot0_joint_pos_sin": np.zeros(6),
            "robot0_joint_vel": np.zeros(6),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0., 0., 0., 1.]),
        }

        if self.has_gripper:
            self.obs.update({
                "robot0_gripper_width": np.zeros(1),
                "is_gripper_closed": np.zeros(1),
            })

    def _get_obs(self):
        qpos, qvel = self.xarm.get_qpos_qvel()
        self.obs['robot0_joint_pos'] = qpos
        self.obs['robot0_joint_pos_cos'] = np.cos(qpos)
        self.obs['robot0_joint_pos_sin'] = np.sin(qpos)
        self.obs['robot0_joint_vel'] = qvel

        self.obs['robot0_eef_pos'] = self.xarm.get_eef_position()
        self.obs['robot0_eef_quat'] = self.xarm.get_eef_quaternion()

        if self.has_gripper:
            self.obs["robot0_gripper_width"] = self.xarm.get_gripper_qpos()
            self.obs["robot0_is_gripper_closed"] = self.xarm.is_gripper_closed()

        return copy.deepcopy(self.obs)

    @property
    def observation_dim(self):
        dim = 0
        for k, v in self.obs.items():
            if isinstance(v, float):
                dim += 1 
            else:
                dim += len(v)
        return dim

    @property
    def observation_names(self):
        return list(self.obs.keys())

    @property
    def action_dim(self):
        if self.mode == 0:
            action_dim = 6      # End-effector control, x, y, z, roll, pitch, yaw
        elif self.mode == 4:
            action_dim = 6      # Joint velocity control
        else:
            raise ValueError(f"xArm mode {self.mode} not end-effector or joint velocity")

        if self.has_gripper:
            action_dim += 1
        return action_dim
    
    def switch_mode(self, mode):
        self.xarm.switch_mode(mode)
        self.mode = mode

    def reset(self, custom_init_position=None):
        """
        Resets xArm and get initial observation
        """
        self.xarm.reset(custom_init_position=custom_init_position)
        return self._get_obs()

    def __del__(self):
        # Ensure we don't raise if initialization failed and xarm doesn't exist
        try:
            print("Resetting xArm, please wait")
            if hasattr(self, "xarm") and self.xarm is not None:
                self.xarm.close()
        except Exception:
            # Suppress all exceptions in destructor to avoid noisy teardown
            pass