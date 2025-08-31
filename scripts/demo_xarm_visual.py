import torch
import numpy as np
from envs.visual_env import VisualEnv, VisualStateEnv
from policies.language_map.visual_ppo import load_policy
import cv2
import os
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
video_dir = os.path.join(parent_dir, "scripts/outputs/video_logs", time.strftime("%Y-%m-%d"), "VisualTask")

def visual_task_demo_simba(simulated=True, task_name="ReachCan", print_info=True, max_steps=2000, save_obs=False):
    from policies.realxarm6.deploy import SimbaModel

    imshape = (128, 128)

    model = SimbaModel()
    valid_tasks =  ["ReachCube", "ReachCan", "LiftCube", "LiftCan", "PlaceCube", "PlaceCan", "StackCube", "StackCan"]
    assert task_name in valid_tasks, f"Task must be one of {valid_tasks}, got {task_name}"     
    if "Reach" in task_name:
        object_to_grip = "none" 
    elif "Can" in task_name:
        object_to_grip = "green_can" 
    elif "Cube" in task_name:
        object_to_grip = "red_cube_40mm"

    model.init_model(task=task_name)

    cam_kwargs = dict(middle_crop_imsize=(480, 480), border_crops=(160, 100, 0, 100))
    env = VisualEnv(mode=4, simulated=simulated, has_gripper=True, imsize=imshape, cam_kwargs=cam_kwargs, 
                    object_to_grip=object_to_grip, snap_gripper=True)
    obs = env.reset()
    done = False

    obs_list = []
    time_list = []
    actions = []
    start_time = time.time()

    for t in range(max_steps):
        if not done:
            obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],\
                                        obs['robot0_joint_pos_sin'],\
                                        obs['robot0_joint_vel'],\
                                        obs['robot0_eef_pos'],\
                                        obs['robot0_eef_quat'],\
                                    ])
            model_observation = {"state": obs_pol, "rgb": obs['rgb']}
            action = model.sample_action(model_observation)
            action[:6] *= 0.5

            if save_obs:
                curr_time = time.time() - start_time
                time_list.append(curr_time)
                obs_list.append(obs_pol)
                actions.append(action)

                if not os.path.exists("outputs/compare_sim_real"):
                    os.makedirs("outputs/compare_sim_real/")
                np.savez(f"outputs/compare_sim_real/obs_xarm_{task_name.lower()}visual.npz", obs_list=obs_list, time_list=time_list, actions=actions)
            
            obs, rew, done, info = env.step(action)

            # time.sleep(1)

    env.reset()
    env.close()

    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_xarm_{task_name.lower()}visual.npz")

    return done


def visual_task_demo_lmap(simulated=True, print_info=True, max_steps=2000):
    imshape = (128, 128)

    policy, obs_rms = load_policy(parent_dir + "/policies/language_map/ckpt_3501.pt",
                              obs_spec={"rgb": (128,128,3),
                                        "state": 13},
                              action_dim=7,
                              device="cuda")

    @torch.no_grad()
    def sample_action(robot_obs):
        st = torch.tensor(robot_obs["state"][None], dtype=torch.float32, device="cuda")
        rgb = torch.tensor(robot_obs["rgb"][None],  dtype=torch.uint8,  device="cuda")
        if obs_rms is not None:
            st = (st - obs_rms["mean"]) / (obs_rms["var"].sqrt() + 1e-8)
        act = policy.get_action({"state": st, "rgb": rgb})
        return act.squeeze(0).cpu().numpy()

    env = VisualStateEnv(mode=4, simulated=simulated, has_gripper=True, imsize=imshape)
    obs = env.reset()
    done = False

    for _ in range(max_steps):
        if not done:
            obs_pol = np.concatenate([obs['robot0_joint_pos'],\
                                        obs['robot0_joint_vel'],\
                                        obs['is_grasped'],\
                                        # obs['robot0_eef_pos_to_cube'],\
                                        # obs['object_to_target_pos'],\
                                    ])
            model_observation = {"state": obs_pol, "rgb": obs['rgb']}
            action = sample_action(model_observation)
            action[:6] *= 0.5
            obs, rew, done, info = env.step(action)

            # time.sleep(1)

    env.reset()
    env.close()

    return done


if __name__ == "__main__":
    # Run the visual task demo
    simulated = False  # Set to False for real robot
    task_name = "LiftCan"  # Choose from valid tasks
    print_info = False  # Print information during the demo
    max_steps = 50  # Maximum number of steps in the demo
    save_obs = True  # Save observations for debugging

    visual_task_demo_simba(simulated, task_name, print_info, max_steps, save_obs)
    # visual_task_demo_lmap(simulated, print_info, max_steps)