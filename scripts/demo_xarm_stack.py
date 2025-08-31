"""
This script runs the xArm6 robot to pick up a red cube and place it on top of a green cube.
"""

import os
import time 
import numpy as np
import torch
from envs.stack import Stack
from utils.debug_utils import format_ndarray

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
policies_dir = os.path.join(parent_dir, "policies")


def demo_with_stack_irl(simulated=True, save_obs=False, print_info=True, max_steps=2000):
    from utils.policy_robosuite_utils import ActorSAC

    obs_dim = 29
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/stack_actor_irl.pt"))

    env = Stack(mode=4, simulated=simulated)
    obs = env.reset()

    print(f"\nTASK: Attempting to pick red cube from {obs['cubeA_pos']} and place it on top of green cube {obs['cubeB_pos']}\n")

    use_gripper_policy = False

    obs_list = []
    time_list = []
    actions = []
    cubeA_pos_list = []
    eef_pos_list = []
    
    start_time = time.time()
    i = 0 
    while i < max_steps:
                                                                           # Corresponding robosuite obs keys:
        obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],             # 'robot0_joint_pos_cos'
                                obs['robot0_joint_pos_sin'],               # 'robot0_joint_pos_sin'
                                obs['robot0_joint_vel'],                   # 'robot0_joint_vel'
                                obs['robot0_eef_to_cubeA_pos'],            # 'gripper_to_cubeA'
                                obs['robot0_eef_to_cubeB_pos'],            # 'gripper_to_cubeB'
                                obs['cubeA_to_cubeB_pos'],                 # 'cubeA_to_cubeB'
                                obs['robot0_touch']])   # (1, 29)          # 'robot0_touch'
        action = model.sample_action(obs_pol)
        action_scaled = action*0.5

        if save_obs:
            curr_time = time.time() - start_time
            time_list.append(curr_time)
            obs_list.append(obs_pol)
            actions.append(action)
            cubeA_pos_list.append(obs['cubeA_pos'])
            eef_pos_list.append(obs['robot0_eef_pos'])
            
            if not os.path.exists("outputs/compare_sim_real"):
                os.makedirs("outputs/compare_sim_real")
            np.savez("outputs/compare_sim_real/obs_xarm_stack.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cubeA_pos_list, eef_pos_list=eef_pos_list)
        
        dist_eef_to_cubeA = np.linalg.norm(obs['robot0_eef_to_cubeA_pos'])
        dist_cubeA_to_cubeB = np.linalg.norm(obs['cubeA_to_cubeB_pos'])
        dist_eef_to_cubeB = np.linalg.norm(obs['robot0_eef_to_cubeB_pos'])
        if use_gripper_policy:
            obs, rew, done, info = env.step(action_scaled)
        else:
            obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))

        # Logging task stages and creating synthetic observations for simulated runs
        if env.curr_stage == 1 and print_info:
            use_gripper_policy = False
            print(f"Reaching red cube | Distance of arm to red cube: {format_ndarray(obs['robot0_eef_to_cubeA_pos']*1000)} mm, norm: {dist_eef_to_cubeA*1000} mm")
        if env.curr_stage >= 2:
            # Stage 2: Bringing red cube to green cube, Stage 3: Stacking red cube on green cube
            if simulated:
                obs['robot0_touch'] = np.array([1, 1])
                obs['robot0_eef_to_cubeA_pos'] = np.zeros(3)
                obs['cubeA_to_cubeB_pos'] = obs['cubeB_pos'] - obs['robot0_eef_pos']
                dist_cubeA_to_cubeB = np.linalg.norm(obs['cubeA_to_cubeB_pos'])
                if env.curr_stage == 2 and env.cubeA_over_cubeB(cubeA_pos=obs['robot0_eef_pos']):
                        print("\nCubeA over CubeB!\n")
                        env.curr_stage += 1
                if env.curr_stage == 3 and action_scaled[-1] < 0:
                    print("\nVirtual red cube stacked on top of virtual green cube\n")
                    break
            use_gripper_policy = True if not env.tumble_check else False
            if print_info:
                print(f"Bringing & stacking red cube on green cube | Distance of cubeA to cubeB: {format_ndarray(obs['cubeA_to_cubeB_pos']*1000)} mm, norm: {dist_cubeA_to_cubeB*1000}")
            
        if done:
            print("\nRed cube stacked on top of green cube!\n")
            break
        
        if env.regrasp_attempt:
            i -= 80
        i += 1

    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_xarm_stack.npz")
    return done


if __name__ == '__main__':
    
    ## Tips for cube placements in real:
        # In robosuite, robot base is at [-0.56, 0, 0.912] (check env.env.robots[0].base_pos)
        # Stack policy is trained with CubeA and CubeB at [-0.08 to 0.08, -0.08 to 0.08, 0.8] in robosuite (check placement_initializer in stack.py)
        # So conversion from robosuite to real (relative to base) is: 
        #               central position:                [0, 0, 0.8] - [-0.56, 0, 0.912] = [0.56, 0, -0.112]
        #                   min position:        [-0.08, -0.08, 0.8] - [-0.56, 0, 0.912] = [0.48, -0.08, -0.112]
        #                   max position:          [0.08, 0.08, 0.8] - [-0.56, 0, 0.912] = [0.64, 0.08, -0.112]
    
    demo_with_stack_irl(simulated=True, save_obs=True)
