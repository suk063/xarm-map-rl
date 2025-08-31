"""
This script runs the xArm6 robot to pick up a green cube and place it in a bin.
"""

import os
import time 
import numpy as np
import torch
from envs.pick_place import PickPlace
from utils.debug_utils import format_ndarray

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
policies_dir = os.path.join(parent_dir, "policies")

def demo_with_pickplace_bc(simulated=True, use_pose=False, save_obs=False, print_info=True, max_steps=250):
    from utils.policy_robosuite_utils import ActorCustomInit

    obs_dim = 21
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCustomInit(obs_dim, act_dim, 3, 256).to(device)
    model.load_state_dict(torch.load(policies_dir + "/pickplace_actor_bc.pt"))

    env = PickPlace(mode=4, simulated=simulated, use_pose=use_pose)    
    obs = env.reset()

    print(f"\nTASK: Attempting to pick cube from {obs['cube_pos']} and place in bin at {obs['bin_pos']}\n")

    use_gripper_policy = False

    obs_list = []
    time_list = []
    actions = []
    cube_pos_list = []
    eef_pos_list = []
    
    start_time = time.time()
    for _ in range(max_steps):
                                                                           # Corresponding robosuite obs keys:
        obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],             # 'robot0_joint_pos_cos'
                                obs['robot0_joint_pos_sin'],               # 'robot0_joint_pos_sin'
                                obs['robot0_gripper_width'],               # 'robot0_gripper_width'
                                obs['cube_to_robot0_eef_pos'],             # 'Bread_to_robot0_eef_pos'
                                obs['cube_to_bin_pos'],                    # 'Bread_to_Bread_bin_pos'
                                obs['robot0_touch']])                      # 'robot0_touch'
        action = model(torch.from_numpy(obs_pol).unsqueeze(0).float().to(device)).cpu().data.numpy().flatten()
        action_scaled = action * 0.3

        # print(f"Gripper width: {obs['robot0_gripper_width']}")
        if save_obs:
            curr_time = time.time() - start_time
            time_list.append(curr_time)
            obs_list.append(obs_pol)
            actions.append(action)
            cube_pos_list.append(obs['cube_pos'])
            eef_pos_list.append(obs['robot0_eef_pos'])
            
            if not os.path.exists("outputs/compare_sim_real"):
                os.makedirs("outputs/compare_sim_real")
            np.savez("outputs/compare_sim_real/obs_xarm_pickplace_bc.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cube_pos_list, eef_pos_list=eef_pos_list)
        
        dist_cube_to_eef = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
        dist_cube_to_bin = np.linalg.norm(obs['cube_to_bin_pos'])
        dist_bin_to_eef = np.linalg.norm(obs['bin_to_robot0_eef_pos'])
        if use_gripper_policy:
            obs, rew, done, info = env.step(action_scaled)
        else:
            obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))

        # Logging task stages and creating synthetic observations for simulated runs
        if env.curr_stage == 1 and print_info:
            print(f"Reaching Cube | Distance of arm to cube: {format_ndarray(obs['cube_to_robot0_eef_pos']*1000)} mm, norm: {dist_cube_to_eef*1000} mm")
        if env.curr_stage == 2:
            # Stage 2: Bringing cube to bin and placing it in
            if simulated:
                obs['robot0_touch'] = np.array([1, 1])
                obs['cube_to_robot0_eef_pos'] = np.zeros(3)
                obs['cube_to_bin_pos'] = obs['bin_pos'] - obs['robot0_eef_pos']
                dist_cube_to_bin = np.linalg.norm(obs['cube_to_bin_pos'])
                if env.cube_over_bin(cube_pos=obs['robot0_eef_pos']):
                    print("Cube over target bin")
                if env.cube_over_bin(cube_pos=obs['robot0_eef_pos']) and action_scaled[-1] < 0:
                    env.curr_stage += 1
            use_gripper_policy = True
            if print_info:
                print(f"Bringing Cube to Bin | Distance of cube to bin: {format_ndarray(obs['cube_to_bin_pos']*1000)} mm, norm: {dist_cube_to_bin*1000}, {env.cube_over_bin()}")
        
        if env.curr_stage == 3:
            # Stage 3: Lifting arm after placing cube
            if simulated:
                obs['cube_to_bin_pos'] = np.zeros(3)
                obs['cube_to_robot0_eef_pos'] = obs["bin_to_robot0_eef_pos"]
                obs['robot0_touch'] = np.array([0, 0])
                if np.tanh(10.0 * dist_bin_to_eef) > 0.4:
                    print("\nVirtual cube placed in bin and arm lifted!\n")
                    done = True
                    break
            use_gripper_policy = False
            if print_info:
                print(f"Placing Cube in Bin | Touch: {obs['robot0_touch']}, Distance of arm to bin: {format_ndarray(obs['bin_to_robot0_eef_pos']*1000)} mm, norm: {dist_bin_to_eef*1000}")
        
        if (obs["robot0_touch"] > 0.9).all():
            obs["robot0_gripper_width"] = np.array([0.0225])
        else:
            obs['robot0_gripper_width'] = np.array([0.0913])
        
        if done:
            action = np.zeros(7)
        # if done:
        #     print("\nCube placed in bin!\n")
        #     break

    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_xarm_pickplace_bc.npz")
    return done


def demo_with_pickplace_irl(simulated=True, use_pose=False, save_obs=False, print_info=True, max_steps=2000):
    from utils.policy_robosuite_utils import ActorSAC

    obs_dim = 26
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/pickplace_actor_irl.pt"))

    env = PickPlace(mode=4, simulated=simulated, use_pose=use_pose)    
    obs = env.reset()

    print(f"\nTASK: Attempting to pick cube from {obs['cube_pos']} and place in bin at {obs['bin_pos']}\n")

    use_gripper_policy = False

    obs_list = []
    time_list = []
    actions = []
    cube_pos_list = []
    eef_pos_list = []
    
    start_time = time.time()
    for _ in range(max_steps):
                                                                           # Corresponding robosuite obs keys:
        obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],             # 'robot0_joint_pos_cos'
                                obs['robot0_joint_pos_sin'],               # 'robot0_joint_pos_sin'
                                obs['robot0_joint_vel'],                   # 'robot0_joint_vel'
                                obs['cube_to_robot0_eef_pos'],             # 'Bread_to_robot0_eef_pos'
                                # obs['cube_to_robot0_eef_quat'],          # 'Bread_to_robot0_eef_quat'
                                obs['cube_to_bin_pos'],                    # 'Bread_to_Bread_bin_pos'
                                # obs['bin_to_robot0_eef_pos'],              # 'Bread_bin_to_robot0_eef_pos'
                                obs['robot0_touch']])   # (1, 26)          # 'robot0_touch'
        action = model.sample_action(obs_pol)
        action_scaled = action*0.5

        if save_obs:
            curr_time = time.time() - start_time
            time_list.append(curr_time)
            obs_list.append(obs_pol)
            actions.append(action)
            cube_pos_list.append(obs['cube_pos'])
            eef_pos_list.append(obs['robot0_eef_pos'])
            
            if not os.path.exists("outputs/compare_sim_real"):
                os.makedirs("outputs/compare_sim_real")
            np.savez("outputs/compare_sim_real/obs_xarm_pickplace.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cube_pos_list, eef_pos_list=eef_pos_list)
        
        dist_cube_to_eef = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
        dist_cube_to_bin = np.linalg.norm(obs['cube_to_bin_pos'])
        dist_bin_to_eef = np.linalg.norm(obs['bin_to_robot0_eef_pos'])
        if use_gripper_policy:
            obs, rew, done, info = env.step(action_scaled)
        else:
            obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))

        # Logging task stages and creating synthetic observations for simulated runs
        if env.curr_stage == 1 and print_info:
            print(f"Reaching Cube | Distance of arm to cube: {format_ndarray(obs['cube_to_robot0_eef_pos']*1000)} mm, norm: {dist_cube_to_eef*1000} mm")
        if env.curr_stage == 2:
            # Stage 2: Bringing cube to bin and placing it in
            if simulated:
                obs['robot0_touch'] = np.array([1, 1])
                obs['cube_to_robot0_eef_pos'] = np.zeros(3)
                obs['cube_to_bin_pos'] = obs['bin_pos'] - obs['robot0_eef_pos']
                dist_cube_to_bin = np.linalg.norm(obs['cube_to_bin_pos'])
                if env.cube_over_bin(cube_pos=obs['robot0_eef_pos']):
                    print("Cube over target bin")
                if env.cube_over_bin(cube_pos=obs['robot0_eef_pos']) and action_scaled[-1] < 0:
                    env.curr_stage += 1
            if obs['is_grasped']:
                obs['robot0_touch'] = np.array([1, 1])
            use_gripper_policy = True
            if print_info:
                print(f"Bringing Cube to Bin | Distance of cube to bin: {format_ndarray(obs['cube_to_bin_pos']*1000)} mm, norm: {dist_cube_to_bin*1000}")
        if env.curr_stage == 3:
            # Stage 3: Lifting arm after placing cube
            if simulated:
                obs['cube_to_bin_pos'] = np.zeros(3)
                obs['cube_to_robot0_eef_pos'] = obs["bin_to_robot0_eef_pos"]
                obs['robot0_touch'] = np.array([0, 0])
                if np.tanh(10.0 * dist_bin_to_eef) > 0.4:
                    print("\nVirtual cube placed in bin and arm lifted!\n")
                    done = True
                    break
            use_gripper_policy = False
            if print_info:
                print(f"Placing Cube in Bin | Touch: {obs['robot0_touch']}, Distance of arm to bin: {format_ndarray(obs['bin_to_robot0_eef_pos']*1000)} mm, norm: {dist_bin_to_eef*1000}")
            
        if done:
            print("\nCube placed in bin!\n")
            break

    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_xarm_pickplace.npz")
    return done


if __name__ == '__main__':
    
    ## Tips for cube and bin placement in real:
        # In robosuite, robot base is at [-0.5, -0.1, 0.912] (check env.env.robots[0].base_pos)
        # Pickplace policy is trained with Bread at approximately [0.1, -0.25, 0.845] in robosuite (check bin1_pos in pick_place.py)
        # Also in robosuite, the target bin placement for Bread is [0.1975, 0.1575, 0.8] (check env.env.target_bin_placements[1])
        # So conversion from robosuite to real (relative to base) is: 
        #          Bread (40mm cube) position:        [0.1, -0.25, 0.845] - [-0.5, -0.1, 0.912] = [0.6, -0.15, -0.067]
        #                        Bin position:      [0.1975, 0.1575, 0.8] - [-0.5, -0.1, 0.912] = [0.6975, 0.2575, -0.112] 
    
    # demo_with_pickplace_irl(simulated=False, save_obs=True)


    ## Tips for cube and bin placement in real:
        # In robosuite, robot base is at [-0.407, 0, 1.083] (check env.env.robots[0].base_pos)
        # Pickplace policy is trained with Bread at approximately [0.1, -0.25, 0.845] in robosuite (check bin1_pos in pick_place.py)
        # Also in robosuite, the target bin placement for Bread is [0.0475, 0.1575, 0.90] (check env.env.target_bin_placements[1])
        # So conversion from robosuite to real (relative to base) is: 
        #          Bread (40mm cube) position:        [0, -0.15, 0.945] - [-0.407, 0, 1.083] = [0.407, -0.15, -0.138]
        #                        Bin position:      [0.0475, 0.1575, 0.90] - [-0.407, 0, 1.083] = [0.4545, 0.1575, -0.183] 
    demo_with_pickplace_bc(simulated=False, save_obs=True)
