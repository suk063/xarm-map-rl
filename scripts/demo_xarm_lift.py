"""
This script runs the xArm6 robot to pick up a green cube
"""

import os
import time 
import numpy as np
import torch

import sys
import os

from envs.lift import Lift
from utils.debug_utils import format_ndarray

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
policies_dir = os.path.join(parent_dir, "policies")

def main():
    # The main thread contains the xArm control and 
    # camera/touch sensors are in separate threads
    print("mainThread pid:", os.getpid())

    env = Lift(mode=0, simulated=True, has_gripper=True, use_pose=True)

    ## TODO: load policy and run this in closed-loop
    T = 100
    obs = env.reset()
    for t in range(T):
        # action = policy.predict(obs)
        action = np.zeros(7)
        # action[0] = 1
        obs, rew, done, info = env.step(action)

        print("Cube pos:", obs['cube_pos'])
        print("eef pos", obs['robot0_eef_pos'])
        print("Touch:", obs['robot0_touch'])
        print("Gripper:", obs['robot0_gripper_width'])
        print("Cube pose quat:", obs['cube_quat'])

        if env._check_success():
            break
            print("Success")

    # Clean up
    env.close()


def demo_with_reach(simulated=True, use_pose=False, tiny_cube=False):
    from utils.policy_robosuite_utils import Actor

    obs_dim = 21
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Actor(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/reach_actor_with_gripper.pt"))

    gripper_action = -1
    error_threshold = 0.01

    env = Lift(mode=4, simulated=simulated, use_pose=use_pose)    
    obs = env.reset()
    
    # 1) Reach above the cube
    target_pos = obs['cube_pos'] + np.array([0, 0, 0.17])
    dist = np.linalg.norm(obs['robot0_eef_pos'] - target_pos)
    while True:
        # Target to reach is 15 cm above cube
        target_pos = obs['cube_pos'] + np.array([0, 0, 0.17])
        # print(f"Distance to target: {dist*1000} mm, target_pos: {target_pos*1000} mm")
        obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],\
                                obs['robot0_joint_pos_sin'],\
                                obs['robot0_joint_vel'],\
                                obs['robot0_eef_pos'] - target_pos])   # (1, 21)
        action = model.sample_action(obs_pol)
        action *= 0.5
        obs, rew, done, info = env.step(np.append(action[:6], gripper_action))

        dist = np.linalg.norm(obs['robot0_eef_pos'] - target_pos)
        reached = dist < error_threshold
        if reached:
            print("\nReached above cube!\n")
            print(f"Distance to target: {dist*1000} mm, target_pos: {target_pos*1000} mm")
            obs, rew, done, info = env.step(np.append([0]*6, gripper_action))
            break
    
    # 2) Reach close the cube (10cm below current position)
    env.switch_mode(0)
    gripper_action = -0.5
    if tiny_cube:
        z_down = -13
    else:
        z_down = -14
    env.step(np.append([0, 0, z_down, 0, 0, 0], gripper_action), speed=50)
    print("\nReached close to cube!\n")

    # 3) Close the gripper till cube is grasped
    if tiny_cube:
        gripper_action = 0.8
        addval = 0.066
    else:
        gripper_action = 0.1
        addval = gripper_action
    env.step(np.append([0]*6, gripper_action))
    while (obs['robot0_touch'] < 0.9).any():
        if not tiny_cube:
            addval = addval*2/3          # Sum of GP = 0.1/(1-(2/3)) = 0.3
        gripper_action += addval
        obs, _, _, _ = env.step(np.append([0]*6, gripper_action))
        print("Touch:", obs['robot0_touch'], "Gripper:", gripper_action)
    print("\nCube grasped!\n")

    # 4) Lift the cube
    while not done:
        obs, _, done, _ = env.step(np.append([0, 0, 5, 0, 0, 0], gripper_action), speed=30)
        print("Cube pos:", obs['cube_pos'], "fsr:", obs['robot0_touch'])
    print("\nCube lifted!\n")

    env.close()


def demo_with_lift(simulated=True, use_pose=False):
    from utils.policy_robosuite_utils import Actor

    obs_dim = 27
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Actor(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/lift_actor.pt"))

    env = Lift(mode=4, simulated=simulated, use_pose=use_pose)    
    obs = env.reset()

    target_pos = obs['cube_pos'] + np.array([0, 0, 0.2])
    dist = np.linalg.norm(obs['robot0_eef_pos'] - target_pos)
    while True:
        # Target is to bring cube 20 cm above its position
        target_pos = obs['cube_pos'] + np.array([0, 0, 0.2])

        obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],             # 'robot0_joint_pos_cos'
                                obs['robot0_joint_pos_sin'],               # 'robot0_joint_pos_sin'
                                obs['robot0_joint_vel'],                   # 'robot0_joint_vel'
                                obs['robot0_gripper_width'],               # 'robot0_gripper_width'
                                obs['cube_to_robot0_eef_pos'],             # 'cube_to_robot0_eef_pos'
                                target_pos - obs['cube_pos'],              # 'cube_to_target_pos'
                                obs['robot0_touch']])   # (1, 32)          # 'robot0_touch'
        action = model.sample_action(obs_pol)
        action *= 0.3
        
        # print(f"obs['robot0_gripper_width']: {obs['robot0_gripper_width']}")
        # print(f"gripper action: {action[6]}")
        dist = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
        # print(f"Distance to cube: {obs['cube_to_robot0_eef_pos']*1000} mm, norm: {dist*1000} mm")
        if dist < 0.1:
            print(f"\nNow using gripper action from policy")
            action[6] = max(action[6], -0.32)
            obs, rew, done, info = env.step(action)
        else:
            obs, rew, done, info = env.step(np.append(action[:6], -1))
            obs['robot0_gripper_width'] = env.xarm.get_gripper_qpos()
        if done:
            print("\nCube Lifted!\n")
            break    

def demo_with_lift_bc(save_obs=False, simulated=True, print_info=False, max_steps=2000, maniskill_policy=False):
    from utils.policy_robosuite_utils import ActorCustomInit, ActorBCManiSkill

    obs_dim = 21
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCustomInit(obs_dim, act_dim, 3, 256).to(device)

    policy_name = "lift_actor_bc_varypos.pt"
    if maniskill_policy:
        obs_dim = 25
        policy_name = "lift_actor_bc_varypos_mnskill.pt"
        model = ActorBCManiSkill(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load(policies_dir + f"/{policy_name}")["actor"])
    print(f"Testing policy {policy_name}")

    env = Lift(mode=4, simulated=simulated)
    obs = env.reset()
    print(f"\nTASK: Attempting to lift cube at {obs['cube_pos']}")

    print(f"Set robosuite cube to = {np.array([-0.2 ,  0.  ,  1.05]) - obs['cube_to_robot0_eef_pos']}")

    target_pos = obs['cube_pos'] + np.array([0, 0, 0.2])
    dist = np.linalg.norm(obs['robot0_eef_pos'] - target_pos)
    use_gripper_policy = False
    enable_fake_touch = False

    obs_list = []
    time_list = []
    actions = []
    cube_pos_list = []
    eef_pos_list = []

    obs_keys_order = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 
                      'robot0_gripper_width', 'cube_to_robot0_eef_pos', 'cube_to_target_pos', 'robot0_touch']
    if maniskill_policy:
        obs_keys_order = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 
                          'is_grasped', 'cube_to_robot0_eef_pos', 'cube_to_target_pos']
        enable_fake_touch = True

    start_time = time.time()
    for _ in range(max_steps):        
        # Virtual gripper width to match robosuite observations
        if (obs["robot0_touch"] > 0.9).all():
            obs["robot0_gripper_width"] = np.array([0.0211])
        else:
            obs['robot0_gripper_width'] = np.array([0.0913])
        
        obs_pol = []
        for k in obs_keys_order:
            # print(f"{k}: {obs[k]}")
            obs_pol.extend(obs[k])
        obs_pol = np.array(obs_pol)
        
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
                os.makedirs("outputs/compare_sim_real/")
            np.savez("outputs/compare_sim_real/obs_xarm_lift_bc.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cube_pos_list, eef_pos_list=eef_pos_list)

        dist_cube_to_eef = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
        dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
        if dist_cube_to_eef < 0.05 or use_gripper_policy:
            use_gripper_policy = True
            obs, rew, done, info = env.step(action_scaled)
        else:
            obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))

        if enable_fake_touch:
            obs['robot0_touch'] = np.array([1, 1])

        # Synthetic observations for simulated runs
        if simulated and (dist_cube_to_eef < 0.05 or enable_fake_touch):
            obs['cube_to_robot0_eef_pos'] = np.zeros(3)
            obs['cube_to_target_pos'] = obs['target_pos'] - obs['robot0_eef_pos']
            if not enable_fake_touch:
                print(f"Virtual Cube Grasped")
            enable_fake_touch = True
            dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
        elif print_info:
            print(f"Distance to cube: {format_ndarray(obs['cube_to_robot0_eef_pos']*1000)} mm, norm: {dist_cube_to_eef*1000} mm") 

        if done:
            print("\nCube Lifted!\n")
            break
        elif simulated and dist_cube_to_target < 0.03:
            done = True
            print("\nVirtual Cube Lifted!\n")
            break
    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_xarm_lift_bc.npz")
    
    # env.xarm.open_gripper()
    return done


def demo_with_lift_irl(simulated=True, use_pose=False, save_obs=False, print_info=True, max_steps=2000):
    from utils.policy_robosuite_utils import ActorSAC

    obs_dim = 27
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/lift_actor_irl_noquat.pt"))
    print(f"Testing policy lift_actor_irl_noquat.pt")

    env = Lift(mode=4, simulated=simulated, use_pose=use_pose)    
    obs = env.reset()
    print(f"\nTASK: Attempting to lift cube at {obs['cube_pos']}")

    target_pos = obs['cube_pos'] + np.array([0, 0, 0.2])
    dist = np.linalg.norm(obs['robot0_eef_pos'] - target_pos)
    use_gripper_policy = False
    enable_fake_touch = False

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
                                obs['robot0_gripper_width'],               # 'robot0_gripper_width'
                                obs['cube_to_robot0_eef_pos'],             # 'cube_to_robot0_eef_pos'
                                # obs['cube_to_robot0_eef_quat'],            # 'cube_to_robot0_eef_quat'
                                obs['cube_to_target_pos'],                 # 'cube_to_target_pos'
                                obs['robot0_touch']])   # (1, 32)          # 'robot0_touch'
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
                os.makedirs("outputs/compare_sim_real/")
            np.savez("outputs/compare_sim_real/obs_xarm_lift.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cube_pos_list, eef_pos_list=eef_pos_list)
        
        dist_cube_to_eef = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
        dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
        if dist_cube_to_eef < 0.05 or use_gripper_policy:
            use_gripper_policy = True
            obs, rew, done, info = env.step(action_scaled)
        else:
            obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))

        # Synthetic observations for simulated runs
        if simulated and (dist_cube_to_eef < 0.05 or enable_fake_touch):
            obs['robot0_touch'] = np.array([1, 1])
            obs['cube_to_robot0_eef_pos'] = np.zeros(3)
            obs['cube_to_target_pos'] = obs['target_pos'] - obs['robot0_eef_pos']
            enable_fake_touch = True
            dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
        elif print_info:
            print(f"Distance to cube: {format_ndarray(obs['cube_to_robot0_eef_pos']*1000)} mm, norm: {dist_cube_to_eef*1000} mm") 
        
        # Virtual gripper width to match robosuite observations
        if (obs["robot0_touch"] > 0.9).all():
            obs["robot0_gripper_width"] = np.array([0.0225])
        else:
            obs['robot0_gripper_width'] = np.array([0.0913])
        
        if done:
            print("\nCube Lifted!\n")
            break
        elif simulated and dist_cube_to_target < 0.03:
            done = True
            print("\nVirtual Cube Lifted!\n")
            break
    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_xarm_lift.npz")
    
    # env.xarm.open_gripper()
    return done


def demo_with_lift_ppo(simulated=True, save_obs=False, print_info=True, max_steps=2000):
    from utils.policy_robosuite_utils import ActorPPOManiSkill

    obs_dim = 19
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorPPOManiSkill(obs_dim, act_dim, device=device)
    model.load_state_dict(torch.load(policies_dir + "/lift_actor_ppo.pt"))
    print(f"Testing policy lift_actor_ppo.pt")

    env = Lift(mode=4, simulated=simulated, use_pose=False)    
    obs = env.reset()
    print(f"\nTASK: Attempting to lift cube at {obs['cube_pos']}")

    target_pos = obs['cube_pos'] + np.array([0, 0, 0.2])
    dist = np.linalg.norm(obs['robot0_eef_pos'] - target_pos)
    use_gripper_policy = False
    enable_fake_touch = False

    obs_list = []
    time_list = []
    actions = []
    cube_pos_list = []
    eef_pos_list = []

    start_time = time.time()
    for _ in range(max_steps):
                                                                           # Corresponding maniskill obs keys:
        obs_pol = np.concatenate([obs['robot0_joint_pos'],                  # 'qpos'[:6]
                                obs['robot0_joint_vel'],                    # 'qvel'[:6]
                                obs['is_grasped'],                          # 'is_grasped'
                                obs['cube_to_robot0_eef_pos'],              # 'obj_to_tcp_pos'
                                obs['cube_to_target_pos'],])                # 'obj_to_goal_pos'
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
                os.makedirs("outputs/compare_sim_real/")
            np.savez("outputs/compare_sim_real/obs_xarm_lift.npz", obs_list=obs_list, time_list=time_list, actions=actions, cube_pos_list=cube_pos_list, eef_pos_list=eef_pos_list)
        
        dist_cube_to_eef = np.linalg.norm(obs['cube_to_robot0_eef_pos'])
        dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
        if dist_cube_to_eef < 0.05 or use_gripper_policy:
            use_gripper_policy = True
            obs, rew, done, info = env.step(action_scaled)
        else:
            obs, rew, done, info = env.step(np.append(action_scaled[:6], -1))

        # Synthetic observations for simulated runs
        if simulated and (dist_cube_to_eef < 0.05 or enable_fake_touch):
            obs['robot0_touch'] = np.array([1, 1])
            obs['cube_to_robot0_eef_pos'] = np.zeros(3)
            obs['cube_to_target_pos'] = obs['target_pos'] - obs['robot0_eef_pos']
            obs['is_grasped'] = np.array([1])
            enable_fake_touch = True
            dist_cube_to_target = np.linalg.norm(obs['cube_to_target_pos'])
        elif print_info:
            print(f"Distance to cube: {format_ndarray(obs['cube_to_robot0_eef_pos']*1000)} mm, norm: {dist_cube_to_eef*1000} mm") 
        
        # Virtual gripper width to match robosuite observations
        if (obs["robot0_touch"] > 0.9).all():
            obs["robot0_gripper_width"] = np.array([0.0225])
        else:
            obs['robot0_gripper_width'] = np.array([0.0913])
        
        if done:
            print("\nCube Lifted!\n")
            break
        elif simulated and dist_cube_to_target < 0.03:
            done = True
            print("\nVirtual Cube Lifted!\n")
            break
    if save_obs:
        print(f"Saved {len(obs_list)} observations to outputs/compare_sim_real/obs_xarm_lift.npz")
    
    # env.xarm.open_gripper()
    return done


if __name__ == '__main__':
    # main()
    
    ## Tip for cube placement in real:
        # Lift policy is trained with approximately [0, 0, 0.825] in robosuite, meaning cube in real should be kept at [0.56, 0, -0.095]
        # NOTE: Use the 40mm green cube, not the 55mm
    
    # demo_with_lift_irl(simulated=False, save_obs=True)

    ## Tip for cube placement in real:
        # Lift policy is trained with approximately [0, 0, 0.938] in robosuite, meaning cube in real should be kept at [0.407, 0, -0.145]
    demo_with_lift_bc(simulated=False)
