"""
This script runs the xArm6 robot to reach a target position with its end-effector
"""

import numpy as np
from stable_baselines3 import SAC
import torch
from utils.policy_robosuite_utils import ActorSAC, Actor

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
policies_dir = os.path.join(parent_dir, "policies")
sys.path.append(parent_dir)

from envs.reach import Reach
from policies.human_policy import ReachPolicy

def test_eef_control():

    env = Reach(mode=0, target_pos=[0.3, 0.1, 0.1], simulated=False, has_gripper=True)

    policy = ReachPolicy(env)
    policy.reset()


    ## TODO: load policy and run this in closed-loop
    T = 1000
    obs = env.reset()
    for t in range(T):
        action, completed = policy.predict(obs)
        obs, rew, done, info = env.step(action)

        print("target to eef pos:", obs['target_to_robot0_eef_pos'])

        if env._check_success():
            print("Success")
            break

    # Clean up
    env.close()


def test_joint_vel_policy(policy_type="SB3", simulated=False, n_episodes=10):
    '''
        policy_type: "Custom" or "SB3" implementation of SAC
    '''
    if policy_type == "SB3":
        controller = "JOINT_VELOCITY"
        # model = SAC.load(f"/home/erl-tianyu/robosuite/logs_jv_no_gripper/Reach_{controller}/best_model")
        model = SAC.load(f"/home/erl-tianyu/robosuite/logs_jv_no_gripper_damp_0.1_fric_0.01/Reach_{controller}/best_model")
    elif policy_type == "Custom":
        obs_dim = 21
        act_dim = 6
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
        model.load_state_dict(torch.load(policies_dir + "/reach_actor_no_gripper.pt"))

    env = Reach(mode=4, simulated=simulated, target_pos=[0.56, 0, 0.17])    # Target corresponding to [0, 0, 1] in robosuite
    obs = env.reset()
    
    avg_errors_mm = []
    for _ in range(n_episodes):
        for t in range(200):
            obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],\
                                    obs['robot0_joint_pos_sin'],\
                                    obs['robot0_joint_vel'],\
                                    -obs['robot0_eef_to_target_pos']])   # (1, 21)
            if policy_type == "SB3":
                action, _states = model.predict(obs_pol, deterministic=True) 
            elif policy_type == "Custom":
                action = model.sample_action(obs_pol)

            # APPLY A FRACTION OF ACTION!!!! Directly applying action leads to dangerous speeds
            action *= 0.5
            obs, rew, done, info = env.step(np.append(action, 1))

            print(f"Distance to target: {obs['robot0_eef_to_target_pos']*1000} mm")
            if policy_type == "SB3":
                done = (np.linalg.norm(obs['robot0_eef_to_target_pos']) < 0.015)
            elif policy_type == "Custom":
                done = (np.linalg.norm(obs['robot0_eef_to_target_pos']) < 0.004)
            
            if done:
                avg_errors_mm.append(obs['robot0_eef_to_target_pos']*1000)
                env.step(np.append([0]*6, -0.275))
                env.reset()
                break
    env.close()
    print(f"Average error across 10 runs with {policy_type} policy: \n{np.mean(np.array(avg_errors_mm), axis=0)} mm")


def test_joint_vel_policy_with_gripper(simulated=False, n_episodes=10):
    obs_dim = 21
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/reach_actor_with_gripper.pt"))

    offsets = [140] # mm
    error_dict = {}
    failed_counts = {offset:0 for offset in offsets}
    for offset in offsets:
        print(f"\nTesting Offset: {offset} mm")
        env = Reach(mode=4, simulated=simulated, target_pos=[0.56, 0, 0.17], \
                    has_gripper=True)
        
        avg_errors_mm = []
        for _ in range(n_episodes):
            obs = env.reset()
            for t in range(200):
                obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],\
                                        obs['robot0_joint_pos_sin'],\
                                        obs['robot0_joint_vel'],\
                                        obs['target_to_robot0_eef_pos']])   # (1, 21)
                action = model.sample_action(obs_pol)

                # APPLY A FRACTION OF ACTION!!!! Directly applying action leads to dangerous speeds
                action *= 0.5
                obs, rew, done, info = env.step(np.append(action[:6], 1))

                done = (np.linalg.norm(obs['target_to_robot0_eef_pos']) < 0.01)
                
                if done:
                    avg_errors_mm.append(obs['target_to_robot0_eef_pos']*1000)
                    print(f"Distance to target: {obs['target_to_robot0_eef_pos']*1000} mm")
                    env.step(np.append([0]*6, -0.275))
                    env.reset()
                    break
            if not done:
                print("Failed to reach target")
                failed_counts[offset] += 1
        env.close()
        error_dict[offset] = np.mean(np.array(avg_errors_mm), axis=0)
        print(f"Average error across {n_episodes} runs for offset {offset}: \n{error_dict[offset]} mm")

    print("\nSUMMARY:\nAverage Errors for all offsets are:")
    for k,v in error_dict.items():
        print(f"Offset {k} mm:\nAverage Error: {v} mm, Error Norm: {np.linalg.norm(v)} mm, Failed Counts: {failed_counts[k]}")


def fixed_target_drift_test(simulated=True):
    obs_dim = 21
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/reach_actor_with_gripper.pt"))

    env = Reach(mode=4, simulated=simulated, target_pos=[0.56, 0, 0.17], \
                has_gripper=True)
    gripper_action = 1
    
    avg_errors_mm = []
    obs = env.reset()
    for t in range(1000):
        obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],\
                                obs['robot0_joint_pos_sin'],\
                                obs['robot0_joint_vel'],\
                                obs['target_to_robot0_eef_pos']])   # (1, 21)
        action = model.sample_action(obs_pol)

        # APPLY A FRACTION OF ACTION!!!! Directly applying action leads to dangerous speeds
        action *= 0.5
        obs, rew, done, info = env.step(np.append(action[:6], gripper_action))

        done = (np.linalg.norm(obs['target_to_robot0_eef_pos']) < 0.01)
        
        print(f"Distance to target: {obs['target_to_robot0_eef_pos']*1000} mm, Norm: {np.linalg.norm(obs['target_to_robot0_eef_pos']*1000)} mm")
        if done:
            avg_errors_mm.append(obs['target_to_robot0_eef_pos']*1000)
            print(f"Distance to target: {obs['target_to_robot0_eef_pos']*1000} mm")
            gripper_action = -0.275         # Closing upto size of cube
            env.step(np.append([0]*6, gripper_action))
            # env.reset()
        else:
            gripper_action = 1
    env.close()
    

def test_joint_vel_policy_with_camera(simulated=False, use_gripper=False, n_episodes=10):
    obs_dim = 21
    act_dim = 7 if use_gripper else 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
    if use_gripper:
        policy_name = "reach_actor_with_gripper.pt"
    else:
        policy_name = "reach_actor_no_gripper.pt"
    model.load_state_dict(torch.load(policies_dir + f"/{policy_name}"))

    gripper_action = 1

    env = Reach(mode=4, simulated=simulated, target_pos=None, has_gripper=use_gripper)    
    obs = env.reset()

    avg_errors_mm = []
    failed_counts = 0
    for _ in range(n_episodes):
        dist = np.linalg.norm(obs['target_to_robot0_eef_pos'])
        for t in range(200):
            print(f"Distance to target: {dist*1000} mm")
            obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],\
                                    obs['robot0_joint_pos_sin'],\
                                    obs['robot0_joint_vel'],\
                                    obs['target_to_robot0_eef_pos']])   # (1, 21)
            action = model.sample_action(obs_pol)

            # APPLY A FRACTION OF ACTION!!!! Directly applying action leads to dangerous speeds
            action *= 0.5
            obs, rew, done, info = env.step(np.append(action[:6], gripper_action))

            dist = np.linalg.norm(obs['target_to_robot0_eef_pos'])
            done = dist < 0.01
            if done:
                print("Reached target!")
                avg_errors_mm.append(obs['target_to_robot0_eef_pos']*1000)
                print(f"Distance to target: {dist*1000} mm")
                gripper_action = -0.275         # Closing upto size of cube
                env.step(np.append([0]*6, gripper_action))
                env.reset()
                break
            else:
                gripper_action = 1
        if not done:
            print("Failed to reach target")
            failed_counts += 1
    env.reset()
    env.close()
    print(f"Average error across {n_episodes} runs with camera: \n{np.mean(np.array(avg_errors_mm), axis=0)} mm, Norm: {np.linalg.norm(np.mean(np.array(avg_errors_mm), axis=0))} mm")
    print(f"Failed counts: {failed_counts}")


def full_reach_demo(simulated=True, target_src="camera", episodic=False, print_info=True, max_steps=2000):
    obs_dim = 21
    act_dim = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = ActorSAC(obs_dim, act_dim, 3, 512, device).to(device)
    model = Actor(obs_dim, act_dim, 3, 512, device).to(device)
    model.load_state_dict(torch.load(policies_dir + "/reach_actor_with_gripper.pt"))

    gripper_action = 1
    error_threshold = 0.01

    env = Reach(mode=4, simulated=simulated, target_src=target_src, has_gripper=True)    
    obs = env.reset()
    dist = np.linalg.norm(obs['target_to_robot0_eef_pos'])
    done = False

    for _ in range(max_steps):
        if not done:
            if print_info:
                print(f"Distance to target: {dist*1000} mm, target_pos: {obs['target_pos']*1000} mm")
            obs_pol = np.concatenate([obs['robot0_joint_pos_cos'],\
                                    obs['robot0_joint_pos_sin'],\
                                    obs['robot0_joint_vel'],\
                                    obs['target_to_robot0_eef_pos']])   # (1, 21)
            action = model.sample_action(obs_pol)
            action *= 0.5
            obs, rew, done, info = env.step(np.append(action[:6], gripper_action))

        dist = np.linalg.norm(obs['target_to_robot0_eef_pos'])
        done = dist < error_threshold
        if done:
            print("Reached target!")
            if print_info:
                print(f"Distance to target: {dist*1000} mm, target_pos: {obs['target_pos']*1000} mm")
            if episodic:
                break
            gripper_action = -0.275         # Closing upto size of cube
            error_threshold = 0.015
            obs, rew, done, info = env.step(np.append([0]*6, gripper_action))
        else:
            gripper_action = 1
            error_threshold = 0.01
    env.reset()
    env.close()
    return done


if __name__ == '__main__':
    # test_eef_control()
    # test_joint_vel_policy(policy_type="SB3", simulated=True)
    # test_joint_vel_policy_with_gripper(simulated=True)
    # fixed_target_drift_test(simulated=True)
    # test_joint_vel_policy_with_camera(simulated=True, use_gripper=True)

    full_reach_demo(simulated=True, target_src="camera", episodic=False)